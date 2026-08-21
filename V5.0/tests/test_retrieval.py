"""Retrieval tests — vector index, BM25, and RRF fusion.

Uses a deterministic stub embedder rather than the real model: these assert ranking and
fusion *logic*, and a 220MB ONNX load per test run would make the suite too slow to run
on every change. Retrieval quality against the real model and real corpus is measured by
the benchmark, which is the right tool for that question.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.types import Chunk  # noqa: E402
from retrieval.lexical import LexicalIndex, STOPWORDS, tokenize  # noqa: E402
from retrieval.retriever import RetrievalConfig, Retriever  # noqa: E402
from retrieval.vector_index import VectorIndex, _NumpyFlatIP  # noqa: E402


def make_chunk(i: int, text: str, **kw) -> Chunk:
    base = dict(chunk_id=f"c{i}", strategy="test", query_id=f"q{i}",
                passage_idx=0, source_lang="hi", char_len=len(text))
    base.update(kw)
    return Chunk(text=text, **base)


def unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype="float32")
    return arr / max(float(np.linalg.norm(arr)), 1e-12)


class StubEmbedder:
    """Maps text to a fixed vector by keyword, so ranking is fully predictable."""

    model_name = "stub"
    dim = 3

    TABLE = {
        "ganges": [1.0, 0.0, 0.0],
        "corporation": [0.0, 1.0, 0.0],
        "cricket": [0.0, 0.0, 1.0],
    }

    def _vec(self, text: str) -> np.ndarray:
        low = text.lower()
        for key, vec in self.TABLE.items():
            if key in low:
                return unit(vec)
        return unit([0.577, 0.577, 0.577])

    def embed_texts(self, texts, batch_size=256):
        return np.vstack([self._vec(t) for t in texts]).astype("float32")

    def embed_query(self, text):
        return self._vec(text)

    def warmup(self):
        return 0.0


@pytest.fixture
def index() -> VectorIndex:
    chunks = [
        make_chunk(0, "The Ganges river flows through India in 1947"),
        make_chunk(1, "A corporation is a legal entity"),
        make_chunk(2, "Cricket is played with a bat"),
        make_chunk(3, "The Ganges is sacred"),
    ]
    embedder = StubEmbedder()
    idx = VectorIndex(dim=3)
    idx.add(chunks, embedder.embed_texts([c.text for c in chunks]))
    return idx


class TestVectorIndex:
    def test_ranks_by_cosine_similarity(self, index):
        hits = index.search(StubEmbedder().embed_query("ganges"), k=4)
        assert {index.chunks[h.idx].chunk_id for h in hits[:2]} == {"c0", "c3"}
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    def test_scores_are_cosine_because_vectors_are_normalised(self, index):
        for hit in index.search(StubEmbedder().embed_query("cricket"), k=4):
            assert -1.0001 <= hit.score <= 1.0001

    def test_k_larger_than_corpus_is_clamped(self, index):
        assert len(index.search(StubEmbedder().embed_query("ganges"), k=99)) == 4

    def test_empty_index_returns_no_hits(self):
        assert VectorIndex(dim=3).search(unit([1, 0, 0]), k=5) == []

    def test_mismatched_chunk_and_vector_counts_are_rejected(self):
        idx = VectorIndex(dim=3)
        with pytest.raises(ValueError, match="mismatch"):
            idx.add([make_chunk(0, "a")], np.zeros((2, 3), dtype="float32"))

    def test_wrong_dimensionality_is_rejected(self):
        idx = VectorIndex(dim=3)
        with pytest.raises(ValueError, match="dim"):
            idx.add([make_chunk(0, "a")], np.zeros((1, 5), dtype="float32"))

    def test_centroid_is_normalised(self, index):
        assert float(np.linalg.norm(index.centroid())) == pytest.approx(1.0, abs=1e-5)

    def test_empty_index_centroid_does_not_divide_by_zero(self):
        assert not np.isnan(VectorIndex(dim=3).centroid()).any()

    def test_save_and_load_round_trip(self, index, tmp_path):
        index.save(tmp_path)
        loaded = VectorIndex.load(tmp_path)
        assert loaded.size == index.size
        assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in index.chunks]
        q = StubEmbedder().embed_query("corporation")
        assert loaded.search(q, 1)[0].idx == index.search(q, 1)[0].idx


class TestNumpyFallbackParity:
    """The fallback must not quietly change retrieval quality."""

    def test_numpy_and_faiss_agree_on_ranking(self):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(200, 8)).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        chunks = [make_chunk(i, f"chunk {i}") for i in range(200)]

        np_index = VectorIndex(dim=8, backend="numpy")
        np_index.add(chunks, vectors)

        query = vectors[7:8]
        np_hits = np_index.search(query[0], k=10)

        from retrieval.vector_index import FAISS_AVAILABLE

        if not FAISS_AVAILABLE:
            pytest.skip("faiss not installed in this environment")

        fa_index = VectorIndex(dim=8, backend="faiss")
        fa_index.add(chunks, vectors)
        fa_hits = fa_index.search(query[0], k=10)

        assert [h.idx for h in np_hits] == [h.idx for h in fa_hits]
        for a, b in zip(np_hits, fa_hits):
            assert a.score == pytest.approx(b.score, abs=1e-5)

    def test_numpy_topk_ordering_is_descending(self):
        idx = _NumpyFlatIP(3)
        idx.add(np.asarray([[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0]], dtype="float32"))
        scores, _ = idx.search(np.asarray([[1, 0, 0]], dtype="float32"), k=3)
        assert list(scores[0]) == sorted(scores[0], reverse=True)


class TestLexical:
    def test_devanagari_is_tokenised(self):
        """ASCII-only tokenisers silently produce an empty Hindi index."""
        assert tokenize("गंगा नदी भारत") == ["गंगा", "नदी", "भारत"]

    def test_stopwords_are_dropped(self):
        assert "है" not in tokenize("गंगा नदी क्या है")

    def test_all_stopword_query_falls_back_to_raw_tokens(self):
        """Otherwise such a query scores nothing at all."""
        assert tokenize("यह क्या है") != []

    def test_exact_rare_term_outranks_topical_match(self):
        """BM25's reason for existing in this pipeline."""
        chunks = [make_chunk(i, f"The Ganges river flows through India section {i}")
                  for i in range(30)]
        chunks.append(make_chunk(99, "The Ganges river flooded in 1947 severely"))
        hits = LexicalIndex(chunks).search("1947", k=5)
        assert hits and hits[0].idx == 30

    def test_idf_stays_positive_on_a_degenerate_corpus(self):
        """The `log(1 + ...)` IDF form, pinned deliberately.

        The textbook Okapi IDF, `log((N - df + 0.5) / (df + 0.5))`, is *zero* for a term
        in one of two documents and goes negative for terms in more than half the
        corpus — a common word would then actively push documents down the ranking.
        The +1 inside the log keeps every IDF positive, so a rare term is discriminative
        even on a corpus far smaller than the real index.
        """
        chunks = [
            make_chunk(0, "The Ganges river flows through India"),
            make_chunk(1, "The Ganges river flooded in 1947 severely"),
        ]
        hits = LexicalIndex(chunks).search("1947", k=2)
        assert hits and hits[0].idx == 1 and hits[0].score > 0.0

    def test_common_term_never_scores_negative(self):
        chunks = [make_chunk(i, "ganges river") for i in range(10)]
        assert all(h.score > 0.0 for h in LexicalIndex(chunks).search("ganges", k=10))

    def test_term_frequency_saturates(self):
        """K1 saturation: 20 repeats must not score 20x a single occurrence."""
        chunks = [
            make_chunk(0, "ganges " * 20 + "padding text here"),
            make_chunk(1, "ganges padding text here"),
        ] + [make_chunk(i + 2, f"unrelated document {i}") for i in range(20)]
        hits = {h.idx: h.score for h in LexicalIndex(chunks).search("ganges", k=5)}
        assert hits[0] > hits[1]
        assert hits[0] < hits[1] * 5, "term frequency is not saturating"

    def test_longer_documents_are_length_normalised(self):
        """Same single occurrence; the longer document must score lower."""
        chunks = [
            make_chunk(0, "ganges"),
            make_chunk(1, "ganges " + "filler word here plenty more " * 20),
        ] + [make_chunk(i + 2, f"unrelated document {i}") for i in range(20)]
        hits = {h.idx: h.score for h in LexicalIndex(chunks).search("ganges", k=5)}
        assert hits[0] > hits[1]

    def test_zero_score_documents_are_excluded(self):
        chunks = [make_chunk(0, "cricket bat"), make_chunk(1, "corporation law")]
        assert all(h.score > 0 for h in LexicalIndex(chunks).search("cricket", k=2))

    def test_empty_index_returns_nothing(self):
        assert LexicalIndex().search("anything", k=5) == []

    def test_stopword_list_covers_both_scripts(self):
        assert "है" in STOPWORDS and "the" in STOPWORDS


class TestHybridFusion:
    def test_rrf_rewards_agreement_between_retrievers(self):
        fused = dict(Retriever._rrf(dense=[5, 1, 2], lexical=[1, 9, 5], k=60))
        # 1 and 5 appear in both lists; 2 and 9 appear in only one.
        assert fused[1] > fused[2]
        assert fused[5] > fused[9]

    def test_rrf_score_is_rank_based_not_magnitude_based(self):
        a = dict(Retriever._rrf(dense=[7], lexical=[], k=60))
        b = dict(Retriever._rrf(dense=[3], lexical=[], k=60))
        assert a[7] == b[3]

    def test_hybrid_surfaces_a_bm25_only_hit(self, index):
        retriever = Retriever(index, LexicalIndex(index.chunks), StubEmbedder(),
                              RetrievalConfig(top_k=4, candidate_k=2))
        result = retriever.retrieve("1947")
        assert result.strategy == "hybrid_rrf"
        assert any(s.chunk.chunk_id == "c0" for s in result.chunks)

    def test_bm25_only_hits_still_get_a_dense_score(self, index):
        """The confidence gate reads dense_score; a None would exclude the chunk."""
        retriever = Retriever(index, LexicalIndex(index.chunks), StubEmbedder(),
                              RetrievalConfig(top_k=4, candidate_k=1))
        for scored in retriever.retrieve("1947").chunks:
            assert scored.dense_score is not None
            assert -1.0001 <= scored.dense_score <= 1.0001

    def test_dense_only_mode_reports_its_strategy(self, index):
        retriever = Retriever(index, None, StubEmbedder(), RetrievalConfig(use_hybrid=False))
        assert retriever.retrieve("ganges").strategy == "dense"

    def test_top_score_is_cosine_not_fused(self, index):
        """Fused RRF scores are ~0.016; a cosine top score must be near 1.0 here."""
        retriever = Retriever(index, LexicalIndex(index.chunks), StubEmbedder(),
                              RetrievalConfig(top_k=3))
        assert retriever.retrieve("ganges").top_score == pytest.approx(1.0, abs=1e-4)

    def test_supplied_query_vector_avoids_a_second_embedding(self, index):
        calls = {"n": 0}

        class Counting(StubEmbedder):
            def embed_query(self, text):
                calls["n"] += 1
                return super().embed_query(text)

        retriever = Retriever(index, None, Counting(), RetrievalConfig())
        vector = Counting().embed_query("ganges")
        calls["n"] = 0
        retriever.retrieve("ganges", query_vector=vector)
        assert calls["n"] == 0, "retrieve() re-embedded a query it was handed"

    def test_top_k_is_respected(self, index):
        retriever = Retriever(index, LexicalIndex(index.chunks), StubEmbedder(),
                              RetrievalConfig(top_k=2))
        assert len(retriever.retrieve("ganges").chunks) == 2

    def test_ranks_are_sequential_from_one(self, index):
        retriever = Retriever(index, LexicalIndex(index.chunks), StubEmbedder(),
                              RetrievalConfig(top_k=3))
        assert [s.rank for s in retriever.retrieve("ganges").chunks] == [1, 2, 3]

    def test_stage_timings_are_recorded(self, index):
        retriever = Retriever(index, LexicalIndex(index.chunks), StubEmbedder())
        retriever.retrieve("ganges")
        assert {"dense_ms", "lexical_ms", "fuse_ms"} <= set(retriever.last_timings)

    def test_passage_dedupe_collapses_same_passage_chunks(self):
        chunks = [
            make_chunk(0, "Ganges part one", query_id="q1", passage_idx=0),
            make_chunk(1, "Ganges part two", query_id="q1", passage_idx=0),
            make_chunk(2, "Ganges elsewhere", query_id="q2", passage_idx=0),
        ]
        idx = VectorIndex(dim=3)
        idx.add(chunks, StubEmbedder().embed_texts([c.text for c in chunks]))
        retriever = Retriever(idx, None, StubEmbedder(),
                              RetrievalConfig(top_k=5, dedupe_by_passage=True))
        results = retriever.retrieve("ganges")
        keys = [(s.chunk.query_id, s.chunk.passage_idx) for s in results.chunks]
        assert len(keys) == len(set(keys))
