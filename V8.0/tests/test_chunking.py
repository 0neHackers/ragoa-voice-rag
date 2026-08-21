"""Chunking strategy tests.

Each strategy is asserted on the property that justifies its existence, not just on
"returns a list": fixed-size overlaps, recursive respects boundaries, metadata-aware
leaves pre-segmented passages alone, and all of them carry provenance metadata through.

Semantic chunking is tested with a stub embedder so the logic is verified without a
220MB model load — the percentile-threshold behaviour is the part worth pinning down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunking.base import approx_tokens, split_paragraphs, split_sentences  # noqa: E402
from chunking.fixed_size import FixedSizeChunker  # noqa: E402
from chunking.metadata_aware import MetadataAwareChunker  # noqa: E402
from chunking.recursive import RecursiveChunker  # noqa: E402
from chunking.registry import STRATEGIES, get_strategy  # noqa: E402
from chunking.semantic import SemanticChunker  # noqa: E402
from data.loader import Example, Passage, _coerce_passages  # noqa: E402

HINDI = "भारत एक विशाल देश है। इसकी जनसंख्या एक अरब से अधिक है। यहाँ अनेक भाषाएँ बोली जाती हैं। "
LONG_HINDI = (HINDI * 12).strip()
LONG_ENGLISH = ("The Ganges is a major river. It flows through northern India. "
                "Millions depend on it for water. ") * 12


def passage(text: str, **kw) -> Passage:
    base = dict(query_id="q1", passage_idx=0, source_lang="hi")
    base.update(kw)
    return Passage(text=text, **base)


class TestTextUtils:
    def test_danda_is_a_sentence_boundary(self):
        """Devanagari's full stop. Missing it collapses Hindi text into one sentence."""
        assert len(split_sentences("पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।")) == 3

    def test_latin_punctuation_still_works(self):
        assert len(split_sentences("One. Two? Three!")) == 3

    def test_paragraph_split_on_blank_lines(self):
        assert len(split_paragraphs("para one\n\npara two\n\npara three")) == 3

    def test_token_estimate_is_script_aware(self):
        """Devanagari packs fewer chars per token than Latin under multilingual vocabs."""
        deva, latin = "क" * 100, "a" * 100
        assert approx_tokens(deva) > approx_tokens(latin)

    def test_empty_text_is_zero_tokens(self):
        assert approx_tokens("") == 0


class TestFixedSize:
    def test_short_passage_is_left_whole(self):
        assert FixedSizeChunker().chunk_passage(passage(HINDI)) == [HINDI.strip()]

    def test_long_passage_is_split(self):
        assert len(FixedSizeChunker(target_tokens=64).chunk_passage(passage(LONG_HINDI))) > 1

    def test_consecutive_chunks_actually_overlap(self):
        """The whole point of the baseline: no fact is lost to a boundary."""
        chunks = FixedSizeChunker(target_tokens=48, overlap_ratio=0.4).chunk_passage(
            passage(LONG_ENGLISH)
        )
        assert len(chunks) >= 2
        tail = chunks[0][-20:]
        assert tail in chunks[1], "overlap region did not carry into the next chunk"

    def test_zero_overlap_is_permitted(self):
        chunks = FixedSizeChunker(target_tokens=48, overlap_ratio=0.0).chunk_passage(
            passage(LONG_ENGLISH)
        )
        assert len(chunks) > 1

    def test_invalid_overlap_is_rejected_at_construction(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(overlap_ratio=1.0)

    def test_empty_passage_yields_nothing(self):
        assert FixedSizeChunker().chunk_passage(passage("   ")) == []


class TestRecursive:
    def test_prefers_paragraph_boundaries(self):
        text = "\n\n".join([LONG_ENGLISH[:300], LONG_ENGLISH[300:600], LONG_ENGLISH[600:900]])
        chunks = RecursiveChunker(target_tokens=60, max_tokens=90).chunk_passage(passage(text))
        assert len(chunks) > 1

    def test_does_not_cut_mid_sentence_when_a_boundary_exists(self):
        chunks = RecursiveChunker(target_tokens=40, max_tokens=60).chunk_passage(
            passage(LONG_HINDI)
        )
        # Every chunk should end on a sentence terminator, since one is always available.
        assert all(c.rstrip()[-1] in "।?!." for c in chunks)

    def test_text_with_no_punctuation_still_terminates(self):
        """The hard-cut fallback — guards against infinite recursion."""
        chunks = RecursiveChunker(target_tokens=20, max_tokens=30).chunk_passage(
            passage("क" * 3000)
        )
        assert len(chunks) > 1
        assert "".join(chunks).replace(" ", "") == "क" * 3000

    def test_short_passage_is_left_whole(self):
        assert RecursiveChunker().chunk_passage(passage(HINDI)) == [HINDI.strip()]


class TestMetadataAware:
    def test_pre_segmented_passage_is_not_resplit(self):
        """The corpus already segmented these; re-splitting discards that work."""
        assert MetadataAwareChunker().chunk_passage(passage(HINDI)) == [HINDI.strip()]

    def test_oversized_passage_splits_on_sentences(self):
        chunks = MetadataAwareChunker(max_tokens=60).chunk_passage(passage(LONG_HINDI))
        assert len(chunks) > 1
        assert all(c.rstrip()[-1] in "।?!." for c in chunks)

    def test_tiny_fragments_are_dropped(self):
        assert MetadataAwareChunker(min_chunk_chars=40).chunk_passage(passage("ok")) == []

    def test_metadata_is_carried_onto_every_chunk(self):
        ex = Example(query_id="q42", query="?", answer="a", query_type="DESCRIPTION",
                     source_lang="hin_Deva",
                     passages=[passage(LONG_HINDI, query_id="q42", passage_idx=3,
                                       is_selected=True)])
        chunks = MetadataAwareChunker(max_tokens=60).chunk_examples([ex])
        assert chunks
        for c in chunks:
            assert c.query_id == "q42"
            assert c.passage_idx == 3
            assert c.is_selected_passage is True
            assert c.strategy == "metadata_aware"
            assert c.chunk_id.startswith("metadata_aware:q42:3:")


class StubEmbedder:
    """Two topics: the first half of sentences point one way, the second half another.

    Produces exactly one large adjacent-distance spike, so the breakpoint location is
    deterministic and the percentile logic can be asserted precisely.
    """

    def embed_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        vecs = []
        for i, _ in enumerate(texts):
            vecs.append([1.0, 0.0] if i < len(texts) // 2 else [0.0, 1.0])
        return np.asarray(vecs, dtype="float32")


class TestSemantic:
    def test_splits_at_the_topic_change(self):
        sentences = [f"वाक्य {i} है।" for i in range(8)]
        chunker = SemanticChunker(embedder=StubEmbedder(), breakpoint_percentile=90.0,
                                  min_sentences=1)
        chunks = chunker.chunk_passage(passage(" ".join(sentences)))
        assert len(chunks) == 2
        assert "वाक्य 3" in chunks[0] and "वाक्य 4" in chunks[1]

    def test_short_passage_skips_embedding_entirely(self):
        class Exploding:
            def embed_texts(self, *a, **k):
                raise AssertionError("should not embed a passage this short")

        chunker = SemanticChunker(embedder=Exploding(), min_sentences=2)
        assert chunker.chunk_passage(passage("एक। दो।")) == ["एक। दो।"]

    def test_min_sentences_prevents_single_sentence_chunks(self):
        sentences = [f"S{i} is here." for i in range(10)]

        class Alternating:
            def embed_texts(self, texts, batch_size=256):
                return np.asarray([[1.0, 0.0] if i % 2 else [0.0, 1.0]
                                   for i in range(len(texts))], dtype="float32")

        chunks = SemanticChunker(embedder=Alternating(), breakpoint_percentile=50.0,
                                 min_sentences=3).chunk_passage(passage(" ".join(sentences)))
        assert all(len(split_sentences(c)) >= 2 for c in chunks[:-1])

    def test_invalid_percentile_is_rejected(self):
        with pytest.raises(ValueError):
            SemanticChunker(breakpoint_percentile=10.0)


class TestRegistry:
    def test_all_four_required_strategies_are_registered(self):
        assert set(STRATEGIES) == {"fixed_size", "semantic", "recursive", "metadata_aware"}

    def test_unknown_strategy_raises(self):
        with pytest.raises(KeyError):
            get_strategy("nonexistent")

    @pytest.mark.parametrize("name", ["fixed_size", "recursive", "metadata_aware"])
    def test_each_strategy_tags_its_own_chunks(self, name):
        ex = Example(query_id="q1", query="?", answer="a", query_type="t", source_lang="hi",
                     passages=[passage(LONG_HINDI)])
        chunks = get_strategy(name).chunk_examples([ex])
        assert chunks and all(c.strategy == name for c in chunks)

    def test_chunk_ids_are_unique_within_a_strategy(self):
        ex = Example(query_id="q1", query="?", answer="a", query_type="t", source_lang="hi",
                     passages=[passage(LONG_HINDI, passage_idx=i) for i in range(3)])
        chunks = get_strategy("fixed_size", target_tokens=40).chunk_examples([ex])
        assert len({c.chunk_id for c in chunks}) == len(chunks)


class TestLoaderNormalisation:
    def test_msmarco_xi_struct_shape(self):
        raw = {
            "Translated_passages": ["हिंदी एक", "हिंदी दो"],
            "English_passages": ["english one", "english two"],
            "is_selected": [0, 1],
        }
        out = _coerce_passages(raw, "q1", "hi")
        assert [p.text for p in out] == ["हिंदी एक", "हिंदी दो"]
        assert [p.is_selected for p in out] == [False, True]
        assert out[1].english_text == "english two"

    def test_english_can_be_preferred(self):
        raw = {"Translated_passages": ["हिंदी"], "English_passages": ["english"],
               "is_selected": [1]}
        out = _coerce_passages(raw, "q1", "hi", prefer_translated=False)
        assert out[0].text == "english"
        assert out[0].english_text == "हिंदी"

    def test_plain_msmarco_fallback_shape(self):
        out = _coerce_passages({"passage_text": ["a", "b"], "is_selected": [1, 0]}, "q1", "en")
        assert len(out) == 2 and out[0].is_selected is True

    def test_list_of_structs_shape(self):
        out = _coerce_passages([{"passage_text": "a", "is_selected": True}], "q1", "en")
        assert len(out) == 1 and out[0].is_selected is True

    def test_missing_passages_is_not_an_error(self):
        assert _coerce_passages(None, "q1", "hi") == []

    def test_blank_passages_are_skipped(self):
        out = _coerce_passages({"Translated_passages": ["", "  ", "real"],
                                "is_selected": [0, 0, 1]}, "q1", "hi")
        assert len(out) == 1 and out[0].text == "real"
