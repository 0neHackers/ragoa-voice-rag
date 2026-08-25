"""Strategy 2 — semantic chunking on embedding-similarity breakpoints.

Sentences are embedded, adjacent pairs are compared, and a chunk boundary is placed
wherever consecutive sentences are *unusually dissimilar* — i.e. where the topic turns.

The threshold is **percentile-based, not absolute**. A fixed cosine cutoff like 0.5 is
the standard mistake here: cosine distances between adjacent sentences of a single
coherent passage sit in a narrow, corpus-dependent band, so a constant either splits at
every sentence or never splits at all. Taking the Nth percentile of *this passage's own*
distance distribution makes the strategy self-calibrating across languages and topics.

This is the most expensive strategy — it costs one embedding pass over every sentence in
the corpus at index build time. That is a build-time cost only, paid once, and never on
the query path.
"""

from __future__ import annotations

import numpy as np

from chunking.base import ChunkingStrategy, approx_tokens, split_sentences
from data.loader import Example, Passage
from harness.types import Chunk


class SemanticChunker(ChunkingStrategy):
    def __init__(
        self,
        embedder=None,
        breakpoint_percentile: float = 80.0,
        min_sentences: int = 2,
        max_tokens: int = 320,
        min_chunk_chars: int = 80,
        name: str = "semantic",
    ) -> None:
        if not 50.0 <= breakpoint_percentile <= 99.0:
            raise ValueError("breakpoint_percentile should be between 50 and 99")
        self.name = name
        self.breakpoint_percentile = breakpoint_percentile
        self.min_sentences = min_sentences
        self.max_tokens = max_tokens
        self.min_chunk_chars = min_chunk_chars
        self._embedder = embedder

    @property
    def embedder(self):
        # Imported lazily so tests of the pure splitting logic don't pay a model load.
        if self._embedder is None:
            from retrieval.embedder import Embedder

            self._embedder = Embedder()
        return self._embedder

    # -- core ---------------------------------------------------------------- #

    def _boundaries(self, sentences: list[str]) -> list[int]:
        """Indices *after which* a chunk boundary should be placed."""
        vectors = self.embedder.embed_texts(sentences)
        # Vectors are L2-normalised, so a dot product is cosine similarity.
        sims = np.sum(vectors[:-1] * vectors[1:], axis=1)
        distances = 1.0 - sims

        if distances.size == 0:
            return []

        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        breaks: list[int] = []
        run = 0
        for i, dist in enumerate(distances):
            run += 1
            # `min_sentences` stops a passage of uniformly high distances (a list of
            # unrelated facts) from degenerating into one-sentence chunks.
            if dist >= threshold and run >= self.min_sentences:
                breaks.append(i)
                run = 0
        return breaks

    def _assemble(self, sentences: list[str], breaks: list[int]) -> list[str]:
        chunks: list[str] = []
        start = 0
        for b in breaks + [len(sentences) - 1]:
            group = sentences[start : b + 1]
            if group:
                chunks.append(" ".join(group))
            start = b + 1

        # Enforce the hard ceiling: a semantically coherent run can still be too long
        # to embed usefully, so oversized chunks are halved by sentence count.
        bounded: list[str] = []
        for chunk in chunks:
            if approx_tokens(chunk) <= self.max_tokens:
                bounded.append(chunk)
                continue
            sents = split_sentences(chunk)
            mid = max(1, len(sents) // 2)
            bounded.extend([" ".join(sents[:mid]), " ".join(sents[mid:])])

        return [c.strip() for c in bounded if c.strip()]

    def chunk_passage(self, passage: Passage) -> list[str]:
        sentences = split_sentences(passage.text)
        if len(sentences) <= self.min_sentences:
            return [passage.text.strip()] if passage.text.strip() else []
        return self._assemble(sentences, self._boundaries(sentences))

    # -- batched corpus pass -------------------------------------------------- #

    #: Sentences embedded per call. Big enough that the batch dimension is well used,
    #: small enough that the intermediate activations stay bounded. See `_embed_windowed`.
    EMBED_WINDOW = 4096

    def _embed_windowed(self, sentences: list[str]) -> np.ndarray:
        """Embed sentences in fixed windows rather than one enormous call.

        The original version passed every sentence in the corpus to `embed_texts` at
        once. That is fine at benchmark scale and dies at production scale: at 1,500
        examples it produced roughly 150k sentences, and the ONNX session raised
        `bad allocation` partway through — which is why the shipped index shipped without
        this strategy even though it scored best on recall.

        Windowing costs nothing measurable. The batch dimension is already saturated well
        below 4096, so throughput is unchanged; the only thing that changes is that peak
        memory stops scaling with corpus size.
        """
        if not sentences:
            return np.zeros((0, 1), dtype="float32")

        if len(sentences) <= self.EMBED_WINDOW:
            return self.embedder.embed_texts(sentences)

        parts = [
            self.embedder.embed_texts(sentences[i : i + self.EMBED_WINDOW])
            for i in range(0, len(sentences), self.EMBED_WINDOW)
        ]
        return np.vstack(parts)

    def chunk_examples(self, examples) -> list[Chunk]:
        """Override to embed sentences in large batches rather than one call per passage.

        Calling `chunk_passage` per passage would issue one tiny embedding call per
        passage — thousands of round trips through the ONNX session, each too small to
        use the batch dimension. Collecting first turns index build time from minutes
        into seconds; `_embed_windowed` then keeps peak memory flat as the corpus grows.
        """
        examples = list(examples)
        jobs: list[tuple[int, Passage, list[str]]] = []
        all_sentences: list[str] = []

        for ex_i, ex in enumerate(examples):
            for passage in ex.passages:
                sents = split_sentences(passage.text)
                if len(sents) <= self.min_sentences:
                    jobs.append((ex_i, passage, []))  # short: emit whole, no embedding
                else:
                    jobs.append((ex_i, passage, sents))
                    all_sentences.extend(sents)

        vectors = self._embed_windowed(all_sentences)

        chunks: list[Chunk] = []
        cursor = 0
        for ex_i, passage, sents in jobs:
            ex = examples[ex_i]
            if not sents:
                texts = [passage.text.strip()] if passage.text.strip() else []
            else:
                vecs = vectors[cursor : cursor + len(sents)]
                cursor += len(sents)
                texts = self._assemble(sents, self._boundaries_from_vectors(vecs))

            for i, text in enumerate(texts):
                if not text.strip():
                    continue
                chunks.append(Chunk(
                    chunk_id=f"{self.name}:{ex.query_id}:{passage.passage_idx}:{i}",
                    text=text.strip(),
                    strategy=self.name,
                    query_id=ex.query_id,
                    passage_idx=passage.passage_idx,
                    source_lang=passage.source_lang,
                    char_len=len(text.strip()),
                    is_selected_passage=passage.is_selected,
                ))
        return chunks

    def _boundaries_from_vectors(self, vectors: np.ndarray) -> list[int]:
        if len(vectors) < 2:
            return []
        distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        breaks: list[int] = []
        run = 0
        for i, dist in enumerate(distances):
            run += 1
            if dist >= threshold and run >= self.min_sentences:
                breaks.append(i)
                run = 0
        return breaks

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "breakpoint_percentile": self.breakpoint_percentile,
            "max_tokens": self.max_tokens,
            "rationale": "splits where the topic turns; percentile threshold self-calibrates per passage",
        }


__all__ = ["SemanticChunker"]
