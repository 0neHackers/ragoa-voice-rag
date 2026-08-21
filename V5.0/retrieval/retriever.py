"""Hybrid retriever — dense + BM25, fused with reciprocal rank fusion.

Why RRF rather than a weighted sum of scores: cosine similarity lives in [-1, 1] and
BM25 is unbounded and corpus-dependent, so any `alpha * dense + (1-alpha) * bm25` needs
a normalisation that has to be re-tuned per corpus and drifts as the corpus grows. RRF
throws the score magnitudes away and fuses on *rank* — `sum(1 / (k + rank))` — which
needs no tuning and no normalisation.

The cost of that is real and worth stating: RRF scores have no absolute meaning, so they
cannot be thresholded. That is exactly why `ScoredChunk` keeps `dense_score` separately —
the confidence guardrail thresholds on raw cosine, which does mean something, and never
on the fused score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from harness.types import Chunk, RetrievalResult, ScoredChunk, Timer
from retrieval.embedder import Embedder, get_embedder
from retrieval.lexical import LexicalIndex
from retrieval.vector_index import VectorIndex

#: RRF smoothing constant. 60 is the value from the original Cormack et al. paper and
#: the de-facto default; it damps the influence of the very top rank so a single
#: retriever cannot dominate the fusion on its own.
RRF_K = 60


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 30       # per-retriever depth before fusion
    use_hybrid: bool = True
    rrf_k: int = RRF_K
    dedupe_by_passage: bool = False

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        return cls(
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            candidate_k=int(os.getenv("RETRIEVAL_CANDIDATE_K", "30")),
            use_hybrid=os.getenv("RETRIEVAL_HYBRID", "1") not in ("0", "false", "False"),
        )


class Retriever:
    """Wraps the dense and lexical indexes behind one `retrieve()` call."""

    def __init__(
        self,
        vector_index: VectorIndex,
        lexical_index: LexicalIndex | None = None,
        embedder: Embedder | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.lexical_index = lexical_index
        self.embedder = embedder or get_embedder()
        self.config = config or RetrievalConfig.from_env()
        self._centroid: np.ndarray | None = None
        self.last_timings: dict[str, float] = {}

    # -- corpus-level helpers -------------------------------------------------- #

    @property
    def centroid(self) -> np.ndarray:
        """Cached corpus centroid for the off-topic guardrail."""
        if self._centroid is None:
            self._centroid = self.vector_index.centroid()
        return self._centroid

    @property
    def chunks(self) -> list[Chunk]:
        return self.vector_index.chunks

    # -- the query path -------------------------------------------------------- #

    def embed_query(self, query: str) -> np.ndarray:
        with Timer() as t:
            vector = self.embedder.embed_query(query)
        self.last_timings["embed_ms"] = t.ms
        return vector

    def retrieve(
        self,
        query: str,
        query_vector: np.ndarray | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Retrieve `top_k` chunks.

        `query_vector` is accepted so the caller can pass a vector it already computed.
        The off-topic guardrail needs the query embedded before retrieval runs, and at
        ~100ms per embedding (DECISIONS.md D8) embedding it twice would put the pipeline
        over budget on its own.
        """
        cfg = self.config
        k = top_k or cfg.top_k
        self.last_timings = {}

        if query_vector is None:
            query_vector = self.embed_query(query)

        with Timer() as t_dense:
            dense_hits = self.vector_index.search(query_vector, cfg.candidate_k)
        self.last_timings["dense_ms"] = t_dense.ms

        dense_scores = {h.idx: h.score for h in dense_hits}
        dense_ranked = [h.idx for h in dense_hits]

        lexical_ranked: list[int] = []
        lexical_scores: dict[int, float] = {}
        if cfg.use_hybrid and self.lexical_index is not None:
            with Timer() as t_lex:
                lex_hits = self.lexical_index.search(query, cfg.candidate_k)
            self.last_timings["lexical_ms"] = t_lex.ms
            lexical_ranked = [h.idx for h in lex_hits]
            lexical_scores = {h.idx: h.score for h in lex_hits}

        with Timer() as t_fuse:
            fused = (
                self._rrf(dense_ranked, lexical_ranked, cfg.rrf_k)
                if lexical_ranked
                # Dense-only: rank by cosine directly. Running RRF over a single list
                # would just re-rank it identically while discarding the scores.
                else [(idx, dense_scores[idx]) for idx in dense_ranked]
            )
            scored = self._materialise(fused, dense_scores, lexical_scores, k, query_vector)
        self.last_timings["fuse_ms"] = t_fuse.ms

        dense_only = [s.dense_score for s in scored if s.dense_score is not None]
        return RetrievalResult(
            chunks=scored,
            strategy="hybrid_rrf" if lexical_ranked else "dense",
            # Reported in cosine terms, not fused terms — this is what gets thresholded.
            top_score=max(dense_only, default=0.0),
            mean_score=float(np.mean(dense_only)) if dense_only else 0.0,
            n_candidates=len(set(dense_ranked) | set(lexical_ranked)),
        )

    # -- fusion ---------------------------------------------------------------- #

    @staticmethod
    def _rrf(dense: list[int], lexical: list[int], k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for ranked in (dense, lexical):
            for rank, idx in enumerate(ranked, start=1):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def _materialise(
        self,
        fused: list[tuple[int, float]],
        dense_scores: dict[int, float],
        lexical_scores: dict[int, float],
        k: int,
        query_vector: np.ndarray,
    ) -> list[ScoredChunk]:
        out: list[ScoredChunk] = []
        seen_passages: set[tuple[str | None, int | None]] = set()
        corpus_vectors: np.ndarray | None = None

        for idx, score in fused:
            if idx < 0 or idx >= len(self.vector_index.chunks):
                continue
            chunk = self.vector_index.chunks[idx]

            if self.config.dedupe_by_passage:
                key = (chunk.query_id, chunk.passage_idx)
                if key in seen_passages:
                    continue
                seen_passages.add(key)

            dense_score = dense_scores.get(idx)
            if dense_score is None:
                # Surfaced by BM25 but outside the dense candidate list, so the dense
                # search never scored it. Compute the cosine directly rather than
                # leaving it None: the confidence gate reads this field, and a missing
                # value would exclude the chunk from the very check that decides
                # whether we are allowed to answer. Both vectors are already
                # L2-normalised, so this is one dot product — microseconds.
                if corpus_vectors is None:
                    corpus_vectors = self.vector_index._all_vectors()  # noqa: SLF001
                dense_score = float(query_vector @ corpus_vectors[idx])

            out.append(ScoredChunk(
                chunk=chunk,
                score=float(score),
                dense_score=dense_score,
                lexical_score=lexical_scores.get(idx),
                rank=len(out) + 1,
            ))
            if len(out) >= k:
                break
        return out


__all__ = ["Retriever", "RetrievalConfig", "RRF_K"]
