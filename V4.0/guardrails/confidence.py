"""Guardrail 3 — retrieval-confidence gate, post-retrieval and pre-generation.

The most important guardrail in the system. It prevents the characteristic RAG failure:
the corpus does not contain the answer, retrieval dutifully returns its five least-bad
chunks anyway (a vector search always returns *something*), and the LLM writes a fluent,
confident, entirely invented answer from them. Nothing downstream can recover from that —
by the time generation has happened the damage is done and a groundedness check will
happily confirm that the invention is "grounded" in the irrelevant context it was
given.

Two signals, both required to pass:

1. **Absolute top score.** The best chunk's raw cosine similarity must clear
   `min_top_score`. This thresholds on cosine specifically, never on the fused RRF score —
   RRF is a rank statistic with no absolute meaning, so `rrf > x` says nothing about
   whether anything relevant was found.

2. **Margin over the tail.** The top score must exceed the mean of the remaining
   retrieved chunks by `min_margin`. A query the corpus genuinely answers produces a
   peaked score distribution — one or two strongly matching chunks standing above the
   rest. A query it does not answer produces a flat one, where everything is weakly and
   equally similar. A flat distribution at a decent absolute level is the signature of
   "this corpus is vaguely about this topic but contains no answer", which the absolute
   threshold alone will wave through.

Thresholds are model-specific and honest about it — the defaults below are for
`paraphrase-multilingual-MiniLM-L12-v2` and are calibrated by
`benchmarks/calibrate_thresholds.py` against real in-corpus and out-of-corpus queries.
Changing the embedding model invalidates them.
"""

from __future__ import annotations

import os

from harness.types import GuardrailVerdict, RetrievalResult
from guardrails.base import Guardrail

#: Minimum cosine similarity for the single best chunk.
DEFAULT_MIN_TOP_SCORE = 0.42

#: Minimum gap between the best chunk and the mean of the rest.
DEFAULT_MIN_MARGIN = 0.02


class ConfidenceGateGuardrail(Guardrail):
    name = "low_confidence"
    fail_open = True

    def __init__(
        self,
        min_top_score: float | None = None,
        min_margin: float | None = None,
    ) -> None:
        self.min_top_score = (
            float(os.getenv("GUARDRAIL_MIN_TOP_SCORE", DEFAULT_MIN_TOP_SCORE))
            if min_top_score is None else float(min_top_score)
        )
        self.min_margin = (
            float(os.getenv("GUARDRAIL_MIN_MARGIN", DEFAULT_MIN_MARGIN))
            if min_margin is None else float(min_margin)
        )

    def _check(self, query: str, retrieval: RetrievalResult) -> GuardrailVerdict:  # type: ignore[override]
        if not retrieval.chunks:
            return self._block(
                reason="Retrieval returned no chunks at all.",
                score=0.0, threshold=self.min_top_score,
            )

        # Rank by dense cosine, not by fused rank: the fused #1 can be a BM25-only hit
        # that shares a rare token with the query while being semantically unrelated.
        dense_scores = [
            c.dense_score for c in retrieval.chunks if c.dense_score is not None
        ]
        if not dense_scores:
            # `_materialise` backfills dense scores precisely so this cannot happen.
            # If it somehow does, say so rather than silently passing on no evidence.
            return self._block(
                reason="No dense similarity scores available to judge confidence against.",
                threshold=self.min_top_score,
            )

        top = max(dense_scores)

        if top < self.min_top_score:
            return self._block(
                reason=(
                    f"Best retrieved passage scored {top:.3f}, below the "
                    f"{self.min_top_score:.2f} confidence threshold — the corpus does not "
                    "appear to contain an answer to this question."
                ),
                score=top, threshold=self.min_top_score,
            )

        rest = [s for s in dense_scores if s != top] or dense_scores[1:]
        if rest:
            margin = top - (sum(rest) / len(rest))
            if margin < self.min_margin:
                return self._block(
                    reason=(
                        f"Retrieved passages are uniformly weak (top {top:.3f}, margin over "
                        f"the rest {margin:.3f} < {self.min_margin:.2f}) — no passage stands "
                        "out as actually answering the question."
                    ),
                    score=margin, threshold=self.min_margin,
                )

        return self._pass(
            score=top, threshold=self.min_top_score,
            reason=f"Top passage scored {top:.3f}, above the {self.min_top_score:.2f} threshold.",
        )


__all__ = ["ConfidenceGateGuardrail", "DEFAULT_MIN_TOP_SCORE", "DEFAULT_MIN_MARGIN"]
