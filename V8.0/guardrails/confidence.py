"""Guardrail 3 — retrieval-confidence gate, post-retrieval and pre-generation.

Prevents the characteristic RAG failure: the corpus does not contain the answer,
retrieval returns its five least-bad chunks anyway (a vector search always returns
*something*), and the LLM writes a fluent, confident, invented answer from them.

**Three signals, and the mix is the result of measurement rather than design.** The gate
originally thresholded on dense cosine alone. Calibrated against the real 15,449-chunk
index, that was close to useless on its own:

    population        n   min    p05   median   p95    max
    answerable       60  0.472  0.547  0.675   0.837  0.860
    heldout          30  0.450  0.513  0.624   0.702  0.747
    out_of_domain     6  0.576  0.583  0.611   0.652  0.662
    gibberish         6  0.655  0.672  0.774   0.831  0.845

Gibberish — random Devanagari syllables — scored a **higher** median cosine (0.774) than
real answerable questions (0.675). Across all 42 unanswerable queries the best achievable
dense-only threshold was 0.67, which blocks 78.6% of them by also refusing **47% of real
questions** — an unusable trade for a demo. The reason is a property of the embedding model,
not a tuning error: a multilingual sentence encoder maps arbitrary Devanagari into the
same neighbourhood as Hindi prose, because that is what it was trained to do. Cosine
measures "is this the same kind of text", and nonsense Hindi is, in that sense, Hindi.

The signal that *does* separate them is lexical:

    population        n   top BM25: min   median    max   zero-scoring
    answerable       60        8.54        20.28    43.33      0/60
    heldout          30        0.00        13.26    23.38      1/30
    out_of_domain     6       12.76        16.47    19.38       0/6
    gibberish         6        0.00         0.00     0.00       6/6

Every gibberish query scored exactly zero, because none of its tokens exists anywhere in
the corpus, while **every one of the 60 answerable queries matched something**. Requiring
BM25 > 0 refuses 7 of 42 unanswerable queries at a cost of **zero false refusals** — a
free gain, since hybrid retrieval already computes the number.

So the gate now requires:

1. **Lexical support** — at least one query term must appear somewhere in the corpus.
   Catches out-of-vocabulary nonsense that cosine rates highly.
2. **Absolute top cosine** ≥ `min_top_score`. Catches semantically distant retrievals.
   Set at 0.45, just below the observed answerable minimum of 0.472. Deliberately
   permissive: the populations overlap so heavily (answerable p05 0.547 vs unanswerable
   p95 0.775) that every threshold high enough to block meaningfully also refuses real
   questions in bulk. This catches genuinely distant retrievals and nothing else, and
   the module is honest that this is most of what dense scoring can do here.
3. **Margin over the tail** — the top score must exceed the mean of the rest. A query the
   corpus answers produces a peaked distribution; one it cannot produces a flat one.

**What this gate honestly cannot do.** Look again at `out_of_domain`: plausible Hindi
questions the corpus has no answer to ("मेरा पासवर्ड क्या है?") score 0.576–0.662 on
cosine and 12.8–17.9 on BM25 — inside the answerable range on both. They are built from
ordinary Hindi words that genuinely do appear across a web corpus, so *no* retrieval-score
threshold separates them without refusing real questions too. Catching those is the job of
the two checks downstream: the generator's authorised `NO_ANSWER` refusal, and the
groundedness check. This gate is one layer of three, and pretending it were sufficient
would be the same mistake as trusting the centroid detector that preceded it.

Thresholds are model- and corpus-specific. `benchmarks/calibrate_thresholds.py`
regenerates them; changing the embedding model invalidates them.
"""

from __future__ import annotations

import os

from harness.types import GuardrailVerdict, RetrievalResult
from guardrails.base import Guardrail

#: Minimum cosine similarity for the single best chunk. Just below the measured
#: answerable minimum (0.472) — see the module docstring for why it is not higher.
DEFAULT_MIN_TOP_SCORE = 0.45

#: Minimum gap between the best chunk and the mean of the rest.
DEFAULT_MIN_MARGIN = 0.02

#: Minimum best-chunk BM25 score. Any value above 0 means at least one query term exists
#: in the corpus. Kept at 0 rather than tuned upward: the separation it exploits is
#: zero-versus-nonzero, and a higher bar would start rejecting short real queries whose
#: few terms are common.
DEFAULT_MIN_LEXICAL = 0.0


class ConfidenceGateGuardrail(Guardrail):
    name = "low_confidence"
    fail_open = True

    def __init__(
        self,
        min_top_score: float | None = None,
        min_margin: float | None = None,
        min_lexical: float | None = None,
    ) -> None:
        self.min_top_score = (
            float(os.getenv("GUARDRAIL_MIN_TOP_SCORE", DEFAULT_MIN_TOP_SCORE))
            if min_top_score is None else float(min_top_score)
        )
        self.min_margin = (
            float(os.getenv("GUARDRAIL_MIN_MARGIN", DEFAULT_MIN_MARGIN))
            if min_margin is None else float(min_margin)
        )
        self.min_lexical = (
            float(os.getenv("GUARDRAIL_MIN_LEXICAL", DEFAULT_MIN_LEXICAL))
            if min_lexical is None else float(min_lexical)
        )

    def _check(self, query: str, retrieval: RetrievalResult) -> GuardrailVerdict:  # type: ignore[override]
        if not retrieval.chunks:
            return self._block(
                reason="Retrieval returned no chunks at all.",
                score=0.0, threshold=self.min_top_score,
            )

        # Rank by dense cosine, not by fused rank: the fused #1 can be a BM25-only hit
        # that shares a rare token with the query while being semantically unrelated.
        dense_scores = [c.dense_score for c in retrieval.chunks if c.dense_score is not None]
        if not dense_scores:
            # `_materialise` backfills dense scores precisely so this cannot happen. If it
            # somehow does, say so rather than passing on no evidence.
            return self._block(
                reason="No dense similarity scores available to judge confidence against.",
                threshold=self.min_top_score,
            )

        top = max(dense_scores)

        # -- 1. lexical support ------------------------------------------------
        # Checked first because it is the one signal that cleanly separates nonsense,
        # and because its failure has the clearest explanation for the user.
        #
        # `lexical_score` is None on a chunk BM25 never matched, so "every score is None"
        # is the *strongest* possible absence of lexical evidence, not a missing
        # measurement. Treating None as no-data and skipping the check let every
        # gibberish query through — the exact case this signal exists to catch.
        top_lexical = max(
            (c.lexical_score for c in retrieval.chunks if c.lexical_score is not None),
            default=0.0,
        )
        if top_lexical <= self.min_lexical:
            return self._block(
                reason=(
                    "No term in this query appears anywhere in the corpus "
                    f"(best lexical score {top_lexical:.2f}). The embedding similarity of "
                    f"{top:.3f} is not evidence — a multilingual encoder rates arbitrary "
                    "text in a familiar script highly."
                ),
                score=top_lexical, threshold=self.min_lexical,
            )

        # -- 2. absolute dense score -------------------------------------------
        if top < self.min_top_score:
            return self._block(
                reason=(
                    f"Best retrieved passage scored {top:.3f}, below the "
                    f"{self.min_top_score:.2f} confidence threshold — the corpus does not "
                    "appear to contain an answer to this question."
                ),
                score=top, threshold=self.min_top_score,
            )

        # -- 3. margin over the tail -------------------------------------------
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


__all__ = [
    "ConfidenceGateGuardrail",
    "DEFAULT_MIN_TOP_SCORE", "DEFAULT_MIN_MARGIN", "DEFAULT_MIN_LEXICAL",
]
