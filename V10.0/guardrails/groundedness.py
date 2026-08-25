"""Guardrail 4 — groundedness check, post-generation.

The last line of defence: the model had good context and drifted anyway. It's the only
check positioned to catch that, because it's the only one that sees the generated text.

## Two signals, combined with OR, because they fail on different answers

**Semantic** — cosine between the answer's embedding and each retrieved chunk's, taking
the best match. Costs one extra embedding call (~95ms), which sits in the generation half
of the pipeline where a ~2.7s LLM call already dominates, not in the 200ms retrieval
budget.

**Lexical** — the fraction of the answer's content tokens that appear in the context, with
Hindi and English stopwords stripped.

An answer passes if **either** clears its bar. That's not hedging, it's the shape of the
data: a heavy but faithful paraphrase scores low lexically and high semantically, while a
copied-but-irrelevant span does the reverse. Their failures are close to uncorrelated, so
requiring both would compound two false-refusal rates, and requiring either compounds the
two catch rates instead. A genuine hallucination fails both, which is exactly what makes
the OR safe.

## The numbers, and why this replaced a lexical-only check

Measured over 30 real generated answers. "Hallucinated" pairs each answer with a *different*
question's retrieved context — same model, same language, same length, the only difference
being whether the answer is about the passages. That is precisely the judgement this
guardrail has to make.

    semantic (best-matching chunk)
      faithful      min 0.076 · p05 0.170 · p10 0.302 · median 0.708
      hallucinated  median 0.122 · p90 0.256 · max 0.413

    combination                        faithful refused   hallucinations caught
    semantic>=0.25 OR lexical>=0.20            3.3%              80.0%
    semantic>=0.30 OR lexical>=0.25            3.3%              86.7%   <- shipped
    semantic>=0.40 OR lexical>=0.25            6.7%              86.7%

This guardrail was previously lexical-only, and it had to be retuned twice — 0.45 refused
37.5% of faithful answers once real paraphrase replaced extractive copies, and even 0.20 sat
at 7.5%. Worse, a stale process running the intermediate 0.30 refused essentially everything
a user typed. A metric needing that much tuning to stay usable was the wrong metric; adding
the semantic signal cut false refusals to 3.3% *and* roughly doubled what gets caught.

Comparing against the best single chunk rather than the concatenated context matters too:
whole-context embedding dilutes an answer synthesised mostly from one passage, and switching
lifted the faithful floor from 0.005 to 0.076.

## What it still can't do

It cannot detect a confident wrong *inference* drawn entirely from words and ideas present
in the context — that answer is, by both signals, grounded. The residual 3.3% false-refusal
rate is real too. So this stays one layer of three: the unsupported-number check below is
the sharp instrument, and the retrieval-confidence gate upstream stops most ungrounded
answers from ever being generated.

Numbers get extra weight. A fabricated statistic or date is the highest-damage and most
common hallucination in retrieval QA, and it's the case a bounded check detects reliably —
so an unsupported figure is a hard block regardless of how either similarity score lands.
"""

from __future__ import annotations

import os
import re
import unicodedata

import numpy as np

from harness.types import Answer, GuardrailVerdict, RetrievalResult
from guardrails.base import Guardrail

#: Cosine against the best-matching retrieved chunk.
DEFAULT_MIN_SEMANTIC = 0.30

#: Fraction of the answer's content tokens present in the context.
DEFAULT_MIN_OVERLAP = 0.25

#: A figure in the answer that is absent from the context is treated as fabricated
#: regardless of how well the answer scores on either similarity signal.
PENALISE_UNSUPPORTED_NUMBERS = True

_TOKEN_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
_NUMBER_RE = re.compile(r"\d[\d,.०-९]*")
#: Bracketed passage citations, e.g. `[1]` or `[12]`.
_CITATION_RE = re.compile(r"\[\d{1,2}\]")

#: Hindi + English function words. Present in every sentence, so they inflate overlap
#: without evidencing anything.
STOPWORDS: frozenset[str] = frozenset("""
a an the is are was were be been being of in on at to for from with by as and or but if
then than that this these those it its i you he she they we what which who whom how why
when where do does did not no yes can could should would will shall may might must have
has had there their them his her our your my me us
का की के को है हैं था थे थी में से पर और या एक यह वह ये वे कि जो तो ही भी नहीं हो होता होती होते
कर करने किया गया गई गए लिए साथ बाद पहले अपने अपनी इस उस जब तब कोई कुछ सब हम आप वो मैं मेरा
""".split())


class GroundednessGuardrail(Guardrail):
    name = "groundedness"
    fail_open = True

    def __init__(
        self,
        min_overlap: float | None = None,
        *,
        min_semantic: float | None = None,
        embedder: object | None = None,
        use_llm: bool = False,
        llm_client: object | None = None,
    ) -> None:
        self.min_overlap = (
            float(os.getenv("GUARDRAIL_MIN_OVERLAP", DEFAULT_MIN_OVERLAP))
            if min_overlap is None else float(min_overlap)
        )
        self.min_semantic = (
            float(os.getenv("GUARDRAIL_MIN_SEMANTIC", DEFAULT_MIN_SEMANTIC))
            if min_semantic is None else float(min_semantic)
        )
        self.embedder = embedder
        self.use_llm = use_llm
        self.llm_client = llm_client

    def _check(  # type: ignore[override]
        self, query: str, answer: Answer, retrieval: RetrievalResult
    ) -> GuardrailVerdict:
        chunks = [sc.chunk.text for sc in retrieval.chunks]
        context = " ".join(chunks)

        if not answer.text.strip():
            return self._block(reason="Generator returned an empty answer.", score=0.0,
                               threshold=self.min_semantic)

        # An extractive answer is a verbatim span of the context. Scoring it for similarity
        # is measuring whether a copy resembles its original.
        if answer.mode == "extractive":
            return self._pass(score=1.0, threshold=self.min_semantic,
                              reason="Answer is a verbatim span of the retrieved context.")

        # -- hard block: figures the context never mentions ---------------------
        if PENALISE_UNSUPPORTED_NUMBERS:
            context_numbers = set(_NUMBER_RE.findall(_fold(context)))
            # Strip `[1][2]` citation markers first. They're bracket indices into the
            # passage list, not claims about the world, and counting them as fabricated
            # statistics rejected every correctly-cited answer the model produced — the
            # exact behaviour this system asks for.
            answer_numbers = set(_NUMBER_RE.findall(_fold(_CITATION_RE.sub(" ", answer.text))))
            invented = answer_numbers - context_numbers
            if invented:
                return self._block(
                    reason=(
                        "Answer contains figures absent from the retrieved passages "
                        f"({', '.join(sorted(invented)[:3])}) — treating as fabricated."
                    ),
                    score=0.0, threshold=self.min_semantic,
                )

        # -- similarity: either signal is enough --------------------------------
        lexical = _lexical_overlap(answer.text, context)
        semantic = self._semantic_similarity(answer.text, chunks)

        if semantic is None:
            # No embedder wired in — fall back to lexical alone rather than passing
            # everything, and say so, because a silently-disabled guardrail is worse
            # than a noisy one.
            if lexical >= self.min_overlap:
                return self._pass(score=lexical, threshold=self.min_overlap,
                                  reason=f"{lexical:.0%} lexical overlap (no embedder for the "
                                         "semantic check).")
            return self._block(
                reason=(
                    f"Only {lexical:.0%} of the answer's content words appear in the retrieved "
                    f"passages (threshold {self.min_overlap:.0%}), and no embedder was available "
                    "for the semantic check."
                ),
                score=lexical, threshold=self.min_overlap,
            )

        if semantic >= self.min_semantic:
            verdict = self._pass(
                score=semantic, threshold=self.min_semantic,
                reason=f"Answer is semantically grounded in the retrieved passages "
                       f"(similarity {semantic:.2f}, lexical overlap {lexical:.0%}).",
            )
            if self.use_llm and self.llm_client is not None:
                return self._llm_entailment(answer, context, semantic)
            return verdict

        if lexical >= self.min_overlap:
            return self._pass(
                score=lexical, threshold=self.min_overlap,
                reason=f"Answer reuses the passages' wording ({lexical:.0%} overlap) despite a "
                       f"low semantic score ({semantic:.2f}).",
            )

        return self._block(
            reason=(
                f"Answer is not grounded in the retrieved passages — semantic similarity "
                f"{semantic:.2f} (threshold {self.min_semantic:.2f}) and only {lexical:.0%} of "
                f"its content words appear in them (threshold {self.min_overlap:.0%})."
            ),
            score=semantic, threshold=self.min_semantic,
        )

    # ------------------------------------------------------------------ #

    def _semantic_similarity(self, answer: str, chunks: list[str]) -> float | None:
        """Best cosine between the answer and any single retrieved chunk.

        Best-of rather than against the concatenation: an answer drawn mostly from one
        passage gets diluted when compared to the average of five. Measured, that change
        lifted the faithful-answer floor from 0.005 to 0.076.
        """
        if self.embedder is None or not chunks:
            return None
        try:
            vectors = self.embedder.embed_texts(  # type: ignore[attr-defined]
                [answer] + [c[:2000] for c in chunks]
            )
        except Exception:  # noqa: BLE001 - degrade to lexical rather than fail the request
            return None

        answer_vec = np.asarray(vectors[0])
        return max(float(answer_vec @ np.asarray(v)) for v in vectors[1:])

    def _llm_entailment(self, answer: Answer, context: str, score: float) -> GuardrailVerdict:
        """Opt-in second pass. Costs a full round-trip, so it runs only after the cheap
        checks have already passed — no point paying for it to confirm a rejection.
        """
        verdict = self.llm_client.entailment_check(answer.text, context)  # type: ignore[attr-defined]
        if verdict is False:
            return self._block(
                reason="Entailment check: the answer does not follow from the retrieved passages.",
                score=score, threshold=self.min_semantic,
            )
        return self._pass(score=score, threshold=self.min_semantic,
                          reason="Similarity and entailment checks both passed.")


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(_fold(text)) if t not in STOPWORDS and len(t) > 1}


def _lexical_overlap(answer: str, context: str) -> float:
    """Fraction of the answer's content tokens that appear in the context."""
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 1.0
    context_tokens = _content_tokens(context)
    return sum(1 for t in answer_tokens if t in context_tokens) / len(answer_tokens)


__all__ = [
    "GroundednessGuardrail", "DEFAULT_MIN_OVERLAP", "DEFAULT_MIN_SEMANTIC", "STOPWORDS",
]
