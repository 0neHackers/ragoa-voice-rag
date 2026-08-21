"""Guardrail 4 — groundedness check, post-generation.

The last line of defence: the model had good context and drifted anyway. This is the
only check positioned to catch that, because it is the only one that sees the generated
text.

**The default is a lexical-overlap check, and that's a choice.** The textbook answer is an
NLI/entailment pass or a second "is this answer supported? yes/no" LLM call. That is more
accurate and it is implemented here (`use_llm=True`), but it doubles the number of
provider round-trips on a latency-graded pipeline, so it is opt-in rather than default.

What the cheap check measures: the fraction of the answer's *content* tokens that appear
in the retrieved context. An extractive or faithfully-grounded answer reuses the
context's vocabulary heavily — proper nouns, numbers, domain terms. A hallucinated answer
introduces entities that are simply not in the context. Stopwords are stripped in both
Hindi and English, because otherwise a long answer's function words alone push overlap
past any threshold.

**Two limits, stated plainly.** (1) A fluent paraphrase that is perfectly faithful can
score low on lexical overlap, so the threshold is set permissively — this check is tuned
to catch answers that invent *entities*, not answers that reword. (2) It cannot detect a
confident wrong *inference* drawn entirely from words present in the context. Neither
limit is fixable at this cost tier; the LLM path exists for teams willing to pay for it.

Numbers get extra weight. A fabricated statistic or date is the highest-damage and
most-common hallucination in retrieval QA, and it is exactly the case lexical overlap
detects most reliably.
"""

from __future__ import annotations

import os
import re
import unicodedata

from harness.types import Answer, GuardrailVerdict, RetrievalResult
from guardrails.base import Guardrail

DEFAULT_MIN_OVERLAP = 0.45

#: A number in the answer that is absent from the context is treated as fabricated
#: regardless of how well the surrounding prose overlaps.
PENALISE_UNSUPPORTED_NUMBERS = True

_TOKEN_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
_NUMBER_RE = re.compile(r"\d[\d,.०-९]*")

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
        use_llm: bool = False,
        llm_client: object | None = None,
    ) -> None:
        self.min_overlap = (
            float(os.getenv("GUARDRAIL_MIN_OVERLAP", DEFAULT_MIN_OVERLAP))
            if min_overlap is None else float(min_overlap)
        )
        self.use_llm = use_llm
        self.llm_client = llm_client

    def _check(  # type: ignore[override]
        self, query: str, answer: Answer, retrieval: RetrievalResult
    ) -> GuardrailVerdict:
        context = " ".join(sc.chunk.text for sc in retrieval.chunks)

        if not answer.text.strip():
            return self._block(reason="Generator returned an empty answer.", score=0.0,
                               threshold=self.min_overlap)

        # An extractive answer is a verbatim span of the context. Scoring it for lexical
        # overlap is measuring whether a copy resembles its original.
        if answer.mode == "extractive":
            return self._pass(score=1.0, threshold=self.min_overlap,
                              reason="Answer is a verbatim span of the retrieved context.")

        context_tokens = _content_tokens(context)
        answer_tokens = _content_tokens(answer.text)

        if not answer_tokens:
            return self._pass(score=1.0, threshold=self.min_overlap,
                              reason="Answer carries no content tokens to verify.")

        supported = sum(1 for tok in answer_tokens if tok in context_tokens)
        overlap = supported / len(answer_tokens)

        if PENALISE_UNSUPPORTED_NUMBERS:
            context_numbers = set(_NUMBER_RE.findall(_fold(context)))
            answer_numbers = set(_NUMBER_RE.findall(_fold(answer.text)))
            invented = answer_numbers - context_numbers
            if invented:
                return self._block(
                    reason=(
                        f"Answer contains figures absent from the retrieved passages "
                        f"({', '.join(sorted(invented)[:3])}) — treating as fabricated."
                    ),
                    score=overlap, threshold=self.min_overlap,
                )

        if overlap < self.min_overlap:
            return self._block(
                reason=(
                    f"Only {overlap:.0%} of the answer's content words appear in the "
                    f"retrieved passages (threshold {self.min_overlap:.0%}) — the answer is "
                    "not sufficiently grounded in the context."
                ),
                score=overlap, threshold=self.min_overlap,
            )

        if self.use_llm and self.llm_client is not None:
            return self._llm_entailment(answer, context, overlap)

        return self._pass(score=overlap, threshold=self.min_overlap,
                          reason=f"{overlap:.0%} of answer content words are supported by the context.")

    def _llm_entailment(self, answer: Answer, context: str, overlap: float) -> GuardrailVerdict:
        """Opt-in second pass. Costs a full round-trip, so it runs only after the cheap
        check has already passed — there is no point paying for it to confirm a rejection.
        """
        verdict = self.llm_client.entailment_check(answer.text, context)  # type: ignore[attr-defined]
        if verdict is False:
            return self._block(
                reason="Entailment check: the answer does not follow from the retrieved passages.",
                score=overlap, threshold=self.min_overlap,
            )
        return self._pass(score=overlap, threshold=self.min_overlap,
                          reason="Lexical overlap and entailment check both passed.")


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(_fold(text)) if t not in STOPWORDS and len(t) > 1}


__all__ = ["GroundednessGuardrail", "DEFAULT_MIN_OVERLAP", "STOPWORDS"]
