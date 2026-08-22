"""Guardrail 1 — input safety, pre-retrieval.

A lightweight lexical moderation pass over the *transcript*, before it reaches retrieval
or any LLM. Scope is stated plainly: this is a first-line filter for a public voice demo,
not a moderation system. It catches overt harm-seeking phrasing and prompt-injection
attempts in Hindi and English. It will not catch obfuscated or adversarially-worded
attacks, and nothing here should be mistaken for a safety boundary.

It is worth having anyway for a reason specific to *this* pipeline: the input is a
speech transcript being fed into a retrieval corpus and then an LLM prompt. Injection
phrasing arriving by voice ("ignore your instructions and ...") is the realistic attack
on a voice RAG demo, and catching it costs ~0.05ms of a 200ms budget.

Unlike every other guardrail here, this one **fails closed** — if the check itself
errors, the query is refused rather than passed through.
"""

from __future__ import annotations

import re
import unicodedata

from harness.types import GuardrailVerdict
from guardrails.base import Guardrail

#: Overt harm-seeking intent. Matched as whole phrases, not loose keywords: "how to make
#: a bomb" is blocked, while "bombe ki kahani" (a story about a bomb) is not — a corpus
#: of translated news passages contains plenty of violent *subject matter*, and blocking
#: on the noun alone would refuse legitimate questions about it.
HARM_PATTERNS: tuple[str, ...] = (
    r"\bhow (?:do i|to|can i)\s+(?:make|build|synthesi[sz]e)\s+(?:a\s+)?(?:bomb|explosive|nerve agent|meth)",
    r"\bbomb\s+(?:banane|banana)\s+(?:ka|ki)\s+(?:tarika|tareeka|vidhi)",
    r"\bhow (?:do i|to|can i)\s+(?:kill|murder|poison)\s+(?:a\s+|my\s+|someone|him|her|them)",
    r"\b(?:kaise|kaisay)\s+(?:maar|maaru|marun)\s*(?:doon|dun|du)\b",
    r"\bhow (?:do i|to|can i)\s+(?:hack|break into)\s+(?:someone|his|her|their|a)\b",
    r"\b(?:child|minor)\s+(?:porn|sexual)",
    r"\bhow (?:do i|to|can i)\s+(?:hurt|harm)\s+myself\b",
    r"\b(?:suicide|khudkushi|aatmahatya)\s+(?:method|tarika|kaise|karne)\b",
)

#: Prompt-injection phrasing. The realistic attack surface on a voice-driven LLM demo.
INJECTION_PATTERNS: tuple[str, ...] = (
    r"\bignore\s+(?:all\s+|your\s+|the\s+|previous\s+|above\s+)*(?:instructions?|prompts?|rules?)",
    r"\bdisregard\s+(?:all\s+|your\s+|the\s+|previous\s+)*(?:instructions?|prompts?|rules?)",
    r"\b(?:you are now|from now on,? you)\b.{0,40}\b(?:not|no longer)\b",
    r"\b(?:system|developer)\s+prompt\b",
    r"\breveal\s+(?:your\s+)?(?:prompt|instructions?|system message)",
    r"\bpurani?\s+(?:sabhi\s+)?(?:instructions?|nirdesh)\s+(?:bhool|ignore|chhod)",
    r"\bact as\b.{0,30}\b(?:dan|jailbreak|unrestricted|no restrictions)\b",
)

_HARM_RE = re.compile("|".join(HARM_PATTERNS), re.IGNORECASE)
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

#: Transcripts longer than this are rejected before embedding. A 2000-character
#: "question" from a speech recogniser is a malfunction or an injection payload, not a
#: question, and embedding it wastes the latency budget either way.
MAX_QUERY_CHARS = 600


class InputSafetyGuardrail(Guardrail):
    name = "input_safety"
    fail_open = False  # the one guardrail where erroring should withhold, not allow

    def __init__(self, max_chars: int = MAX_QUERY_CHARS) -> None:
        self.max_chars = max_chars

    def _check(self, query: str) -> GuardrailVerdict:  # type: ignore[override]
        normalised = _normalise(query)

        if len(normalised) > self.max_chars:
            return self._block(
                reason=(
                    f"Transcript is {len(normalised)} characters (limit {self.max_chars}) — "
                    "too long to be a spoken question."
                ),
                score=float(len(normalised)), threshold=float(self.max_chars),
            )

        if _INJECTION_RE.search(normalised):
            return self._block(
                reason="Query contains prompt-injection phrasing; refusing before it reaches the model."
            )

        if _HARM_RE.search(normalised):
            return self._block(
                reason="Query solicits harmful instructions; this system will not answer it."
            )

        return self._pass(reason="No unsafe patterns matched.")


def _normalise(text: str) -> str:
    """NFKC-fold and collapse whitespace.

    Normalisation matters more than usual here: the input is a *speech transcript*, and
    Indic STT output routinely varies in Unicode composition for the same word (नुक़्ता
    as a combining mark vs. a precomposed codepoint). Without folding, a pattern that
    matches one spelling silently misses the other.
    """
    folded = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", folded).strip()


__all__ = ["InputSafetyGuardrail", "MAX_QUERY_CHARS"]
