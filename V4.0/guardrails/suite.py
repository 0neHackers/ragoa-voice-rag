"""The guardrail suite — what the orchestrator actually talks to.

Groups the four guardrails by pipeline position and runs each group in the order the
orchestrator expects. The grouping is the whole interface: the orchestrator asks for
"the pre-retrieval checks" rather than knowing which guardrails exist, so adding a fifth
guardrail is a change to this file only.

Ordering inside `run_pre_retrieval` is not arbitrary — input safety runs first. A
harmful query that happens to be written in the corpus's language must still be refused
as unsafe, with `input_safety` as the reason, rather than passing the language check and
being declined later for something less specific.
"""

from __future__ import annotations

import numpy as np

from harness.types import Answer, GuardrailVerdict, RetrievalResult
from guardrails.confidence import ConfidenceGateGuardrail
from guardrails.groundedness import GroundednessGuardrail
from guardrails.input_safety import InputSafetyGuardrail
from guardrails.language_match import LanguageMatchGuardrail


class GuardrailSuite:
    """Four guardrails at three pipeline positions. See DECISIONS.md D7."""

    def __init__(
        self,
        input_safety: InputSafetyGuardrail | None = None,
        language_match: LanguageMatchGuardrail | None = None,
        confidence: ConfidenceGateGuardrail | None = None,
        groundedness: GroundednessGuardrail | None = None,
    ) -> None:
        self.input_safety = input_safety or InputSafetyGuardrail()
        self.language_match = language_match
        self.confidence = confidence or ConfidenceGateGuardrail()
        self.groundedness = groundedness or GroundednessGuardrail()

    # -- construction ----------------------------------------------------- #

    @classmethod
    def from_retriever(cls, retriever: object, **overrides: object) -> "GuardrailSuite":
        """Build a suite wired to a live index.

        The language guardrail infers the corpus script from the indexed chunks, which
        only the retriever can supply — hence this constructor rather than a bare
        `GuardrailSuite()`.
        """
        language_match: LanguageMatchGuardrail | None = None
        try:
            language_match = LanguageMatchGuardrail.from_chunks(retriever.chunks)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            # An index whose script cannot be determined gets no language gate, rather
            # than one guessing. The other three guardrails still run.
            language_match = None

        return cls(language_match=language_match, **overrides)  # type: ignore[arg-type]

    # -- pipeline positions ----------------------------------------------- #

    def run_pre_retrieval(self, query: str, query_vector: np.ndarray) -> list[GuardrailVerdict]:
        """Runs before any retrieval cost is paid. Short-circuits on the first block."""
        verdicts = [self.input_safety.run(query)]
        if not verdicts[0].passed:
            return verdicts

        if self.language_match is not None:
            verdicts.append(self.language_match.run(query, query_vector))
        return verdicts

    def run_post_retrieval(self, query: str, retrieval: RetrievalResult) -> GuardrailVerdict:
        """The gate between retrieval and generation."""
        return self.confidence.run(query, retrieval)

    def run_post_generation(
        self, query: str, answer: Answer, retrieval: RetrievalResult
    ) -> GuardrailVerdict:
        """The last check before an answer reaches the user."""
        return self.groundedness.run(query, answer, retrieval)

    # -- introspection ----------------------------------------------------- #

    def describe(self) -> list[dict[str, object]]:
        """What the demo's /guardrails endpoint reports."""
        rows: list[dict[str, object]] = [
            {"name": self.input_safety.name, "position": "pre_retrieval",
             "threshold": self.input_safety.max_chars, "fail_open": self.input_safety.fail_open},
        ]
        if self.language_match is not None:
            rows.append({
                "name": self.language_match.name, "position": "pre_retrieval",
                "corpus_script": self.language_match.corpus_script,
                "enabled": self.language_match.enabled, "fail_open": True,
            })
        rows.append({
            "name": self.confidence.name, "position": "post_retrieval",
            "threshold": self.confidence.min_top_score,
            "min_margin": self.confidence.min_margin, "fail_open": True,
        })
        rows.append({
            "name": self.groundedness.name, "position": "post_generation",
            "threshold": self.groundedness.min_overlap,
            "uses_llm": self.groundedness.use_llm, "fail_open": True,
        })
        return rows


__all__ = ["GuardrailSuite"]
