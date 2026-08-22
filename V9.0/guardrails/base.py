"""Shared plumbing for guardrails.

Every guardrail is a small object with a `name`, a threshold it can state, and a
`check()` that returns a `GuardrailVerdict` — never a bare bool. The verdict carries the
score and the threshold that produced it so a decline can be explained ("top cosine 0.31
< 0.42"), which is the difference between a system that refuses and a system that can
show *why* it refused.

Guardrails never raise. A guardrail that crashes must not take down a request — if a
check errors, it fails **open** with the error recorded in the verdict's reason. That
direction is deliberate: these are quality gates over a public Q&A demo, not a security
boundary, so an internal bug degrading into "answer anyway, flagged" is better than
degrading into "refuse everything". The one exception is `input_safety`, which fails
closed, because there the whole point is to withhold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from harness.types import GuardrailVerdict, Timer


class Guardrail(ABC):
    """Base class. Subclasses implement `_check`; `run` handles timing and errors."""

    name: str = "guardrail"
    fail_open: bool = True

    @abstractmethod
    def _check(self, *args: Any, **kwargs: Any) -> GuardrailVerdict:
        """Return a verdict. May raise — `run` converts it."""

    def run(self, *args: Any, **kwargs: Any) -> GuardrailVerdict:
        with Timer() as timer:
            try:
                verdict = self._check(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - a guardrail bug must not 500 the request
                verdict = GuardrailVerdict(
                    name=self.name,
                    passed=self.fail_open,
                    reason=(
                        f"guardrail error ({type(exc).__name__}: {exc}); "
                        f"failed {'open' if self.fail_open else 'closed'}"
                    ),
                )
        verdict.elapsed_ms = timer.ms
        return verdict

    # convenience constructors ------------------------------------------- #

    def _pass(self, score: float | None = None, threshold: float | None = None,
              reason: str | None = None) -> GuardrailVerdict:
        return GuardrailVerdict(name=self.name, passed=True, score=score,
                                threshold=threshold, reason=reason)

    def _block(self, reason: str, score: float | None = None,
               threshold: float | None = None) -> GuardrailVerdict:
        return GuardrailVerdict(name=self.name, passed=False, score=score,
                                threshold=threshold, reason=reason)


__all__ = ["Guardrail"]
