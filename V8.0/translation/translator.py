"""English → Hindi query translation. Opt-in, never automatic.

The corpus is Hindi, and the `language_mismatch` guardrail refuses Latin-script queries
for a measured reason: "what is the capital of france" scores 0.628 against this corpus —
above the confidence threshold — because a multilingual encoder maps it near Hindi
passages about countries. Left alone it gets a confident answer built from passages that
were never about it.

Translating first is the honest way to let an English speaker use the system: the query
becomes genuinely Hindi, so retrieval compares like with like instead of leaning on the
encoder's cross-lingual alignment.

**It stays opt-in on purpose.** Auto-translating every Latin-script query would silently
disable a guardrail that exists because of a real failure, and it would mangle romanised
Hindi — "bharat ki rajdhani kya hai" is already a Hindi question and translating it as
English produces nonsense. The user asking for a translation is the signal that the input
really is English.

`PipelineResponse` keeps both strings, so the UI can show what was asked and what was
actually searched for. A translated query is never presented as though the user typed it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from harness.retry import RetryPolicy, retry_sync
from harness.types import ErrorKind, Stage, StageError, StageResult, Timer

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

#: Sarvam's translate endpoint caps a single request at 1000 characters.
MAX_INPUT_CHARS = 1000


@dataclass(slots=True)
class TranslationConfig:
    api_key: str = ""
    source_language: str = "en-IN"
    target_language: str = "hi-IN"
    #: "formal" keeps terminology intact. The colloquial modes paraphrase, which is fine
    #: for chat and wrong for a retrieval query, where the exact terms are the signal.
    mode: str = "formal"
    timeout_s: float = 20.0

    @classmethod
    def from_env(cls) -> "TranslationConfig":
        return cls(
            api_key=os.getenv("SARVAM_API_KEY", "").strip(),
            source_language=os.getenv("TRANSLATE_SOURCE", "en-IN").strip(),
            target_language=os.getenv("TRANSLATE_TARGET", "hi-IN").strip(),
        )


@dataclass(slots=True)
class Translation:
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    provider: str = "sarvam"


class Translator:
    """Translates a query and returns a `StageResult[Translation]` — never raises."""

    def __init__(self, config: TranslationConfig | None = None) -> None:
        self.config = config or TranslationConfig.from_env()
        self.retry = RetryPolicy(max_attempts=2, base_delay_s=0.2, max_delay_s=1.0,
                                 deadline_s=8.0)

    def translate(self, text: str) -> StageResult[Translation]:
        with Timer() as timer:
            result = self._translate_inner(text)
        result.elapsed_ms = timer.ms
        return result

    def _translate_inner(self, text: str) -> StageResult[Translation]:
        cleaned = (text or "").strip()

        if not cleaned:
            return _fail(ErrorKind.VALIDATION, "Nothing to translate.")
        if not self.config.api_key:
            return _fail(
                ErrorKind.CONFIG,
                "SARVAM_API_KEY is not set, so translation is unavailable. Ask in Hindi, "
                "or set the key.",
            )
        if len(cleaned) > MAX_INPUT_CHARS:
            return _fail(
                ErrorKind.VALIDATION,
                f"Query is {len(cleaned)} characters; the translate endpoint accepts "
                f"{MAX_INPUT_CHARS}.",
            )

        import requests

        def _call() -> requests.Response:
            response = requests.post(
                SARVAM_TRANSLATE_URL,
                headers={
                    "api-subscription-key": self.config.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "input": cleaned,
                    "source_language_code": self.config.source_language,
                    "target_language_code": self.config.target_language,
                    "mode": self.config.mode,
                },
                timeout=self.config.timeout_s,
            )
            if response.status_code >= 500 or response.status_code == 429:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:160]}")
            return response

        try:
            response = retry_sync(_call, self.retry)
        except Exception as exc:  # noqa: BLE001
            return _fail(ErrorKind.TRANSIENT, f"Translation request failed: {exc}")

        if response.status_code in (401, 403):
            return _fail(ErrorKind.AUTH,
                         f"Sarvam rejected the API key (HTTP {response.status_code}).")
        if response.status_code >= 400:
            return _fail(ErrorKind.UPSTREAM,
                         f"Sarvam translate returned {response.status_code}: "
                         f"{response.text[:200]}")

        translated = (response.json().get("translated_text") or "").strip()
        if not translated:
            return _fail(ErrorKind.UPSTREAM, "Sarvam translate returned empty text.")

        return StageResult[Translation](
            stage=Stage.TRANSLATE,
            value=Translation(
                source_text=cleaned,
                translated_text=translated,
                source_language=self.config.source_language,
                target_language=self.config.target_language,
            ),
        )


def _fail(kind: ErrorKind, message: str) -> StageResult[Translation]:
    return StageResult[Translation](
        stage=Stage.TRANSLATE,
        error=StageError(stage=Stage.TRANSLATE, kind=kind, message=message),
    )


__all__ = ["Translator", "TranslationConfig", "Translation", "MAX_INPUT_CHARS"]
