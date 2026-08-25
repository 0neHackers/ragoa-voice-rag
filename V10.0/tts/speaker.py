"""Hindi text-to-speech, so the question and the answer can both be heard.

Closes the loop: you speak a question, and the system speaks its answer back. That matters
more than it sounds for a Hindi demo — a viewer who doesn't read Devanagari can still tell
the pipeline understood and answered.

Sarvam's `bulbul` TTS, same key as everything else.

Two constraints the endpoint imposes, both handled here rather than left to blow up at
request time:

- **1500 characters per input.** Answers are prompted to two or three sentences so they
  rarely approach it, but a long extractive passage can, and truncating mid-word sounds
  broken. `_split_for_tts` breaks on sentence boundaries — including the Devanagari danda,
  which a naive `.split('.')` misses entirely.
- **The speaker name must be one Sarvam recognises.** It rejects unknown names with a 400
  listing the valid set; `anushka` is the default here and was verified against the live
  API.

Audio comes back as base64 WAV and is returned as raw bytes for the caller to stream.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass

from harness.retry import RetryPolicy, retry_sync
from harness.types import ErrorKind, Stage, StageError, StageResult, Timer

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

#: Per-input ceiling on Sarvam's TTS endpoint.
MAX_TTS_CHARS = 1500

#: Verified against the live API. An unrecognised speaker is a hard 400.
DEFAULT_SPEAKER = "anushka"


@dataclass(slots=True)
class TTSConfig:
    api_key: str = ""
    language: str = "hi-IN"
    speaker: str = DEFAULT_SPEAKER
    sample_rate: int = 22050   # TTS playback, not the 16k the recogniser wants
    pace: float = 1.0
    timeout_s: float = 45.0

    @classmethod
    def from_env(cls) -> "TTSConfig":
        return cls(
            api_key=os.getenv("SARVAM_API_KEY", "").strip(),
            language=os.getenv("TTS_LANGUAGE", "hi-IN").strip(),
            speaker=os.getenv("TTS_SPEAKER", DEFAULT_SPEAKER).strip() or DEFAULT_SPEAKER,
            sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "22050")),
        )


@dataclass(slots=True)
class Speech:
    wav_bytes: bytes
    text: str
    language: str
    speaker: str
    segments: int = 1

    @property
    def size_kb(self) -> float:
        return len(self.wav_bytes) / 1024.0


class Speaker:
    """Synthesises speech and returns a `StageResult[Speech]` — never raises."""

    def __init__(self, config: TTSConfig | None = None) -> None:
        self.config = config or TTSConfig.from_env()
        self.retry = RetryPolicy(max_attempts=2, base_delay_s=0.3, max_delay_s=1.5,
                                 deadline_s=20.0)

    def speak(self, text: str) -> StageResult[Speech]:
        with Timer() as timer:
            result = self._speak_inner(text)
        result.elapsed_ms = timer.ms
        return result

    def _speak_inner(self, text: str) -> StageResult[Speech]:
        cleaned = _clean_for_speech(text)

        if not cleaned:
            return _fail(ErrorKind.VALIDATION, "Nothing to speak.")
        if not self.config.api_key:
            return _fail(ErrorKind.CONFIG,
                         "SARVAM_API_KEY is not set, so speech synthesis is unavailable.")

        segments = _split_for_tts(cleaned, MAX_TTS_CHARS)

        import requests

        def _call() -> requests.Response:
            response = requests.post(
                SARVAM_TTS_URL,
                headers={
                    "api-subscription-key": self.config.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": segments,
                    "target_language_code": self.config.language,
                    "speaker": self.config.speaker,
                    "speech_sample_rate": self.config.sample_rate,
                    "pace": self.config.pace,
                },
                timeout=self.config.timeout_s,
            )
            if response.status_code >= 500 or response.status_code == 429:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:160]}")
            return response

        try:
            response = retry_sync(_call, self.retry)
        except Exception as exc:  # noqa: BLE001
            return _fail(ErrorKind.TRANSIENT, f"Speech synthesis failed: {exc}")

        if response.status_code in (401, 403):
            return _fail(ErrorKind.AUTH,
                         f"Sarvam rejected the API key (HTTP {response.status_code}).")
        if response.status_code >= 400:
            return _fail(ErrorKind.UPSTREAM,
                         f"Sarvam TTS returned {response.status_code}: {response.text[:220]}")

        clips = response.json().get("audios") or []
        if not clips:
            return _fail(ErrorKind.UPSTREAM, "Sarvam TTS returned no audio.")

        # Multiple segments come back as separate WAVs. Concatenating whole files would
        # embed a 44-byte RIFF header mid-stream, which most players read as corruption —
        # so strip every header but the first and fix up the length fields.
        wav = _concat_wavs([base64.b64decode(c) for c in clips])

        return StageResult[Speech](
            stage=Stage.TTS,
            value=Speech(
                wav_bytes=wav, text=cleaned, language=self.config.language,
                speaker=self.config.speaker, segments=len(clips),
            ),
        )


_CITATION = re.compile(r"\[\d{1,2}\]")
_WS = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[।.!?])\s+")


def _clean_for_speech(text: str) -> str:
    """Strip what shouldn't be read aloud.

    Citation markers are the main one: `[1][2]` is useful on screen and becomes "one two"
    in the middle of a sentence when spoken.
    """
    return _WS.sub(" ", _CITATION.sub("", text or "")).strip()


def _split_for_tts(text: str, limit: int) -> list[str]:
    """Break text into segments under `limit`, on sentence boundaries where possible."""
    if len(text) <= limit:
        return [text]

    segments: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        if len(current) + len(sentence) + 1 <= limit:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            segments.append(current)
        # A single sentence longer than the limit still has to be cut somewhere.
        while len(sentence) > limit:
            segments.append(sentence[:limit])
            sentence = sentence[limit:]
        current = sentence
    if current:
        segments.append(current)
    return segments


def _concat_wavs(clips: list[bytes]) -> bytes:
    """Join WAV clips into one playable file."""
    if len(clips) == 1:
        return clips[0]

    import io
    import wave

    out = io.BytesIO()
    writer: wave.Wave_write | None = None
    try:
        for clip in clips:
            with wave.open(io.BytesIO(clip), "rb") as reader:
                if writer is None:
                    writer = wave.open(out, "wb")
                    writer.setnchannels(reader.getnchannels())
                    writer.setsampwidth(reader.getsampwidth())
                    writer.setframerate(reader.getframerate())
                writer.writeframes(reader.readframes(reader.getnframes()))
    finally:
        if writer is not None:
            writer.close()
    return out.getvalue()


def _fail(kind: ErrorKind, message: str) -> StageResult[Speech]:
    return StageResult[Speech](
        stage=Stage.TTS,
        error=StageError(stage=Stage.TTS, kind=kind, message=message),
    )


__all__ = ["Speaker", "TTSConfig", "Speech", "MAX_TTS_CHARS", "DEFAULT_SPEAKER"]
