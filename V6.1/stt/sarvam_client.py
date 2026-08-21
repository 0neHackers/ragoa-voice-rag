"""Sarvam AI speech-to-text client.

Primary transport is the **streaming WebSocket** endpoint, per DECISIONS.md D1 — the
batch endpoint is built for long-file transcription and its round trip alone would eat
most of the pipeline's latency budget.

The batch REST endpoint is kept as an explicit, *labelled* fallback for one reason: a
WebSocket handshake is the single most environment-fragile call in this pipeline
(corporate proxies, captive wifi, and some PaaS egress rules block `wss://` while
allowing `https://`). When the fallback fires, `Transcript.transport` says `"batch"`,
so a demo never silently claims to be streaming when it wasn't.

Both paths transcribe real audio bytes. There is no text-passthrough path in this file.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from harness.retry import RetryPolicy, retry_async, retry_sync
from harness.types import ErrorKind, Stage, StageError, StageResult, Timer, Transcript

SARVAM_WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"
SARVAM_REST_URL = "https://api.sarvam.ai/speech-to-text"

DEFAULT_MODEL = "saarika:v2.5"
TARGET_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class SarvamConfig:
    api_key: str
    language_code: str = "hi-IN"
    model: str = DEFAULT_MODEL
    sample_rate: int = TARGET_SAMPLE_RATE
    ws_timeout_s: float = 20.0
    chunk_ms: int = 100  # audio frames pushed per streaming message
    #: How long to hold the socket open after the last audio frame so the recogniser can
    #: flush its final segment. Closing immediately truncates the tail of the utterance.
    flush_grace_s: float = 1.5

    @classmethod
    def from_env(cls, **overrides: object) -> "SarvamConfig":
        key = os.getenv("SARVAM_API_KEY", "").strip()
        lang = os.getenv("STT_LANGUAGE_CODE", "hi-IN").strip()
        model = os.getenv("SARVAM_STT_MODEL", DEFAULT_MODEL).strip()
        cfg = cls(api_key=key, language_code=lang, model=model)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class Audio:
    """16-bit mono PCM at a known sample rate — the only shape both endpoints accept."""

    pcm: bytes
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / (2 * self.sample_rate)

    def to_wav_bytes(self) -> bytes:
        import io

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(self.pcm)
        return buf.getvalue()


def load_audio_file(source: str | Path | IO[bytes]) -> Audio:
    """Read audio into 16kHz mono 16-bit PCM.

    Accepts a path or an open binary stream — the demo server hands it an uploaded
    file's `BytesIO` directly rather than staging a temp file per request.

    Resampling happens here rather than being left to the provider because Sarvam's
    models expect 16kHz, and handing them 44.1kHz audio degrades accuracy on Indic
    phonemes.

    Note on formats: this decodes what libsndfile supports — WAV, FLAC, OGG/Opus, MP3.
    It does **not** decode WebM, which is what Chrome's `MediaRecorder` produces by
    default. The browser client sidesteps that by encoding WAV itself via the Web Audio
    API rather than relying on a server-side transcode, so the demo has no ffmpeg
    dependency to install on a deploy target.
    """
    import numpy as np
    import soundfile as sf

    handle = source if hasattr(source, "read") else str(source)
    data, sr = sf.read(handle, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # downmix; stereo mics would otherwise halve amplitude

    if sr != TARGET_SAMPLE_RATE:
        # Linear resample. Adequate for speech at these rates and avoids pulling in
        # scipy/librosa for a single interpolation.
        n_out = int(round(len(mono) * TARGET_SAMPLE_RATE / sr))
        mono = np.interp(
            np.linspace(0.0, len(mono) - 1, n_out, dtype=np.float64),
            np.arange(len(mono), dtype=np.float64),
            mono,
        ).astype("float32")
        sr = TARGET_SAMPLE_RATE

    pcm16 = np.clip(mono, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    return Audio(pcm=pcm16.tobytes(), sample_rate=sr)


def record_microphone(seconds: float = 6.0, sample_rate: int = TARGET_SAMPLE_RATE) -> Audio:
    """Capture live mic audio. Raises if no input device exists — never returns silence."""
    import numpy as np
    import sounddevice as sd

    frames = int(seconds * sample_rate)
    buf = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    mono = buf.reshape(-1)

    peak = float(np.abs(mono).max()) if mono.size else 0.0
    if peak < 1e-4:
        raise RuntimeError(
            "Microphone captured near-silence (peak amplitude "
            f"{peak:.2e}). Check the input device and OS mic permissions — "
            "transcribing this would produce an empty 'voice' demo."
        )

    pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    return Audio(pcm=pcm16.tobytes(), sample_rate=sample_rate)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

class SarvamSTT:
    """Transcribes `Audio` and returns a `StageResult[Transcript]` — never raises."""

    #: Set once a streaming attempt opens fine and returns nothing. Class-level on
    #: purpose: it's a property of the account, not of one client instance, and the demo
    #: server builds its client once but serves many requests.
    _streaming_known_dead: bool = False

    def __init__(self, config: SarvamConfig | None = None) -> None:
        self.config = config or SarvamConfig.from_env()
        self.retry = RetryPolicy(max_attempts=3, base_delay_s=0.25, max_delay_s=2.0)
        # auto  — try streaming, fall back to batch and latch (default)
        # batch — skip streaming entirely
        # streaming — no fallback; fail loudly instead. Useful for verifying the
        #             streaming path once Sarvam enables it on the account.
        self.transport_mode = os.getenv("STT_TRANSPORT", "auto").strip().lower()

    # -- public ------------------------------------------------------------- #

    def transcribe(self, audio: Audio, *, allow_batch_fallback: bool = True) -> StageResult[Transcript]:
        with Timer() as t:
            result = self._transcribe_inner(audio, allow_batch_fallback)
        result.elapsed_ms = t.ms
        return result

    def _transcribe_inner(self, audio: Audio, allow_batch_fallback: bool) -> StageResult[Transcript]:
        if not self.config.api_key:
            return StageResult[Transcript](
                stage=Stage.STT,
                error=StageError(
                    stage=Stage.STT,
                    kind=ErrorKind.CONFIG,
                    message=(
                        "SARVAM_API_KEY is not set. Speech-to-text cannot run. This pipeline "
                        "has no typed-text substitute for the voice stage, by design — set "
                        "the key in .env (see .env.example)."
                    ),
                ),
            )

        if audio.duration_s < 0.15:
            return StageResult[Transcript](
                stage=Stage.STT,
                error=StageError(
                    stage=Stage.STT,
                    kind=ErrorKind.VALIDATION,
                    message=f"Audio is only {audio.duration_s:.3f}s — too short to contain a question.",
                ),
            )

        if self.transport_mode == "batch" or SarvamSTT._streaming_known_dead:
            # Either configured for batch, or an earlier request in this process already
            # established that streaming yields nothing on this account. Re-probing it
            # per request costs ~1.5s of dead time for a result we already know.
            return self._batch(audio)

        ws_err: StageError | None = None
        try:
            streamed = asyncio.run(self._stream(audio))
            if streamed.ok:
                return streamed
            ws_err = streamed.error
        except Exception as exc:  # noqa: BLE001 - transport failures become typed errors
            ws_err = StageError(
                stage=Stage.STT,
                kind=_classify(exc),
                message=f"Streaming transcription failed: {exc}",
                detail={"transport": "streaming"},
            )

        assert ws_err is not None
        if not allow_batch_fallback or self.transport_mode == "streaming":
            return StageResult[Transcript](stage=Stage.STT, error=ws_err)
        if ws_err.kind is ErrorKind.AUTH:
            # An auth failure fails identically over REST — don't burn the budget.
            return StageResult[Transcript](stage=Stage.STT, error=ws_err)

        # A socket that opens, accepts every frame, and returns nothing is not a
        # transient blip — it's this account's streaming tier. Latch it so the rest of
        # the process goes straight to the endpoint that works.
        if ws_err.kind is ErrorKind.UPSTREAM:
            SarvamSTT._streaming_known_dead = True

        batch = self._batch(audio)
        if batch.error is not None:
            batch.error.detail["streaming_error"] = ws_err.message
        return batch

    # -- streaming (primary) ------------------------------------------------- #

    async def _stream(self, audio: Audio) -> StageResult[Transcript]:
        import websockets

        cfg = self.config
        url = f"{SARVAM_WS_URL}?language-code={cfg.language_code}&model={cfg.model}"
        headers = {"api-subscription-key": cfg.api_key}

        bytes_per_chunk = int(cfg.sample_rate * 2 * cfg.chunk_ms / 1000)
        partials: list[str] = []
        finals: list[str] = []
        upstream_errors: list[str] = []

        async def _run() -> None:
            async with websockets.connect(
                url, additional_headers=headers, open_timeout=cfg.ws_timeout_s
            ) as ws:
                async def _send() -> None:
                    for off in range(0, len(audio.pcm), bytes_per_chunk):
                        frame = audio.pcm[off : off + bytes_per_chunk]
                        await ws.send(json.dumps({
                            "audio": {
                                "data": base64.b64encode(frame).decode("ascii"),
                                "encoding": "audio/wav",
                                "sample_rate": cfg.sample_rate,
                            }
                        }))
                        # Pace sends to roughly real time. Blasting the whole clip at once
                        # is what makes a "streaming" demo indistinguishable from batch.
                        await asyncio.sleep(cfg.chunk_ms / 1000 * 0.25)

                    # No terminator frame. Sarvam validates *every* message against its
                    # audio-request schema, so a `{"event": "stop"}` sentinel — which is
                    # what this client sent originally, and what several other streaming
                    # APIs document — comes back as
                    # `Invalid request: 'audio' must not be None`. Verified against the
                    # live endpoint. The stream ends by closing the socket.

                async def _recv() -> None:
                    try:
                        async for raw in ws:
                            msg = json.loads(raw) if isinstance(raw, (str, bytes)) else {}

                            if msg.get("type") == "error":
                                detail = msg.get("data") or {}
                                upstream_errors.append(
                                    str(detail.get("message") or detail)[:300]
                                )
                                continue

                            text, is_final = _extract_transcript(msg)
                            if not text:
                                if msg.get("type") in ("end", "close"):
                                    break
                                continue
                            (finals if is_final else partials).append(text)
                    except Exception:  # noqa: BLE001 - normal close races the iterator
                        pass

                # Send everything, then give the recogniser a bounded window to flush.
                #
                # This cannot be a plain `gather(_send(), _recv())`. Sarvam never closes
                # the socket on its own, so `async for raw in ws` waits forever and the
                # gather only ends when the whole coroutine hits its outer timeout — 34
                # seconds, measured, after which the batch fallback quietly rescued the
                # request and every transcript came back labelled "batch". The streaming
                # path looked broken when it was really just never being allowed to end.
                receiver = asyncio.create_task(_recv())
                await _send()
                try:
                    await asyncio.wait_for(asyncio.shield(receiver), timeout=cfg.flush_grace_s)
                except asyncio.TimeoutError:
                    pass  # grace elapsed — whatever arrived is what we have
                receiver.cancel()

        await asyncio.wait_for(
            _run(), timeout=cfg.ws_timeout_s + audio.duration_s + cfg.flush_grace_s + 10
        )

        # An error frame is the provider telling us why it produced nothing. Reporting
        # "no transcript" over the top of it throws away the only useful diagnostic —
        # which is exactly how the bad terminator above survived until it was tested
        # against the live API.
        if upstream_errors and not (finals or partials):
            raise UpstreamProtocolError("; ".join(upstream_errors[:3]))

        text = " ".join(finals).strip() or (partials[-1].strip() if partials else "")
        if not text:
            return StageResult[Transcript](
                stage=Stage.STT,
                error=StageError(
                    stage=Stage.STT,
                    kind=ErrorKind.UPSTREAM,
                    message="Streaming endpoint returned no transcript for this audio.",
                    detail={"transport": "streaming", "audio_s": round(audio.duration_s, 2)},
                ),
            )

        return StageResult[Transcript](
            stage=Stage.STT,
            value=Transcript(
                text=text,
                language_code=self.config.language_code,
                provider="sarvam",
                model=self.config.model,
                is_real_audio=True,
                audio_duration_s=round(audio.duration_s, 3),
                transport="streaming",
                partials=partials[-5:],
            ),
        )

    # -- batch (labelled fallback) ------------------------------------------- #

    def _batch(self, audio: Audio) -> StageResult[Transcript]:
        import requests

        cfg = self.config

        def _call() -> requests.Response:
            resp = requests.post(
                SARVAM_REST_URL,
                headers={"api-subscription-key": cfg.api_key},
                files={"file": ("query.wav", audio.to_wav_bytes(), "audio/wav")},
                data={"language_code": cfg.language_code, "model": cfg.model},
                timeout=30,
            )
            if resp.status_code >= 500 or resp.status_code == 429:
                raise TransientHTTPError(resp.status_code, resp.text[:200])
            return resp

        try:
            resp = retry_sync(_call, self.retry)
        except Exception as exc:  # noqa: BLE001
            return StageResult[Transcript](
                stage=Stage.STT,
                error=StageError(
                    stage=Stage.STT, kind=_classify(exc),
                    message=f"Batch transcription failed: {exc}",
                    detail={"transport": "batch"},
                    attempts=self.retry.max_attempts,
                ),
            )

        if resp.status_code in (401, 403):
            return StageResult[Transcript](
                stage=Stage.STT,
                error=StageError(stage=Stage.STT, kind=ErrorKind.AUTH,
                                 message=f"Sarvam rejected the API key (HTTP {resp.status_code})."),
            )
        if resp.status_code >= 400:
            return StageResult[Transcript](
                stage=Stage.STT,
                error=StageError(stage=Stage.STT, kind=ErrorKind.UPSTREAM,
                                 message=f"Sarvam returned HTTP {resp.status_code}: {resp.text[:200]}"),
            )

        body = resp.json()
        text = (body.get("transcript") or body.get("text") or "").strip()
        if not text:
            return StageResult[Transcript](
                stage=Stage.STT,
                error=StageError(stage=Stage.STT, kind=ErrorKind.UPSTREAM,
                                 message="Sarvam returned an empty transcript."),
            )

        return StageResult[Transcript](
            stage=Stage.STT,
            value=Transcript(
                text=text,
                language_code=body.get("language_code") or cfg.language_code,
                provider="sarvam",
                model=cfg.model,
                is_real_audio=True,
                audio_duration_s=round(audio.duration_s, 3),
                transport="batch",
            ),
        )


class UpstreamProtocolError(RuntimeError):
    """Sarvam sent a `type: "error"` frame. Carries its message so it can be acted on."""


class TransientHTTPError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status


def _extract_transcript(msg: dict) -> tuple[str, bool]:
    """Pull text out of a streaming frame.

    Written tolerantly, and for a reason: Sarvam has shipped more than one frame shape across
    model versions (`{"type":"data","data":{"transcript":...}}` and flatter variants),
    and a demo should not die because a key moved one level.
    """
    if not isinstance(msg, dict):
        return "", False

    data = msg.get("data") if isinstance(msg.get("data"), dict) else msg
    for key in ("transcript", "text", "partial_transcript"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            is_final = bool(
                data.get("is_final")
                or msg.get("is_final")
                or msg.get("type") in ("data", "final", "transcript")
                and key != "partial_transcript"
            )
            return val.strip(), is_final
    return "", False


def _classify(exc: BaseException) -> ErrorKind:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(s in text for s in ("401", "403", "unauthorized", "forbidden", "invalid api")):
        return ErrorKind.AUTH
    if "429" in text or "rate limit" in text:
        return ErrorKind.RATE_LIMITED
    if any(s in text for s in ("timeout", "timed out", "connection", "refused",
                               "handshake", "dns", "unreachable", "eof", "reset", "502", "503")):
        return ErrorKind.TRANSIENT
    return ErrorKind.UPSTREAM


__all__ = [
    "Audio", "SarvamConfig", "SarvamSTT", "UpstreamProtocolError",
    "load_audio_file", "record_microphone",
]
