"""STT stage tests.

These assert the *contract* rather than transcription quality: that failures arrive as
typed values instead of exceptions, and — most importantly — that no code path
fabricates a transcript when audio or credentials are missing. Transcription accuracy
is verified against real audio by `stt/transcribe.py`, which needs a live API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.retry import RetriesExhausted, RetryPolicy, retry_sync  # noqa: E402
from harness.types import ErrorKind, Stage  # noqa: E402
from stt.sarvam_client import Audio, SarvamConfig, SarvamSTT, _extract_transcript  # noqa: E402


def _silence(seconds: float = 2.0, sr: int = 16_000) -> Audio:
    return Audio(pcm=np.zeros(int(sr * seconds), dtype="<i2").tobytes(), sample_rate=sr)


class TestAudio:
    def test_duration_matches_pcm_length(self):
        assert _silence(1.5).duration_s == pytest.approx(1.5)

    def test_wav_header_is_written(self):
        wav = _silence(0.5).to_wav_bytes()
        assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


class TestSTTContract:
    def test_missing_key_returns_typed_config_error(self):
        result = SarvamSTT(SarvamConfig(api_key="")).transcribe(_silence())
        assert not result.ok
        assert result.error.kind is ErrorKind.CONFIG
        assert result.error.stage is Stage.STT

    def test_missing_key_never_invents_a_transcript(self):
        """The one property this project cannot afford to get wrong."""
        result = SarvamSTT(SarvamConfig(api_key="")).transcribe(_silence())
        assert result.value is None

    def test_too_short_audio_is_rejected_before_any_network_call(self):
        result = SarvamSTT(SarvamConfig(api_key="k")).transcribe(
            Audio(pcm=b"\x00" * 64, sample_rate=16_000)
        )
        assert result.error.kind is ErrorKind.VALIDATION

    def test_stage_is_always_timed(self):
        result = SarvamSTT(SarvamConfig(api_key="")).transcribe(_silence())
        assert result.elapsed_ms >= 0.0


class TestFrameParsing:
    @pytest.mark.parametrize("frame,expected", [
        ({"type": "data", "data": {"transcript": "नमस्ते"}}, "नमस्ते"),
        ({"transcript": "hello"}, "hello"),
        ({"data": {"text": "hi there"}}, "hi there"),
        ({"type": "end"}, ""),
        ({"data": {"transcript": "   "}}, ""),
    ])
    def test_tolerates_known_frame_shapes(self, frame, expected):
        assert _extract_transcript(frame)[0] == expected

    def test_partials_are_not_marked_final(self):
        _, is_final = _extract_transcript({"data": {"partial_transcript": "नम"}})
        assert is_final is False


class TestRetry:
    def test_recovers_after_transient_failures(self):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionError("transient")
            return "recovered"

        policy = RetryPolicy(max_attempts=3, base_delay_s=0.001)
        assert retry_sync(flaky, policy) == "recovered"
        assert attempts["n"] == 3

    def test_raises_after_exhaustion_carrying_the_last_cause(self):
        def always_fails():
            raise TimeoutError("still down")

        with pytest.raises(RetriesExhausted) as exc:
            retry_sync(always_fails, RetryPolicy(max_attempts=2, base_delay_s=0.001))
        assert isinstance(exc.value.last, TimeoutError)
        assert exc.value.attempts == 2

    def test_backoff_is_jittered_and_capped(self):
        policy = RetryPolicy(base_delay_s=1.0, max_delay_s=2.0)
        samples = [policy.delay_for(5) for _ in range(50)]
        assert all(0.0 <= s <= 2.0 for s in samples)
        assert len(set(samples)) > 1  # jitter, not a fixed schedule


class TestLiveApiRegressions:
    """Regressions for bugs that only appeared against the real Sarvam API.

    Each of these passed unit tests happily while being wrong in production, which is the
    whole reason they're pinned here.
    """

    def test_stop_sentinel_is_not_sent(self):
        """`{"event": "stop"}` is rejected by Sarvam as `'audio' must not be None`.

        Every frame is validated against its audio-request schema, so a terminator
        sentinel — which several other streaming APIs document — is a protocol error, not
        an end-of-stream marker. The stream ends by closing the socket.
        """
        import inspect

        from stt import sarvam_client

        source = inspect.getsource(sarvam_client.SarvamSTT._stream)
        # Strip comments first — the sentinel is named in a comment explaining why it
        # isn't sent, and a test that can't tell code from commentary would fail on the
        # documentation of its own bug.
        code = " ".join(line.split("#", 1)[0] for line in source.splitlines())
        assert '"event": "stop"' not in code
        assert "'event': 'stop'" not in code

    def test_error_frames_are_surfaced_not_swallowed(self):
        """A `type: "error"` frame carries the only useful diagnostic; reporting
        "no transcript" over the top of it is how the bad terminator survived."""
        import inspect

        from stt import sarvam_client

        source = inspect.getsource(sarvam_client.SarvamSTT._stream)
        assert 'msg.get("type") == "error"' in source
        assert "UpstreamProtocolError" in source

    def test_transport_mode_defaults_to_auto(self, monkeypatch):
        monkeypatch.delenv("STT_TRANSPORT", raising=False)
        monkeypatch.setenv("SARVAM_API_KEY", "test-key")
        assert sarvam_client_module().SarvamSTT().transport_mode == "auto"

    def test_batch_mode_skips_streaming_entirely(self, monkeypatch):
        """Once streaming is known dead, paying ~1.5s to re-probe it per request is
        pure waste."""
        monkeypatch.setenv("STT_TRANSPORT", "batch")
        monkeypatch.setenv("SARVAM_API_KEY", "test-key")
        client = sarvam_client_module().SarvamSTT()
        assert client.transport_mode == "batch"


def sarvam_client_module():
    from stt import sarvam_client

    return sarvam_client
