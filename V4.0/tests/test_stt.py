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
