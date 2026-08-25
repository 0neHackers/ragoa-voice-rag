"""Demo API tests.

The pipeline is stubbed rather than built for real: these assert the HTTP contract — that
declines return 200 with a machine-readable reason rather than an error status, that the
voice and text paths are distinguishable in the response, and that a missing index yields
a diagnostic instead of a crashed process. Pipeline behaviour itself is covered by
`test_harness.py` and `test_guardrails.py`.
"""

from __future__ import annotations

import io
import wave

import pytest
from fastapi.testclient import TestClient

import demo.app as demo_app
from harness.types import (
    Chunk, GuardrailVerdict, LatencyBreakdown, PipelineResponse, ResponseStatus,
    ScoredChunk, Transcript,
)


def _wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A real (silent) WAV, so the decode path runs rather than being mocked away."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _answered() -> PipelineResponse:
    latency = LatencyBreakdown(retrieval_ms=95.4, generation_ms=410.0, full_ms=505.4)
    latency.record("embed", 92.1)
    return PipelineResponse(
        status=ResponseStatus.ANSWERED,
        answer="भारत की राजधानी नई दिल्ली है।",
        query="भारत की राजधानी क्या है?",
        answer_mode="generated",
        retrieved=[ScoredChunk(
            chunk=Chunk(chunk_id="c1", text="नई दिल्ली भारत की राजधानी है।",
                        strategy="metadata_aware", char_len=28),
            score=0.031, dense_score=0.71, lexical_score=4.2, rank=1,
        )],
        guardrails=[GuardrailVerdict(name="low_confidence", passed=True, score=0.71,
                                     threshold=0.42, elapsed_ms=0.02)],
        latency=latency, trace_id="abc123def456",
    )


class StubPipeline:
    def __init__(self, response: PipelineResponse | None = None) -> None:
        self._response = response or _answered()
        self.text_calls: list[str] = []
        self.translate_flags: list[bool] = []
        self.audio_calls = 0
        self.guardrails = type("G", (), {
            "describe": lambda self: [{"name": "low_confidence", "position": "post_retrieval"}]
        })()
        self.stt = type("S", (), {"config": type("C", (), {"language_code": "hi-IN"})()})()

    def health(self) -> dict:
        return {"index_size": 15449, "index_backend": "faiss", "lexical_size": 15449,
                "embed_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "stt_configured": True, "generation_enabled": True,
                "guardrails_enabled": True, "retrieval_budget_ms": 200.0}

    def run_text(self, query, top_k=None, translate=False):
        self.text_calls.append(query)
        self.translate_flags.append(translate)
        return self._response

    def run_audio(self, audio, top_k=None):
        self.audio_calls += 1
        response = self._response.model_copy(deep=True)
        response.transcript = Transcript(
            text="भारत की राजधानी क्या है?", language_code="hi-IN", provider="sarvam",
            model="saarika:v2.5", is_real_audio=True, audio_duration_s=1.0,
            transport="streaming",
        )
        return response


@pytest.fixture
def client(monkeypatch):
    stub = StubPipeline()
    monkeypatch.setattr(demo_app, "_pipeline", stub)
    monkeypatch.setattr(demo_app, "_startup_error", None)
    with TestClient(demo_app.app) as c:
        c.stub = stub  # type: ignore[attr-defined]
        yield c


class TestHealth:
    def test_reports_index_and_config(self, client):
        body = client.get("/health").json()
        assert body["ok"] is True
        assert body["index_size"] == 15449
        assert body["stt_configured"] is True

    def test_503_with_diagnostic_when_startup_failed(self, monkeypatch, tmp_path):
        """A missing index must not crash the process on boot.

        This runs the real startup path against a directory that does not exist, so it
        asserts the actual failure behaviour rather than a stubbed message: the app
        starts, `/health` returns 503, and the error names the index and the command
        that builds it. A container that exits on boot gives a deploy platform nothing
        to show and turns a missing file into an opaque crash loop.
        """
        monkeypatch.setattr(demo_app, "_pipeline", None)
        monkeypatch.setattr(demo_app, "_startup_error", None)
        monkeypatch.setenv("INDEX_DIR", str(tmp_path / "does_not_exist"))

        with TestClient(demo_app.app) as c:
            response = c.get("/health")

        assert response.status_code == 503
        body = response.json()
        assert body["ok"] is False
        assert "does_not_exist" in body["error"]
        assert "build_index" in body["error"]  # tells the operator how to fix it


class TestTextPath:
    def test_returns_answer_and_both_latencies(self, client):
        body = client.post("/ask/text", json={"query": "भारत की राजधानी क्या है?"}).json()
        assert body["status"] == "answered"
        assert body["answer"]
        assert body["latency"]["retrieval_ms"] == 95.4
        assert body["latency"]["full_ms"] == 505.4

    def test_marked_as_not_voice_input(self, client):
        """A typed question must never be presentable as a spoken one."""
        body = client.post("/ask/text", json={"query": "कुछ"}).json()
        assert body["voice_input"] is False
        assert body["transcript"] is None

    def test_empty_query_rejected_by_validation(self, client):
        assert client.post("/ask/text", json={"query": ""}).status_code == 422

    def test_decline_is_200_with_a_reason(self, client):
        """A decline is a designed outcome, not an HTTP error — the client renders it."""
        client.stub._response = PipelineResponse.declined(
            reason="low_confidence", query="q",
            guardrails=[GuardrailVerdict(name="low_confidence", passed=False, score=0.19,
                                         threshold=0.42, reason="Best passage scored 0.19")],
            latency=LatencyBreakdown(retrieval_ms=88.0, full_ms=88.0),
        )
        response = client.post("/ask/text", json={"query": "अज्ञात"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "declined"
        assert body["reason"] == "low_confidence"
        assert body["answer"] is None
        assert not body["guardrails"][0]["passed"]

    def test_retrieved_passages_are_exposed_with_score_breakdown(self, client):
        body = client.post("/ask/text", json={"query": "q"}).json()
        passage = body["retrieved"][0]
        assert passage["dense_score"] == 0.71
        assert passage["lexical_score"] == 4.2
        assert passage["strategy"] == "metadata_aware"


class TestVoicePath:
    def test_wav_upload_transcribes_and_answers(self, client):
        response = client.post("/ask/voice", files={"audio": ("q.wav", _wav_bytes(), "audio/wav")})
        assert response.status_code == 200
        body = response.json()
        assert body["voice_input"] is True
        assert body["transcript"]["is_real_audio"] is True
        assert body["transcript"]["transport"] == "streaming"
        assert client.stub.audio_calls == 1

    def test_empty_upload_rejected(self, client):
        response = client.post("/ask/voice", files={"audio": ("q.wav", b"", "audio/wav")})
        assert response.status_code == 400

    def test_undecodable_audio_returns_a_useful_message(self, client):
        response = client.post(
            "/ask/voice", files={"audio": ("q.webm", b"\x1aE\xdf\xa3not audio", "audio/webm")}
        )
        assert response.status_code == 400
        assert "decode" in response.json()["detail"].lower()

    def test_language_override_reaches_the_stt_client(self, client):
        client.post("/ask/voice",
                    files={"audio": ("q.wav", _wav_bytes(), "audio/wav")},
                    data={"language": "en-IN"})
        assert client.stub.stt.config.language_code == "en-IN"


class TestMeta:
    def test_guardrails_endpoint_describes_the_suite(self, client):
        body = client.get("/guardrails").json()
        assert body["enabled"] is True
        assert body["guardrails"][0]["name"] == "low_confidence"

    def test_root_serves_the_ui(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Voice RAG" in response.text
        # The Devanagari face is load-bearing, not styling: JetBrains Mono and Cal Sans
        # have no Devanagari glyphs and every passage this app renders is Hindi.
        assert "Noto+Sans+Devanagari" in response.text
        # Same-origin by default; config.js only matters if the frontend is split off.
        assert "__API_BASE__" in response.text


class TestTranslateOption:
    """English→Hindi translation is opt-in, and the flag has to reach the pipeline."""

    def test_defaults_to_off(self, client):
        client.post("/ask/text", json={"query": "what is honesty"})
        assert client.stub.translate_flags == [False]

    def test_flag_is_forwarded(self, client):
        client.post("/ask/text", json={"query": "what is honesty", "translate": True})
        assert client.stub.translate_flags == [True]


class TestSpeakEndpoint:
    def test_rejects_empty_text(self, client):
        assert client.post("/speak", json={"text": ""}).status_code == 422

    def test_rejects_text_over_the_tts_limit(self, client):
        """Sarvam caps a TTS input at 1500 characters; reject before the round trip."""
        assert client.post("/speak", json={"text": "क" * 1600}).status_code == 422
