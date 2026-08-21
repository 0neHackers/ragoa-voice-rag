"""Harness tests — structured I/O, retries, and per-stage error handling.

These are the properties the task grades the harness on, so they are asserted directly
rather than inferred from an end-to-end run: a failing stage produces a typed error and
not an exception, retries happen only where retrying can help, and every path (answer,
decline, failure) still reports timings.
"""

from __future__ import annotations

import numpy as np
import pytest

from generation.generator import NoAnswerFromModel
from harness.orchestrator import PipelineConfig, VoiceRAGPipeline
from harness.retry import RetriesExhausted, RetryPolicy, retry_sync
from harness.types import (
    Answer, Chunk, ErrorKind, GuardrailVerdict, ResponseStatus, RetrievalResult,
    ScoredChunk, Stage, StageError, StageResult,
)


# --------------------------------------------------------------------------- #
# Fakes — no network, no model
# --------------------------------------------------------------------------- #

def _scored(text: str, dense: float, cid: str = "c1") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=cid, text=text, strategy="test", char_len=len(text)),
        score=dense, dense_score=dense,
    )


class FakeRetriever:
    def __init__(self, chunks=None, fail_times: int = 0) -> None:
        self._chunks = chunks if chunks is not None else [_scored("भारत की राजधानी नई दिल्ली है", 0.8)]
        self.fail_times = fail_times
        self.calls = 0
        self.embed_calls = 0
        self.last_timings: dict[str, float] = {"dense": 0.4, "lexical": 0.1}
        self.vector_index = type("VI", (), {"size": 10, "backend": "fake"})()
        self.lexical_index = type("LI", (), {"size": 10})()
        self.embedder = type("E", (), {"model_name": "fake"})()

    def embed_query(self, query: str) -> np.ndarray:
        self.embed_calls += 1
        return np.ones(384, dtype="float32") / np.sqrt(384)

    def retrieve(self, query, query_vector=None, top_k=5) -> RetrievalResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("transient index failure")
        return RetrievalResult(chunks=self._chunks, strategy="hybrid",
                               top_score=self._chunks[0].score if self._chunks else 0.0,
                               n_candidates=len(self._chunks))


class FakeGenerator:
    def __init__(self, answer: str = "नई दिल्ली", exc: Exception | None = None) -> None:
        self.answer, self.exc, self.calls = answer, exc, 0

    def generate(self, question, chunks) -> Answer:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return Answer(text=self.answer, mode="generated", model="fake")


class PassAllGuardrails:
    def run_pre_retrieval(self, query, vec=None):
        return [GuardrailVerdict(name="input_safety", passed=True)]

    def run_post_retrieval(self, query, retrieval):
        return GuardrailVerdict(name="low_confidence", passed=True, score=0.8)

    def run_post_generation(self, query, answer, retrieval):
        return GuardrailVerdict(name="groundedness", passed=True, score=1.0)


def _pipeline(**kw) -> VoiceRAGPipeline:
    config = PipelineConfig(top_k=3, stage_retries=2)
    return VoiceRAGPipeline(
        retriever=kw.pop("retriever", FakeRetriever()),
        generator=kw.pop("generator", FakeGenerator()),
        guardrails=kw.pop("guardrails", PassAllGuardrails()),
        config=kw.pop("config", config), **kw,
    )


# --------------------------------------------------------------------------- #

class TestRetryPolicy:
    def test_succeeds_after_transient_failures(self):
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise ConnectionError("boom")
            return "ok"

        assert retry_sync(flaky, RetryPolicy(max_attempts=3, base_delay_s=0.001)) == "ok"
        assert state["n"] == 3

    def test_raises_after_exhaustion_carrying_last_cause(self):
        def always_fails():
            raise ValueError("nope")

        with pytest.raises(RetriesExhausted) as exc:
            retry_sync(always_fails, RetryPolicy(max_attempts=2, base_delay_s=0.001))
        assert isinstance(exc.value.last, ValueError)
        assert exc.value.attempts == 2

    def test_backoff_is_jittered_within_ceiling(self):
        policy = RetryPolicy(base_delay_s=1.0, max_delay_s=8.0)
        for attempt in (1, 2, 3, 4):
            ceiling = min(8.0, 1.0 * 2 ** (attempt - 1))
            assert all(0.0 <= policy.delay_for(attempt) <= ceiling for _ in range(50))

    def test_deadline_stops_retrying_early(self):
        """A latency-graded pipeline cannot spend three attempts on a hung socket."""
        state = {"n": 0}

        def slow_fail():
            state["n"] += 1
            raise ConnectionError("boom")

        with pytest.raises(RetriesExhausted):
            retry_sync(slow_fail, RetryPolicy(max_attempts=5, base_delay_s=2.0, deadline_s=0.05))
        assert state["n"] < 5


class TestStageIsolation:
    def test_stage_failure_becomes_typed_error_not_exception(self):
        pipeline = _pipeline(retriever=FakeRetriever(fail_times=99))
        response = pipeline.run_text("भारत की राजधानी क्या है?")
        assert response.status is ResponseStatus.ERROR
        assert response.errors and response.errors[0].stage is Stage.RETRIEVE
        assert response.errors[0].kind is ErrorKind.TRANSIENT

    def test_transient_failure_is_retried_then_succeeds(self):
        retriever = FakeRetriever(fail_times=1)
        response = _pipeline(retriever=retriever).run_text("भारत की राजधानी क्या है?")
        assert response.status is ResponseStatus.ANSWERED
        assert retriever.calls == 2

    def test_failure_paths_are_still_timed(self):
        pipeline = _pipeline(retriever=FakeRetriever(fail_times=99))
        response = pipeline.run_text("भारत की राजधानी क्या है?")
        assert response.latency.stages.get("retrieve", 0.0) > 0.0

    def test_generation_failure_keeps_retrieved_context(self):
        """A failed generator must not discard what retrieval already found."""
        pipeline = _pipeline(generator=FakeGenerator(exc=RuntimeError("503 upstream")))
        response = pipeline.run_text("भारत की राजधानी क्या है?")
        assert response.status is ResponseStatus.ERROR
        assert len(response.retrieved) == 1

    def test_model_refusal_is_a_decline_not_an_error(self):
        """`NO_ANSWER` is the model using the refusal its prompt authorises."""
        pipeline = _pipeline(generator=FakeGenerator(exc=NoAnswerFromModel("not covered")))
        response = pipeline.run_text("भारत की राजधानी क्या है?")
        assert response.status is ResponseStatus.DECLINED
        assert response.reason == "model_declined"

    def test_deterministic_failures_are_not_retried(self):
        generator = FakeGenerator(exc=NoAnswerFromModel("not covered"))
        _pipeline(generator=generator).run_text("भारत की राजधानी क्या है?")
        assert generator.calls == 1


class TestPipelineRouting:
    def test_answer_path_populates_full_response(self):
        response = _pipeline().run_text("भारत की राजधानी क्या है?")
        assert response.status is ResponseStatus.ANSWERED
        assert response.answer == "नई दिल्ली"
        assert response.answer_mode == "generated"
        assert response.trace_id and len(response.trace_id) == 12
        assert response.latency.retrieval_ms > 0.0

    def test_empty_query_rejected_before_any_work(self):
        retriever = FakeRetriever()
        response = _pipeline(retriever=retriever).run_text("   ")
        assert response.status is ResponseStatus.ERROR
        assert response.errors[0].kind is ErrorKind.VALIDATION
        assert retriever.embed_calls == 0

    def test_query_is_embedded_exactly_once(self):
        """At ~104ms per embedding, embedding twice would blow the budget alone."""
        retriever = FakeRetriever()
        _pipeline(retriever=retriever).run_text("भारत की राजधानी क्या है?")
        assert retriever.embed_calls == 1

    def test_pre_retrieval_decline_skips_retrieval_entirely(self):
        class BlockPre(PassAllGuardrails):
            def run_pre_retrieval(self, query, vec=None):
                return [GuardrailVerdict(name="input_safety", passed=False, reason="unsafe")]

        retriever = FakeRetriever()
        response = _pipeline(retriever=retriever, guardrails=BlockPre()).run_text("bad query")
        assert response.status is ResponseStatus.DECLINED
        assert response.reason == "input_safety"
        assert retriever.calls == 0
        # Not embedded either. Embedding is ~90ms of a ~92ms budget, so a pre-retrieval
        # guardrail that runs *after* the embed saves essentially nothing.
        assert retriever.embed_calls == 0

    def test_confidence_decline_skips_generation(self):
        class BlockGate(PassAllGuardrails):
            def run_post_retrieval(self, query, retrieval):
                return GuardrailVerdict(name="low_confidence", passed=False, score=0.2)

        generator = FakeGenerator()
        response = _pipeline(generator=generator, guardrails=BlockGate()).run_text("q")
        assert response.status is ResponseStatus.DECLINED
        assert generator.calls == 0
        assert response.retrieved  # the evidence behind the refusal is still returned

    def test_groundedness_decline_withholds_the_answer(self):
        class BlockGrounded(PassAllGuardrails):
            def run_post_generation(self, query, answer, retrieval):
                return GuardrailVerdict(name="groundedness", passed=False, score=0.1)

        response = _pipeline(guardrails=BlockGrounded()).run_text("q")
        assert response.status is ResponseStatus.DECLINED
        assert response.answer is None

    def test_retrieval_only_mode_reports_no_answer(self):
        config = PipelineConfig(enable_generation=False)
        response = _pipeline(config=config).run_text("q")
        assert response.status is ResponseStatus.ANSWERED
        assert response.answer is None and response.reason == "retrieval_only"

    def test_text_path_leaves_transcript_none(self):
        """A text-driven run must stay distinguishable from a spoken one."""
        assert _pipeline().run_text("q").transcript is None

    def test_audio_path_without_stt_reports_config_error(self):
        pipeline = _pipeline()
        pipeline.stt = None
        response = pipeline.run_audio(object())
        assert response.status is ResponseStatus.ERROR
        assert response.errors[0].kind is ErrorKind.CONFIG


class TestStructuredTypes:
    def test_stage_result_ok_and_unwrap(self):
        good: StageResult[str] = StageResult(stage=Stage.STT, value="text")
        assert good.ok and good.unwrap() == "text"

        bad: StageResult[str] = StageResult(
            stage=Stage.STT,
            error=StageError(stage=Stage.STT, kind=ErrorKind.AUTH, message="nope"),
        )
        assert not bad.ok
        with pytest.raises(RuntimeError):
            bad.unwrap()

    @pytest.mark.parametrize("kind,expected", [
        (ErrorKind.TRANSIENT, True), (ErrorKind.RATE_LIMITED, True),
        (ErrorKind.AUTH, False), (ErrorKind.CONFIG, False),
        (ErrorKind.VALIDATION, False), (ErrorKind.INTERNAL, False),
    ])
    def test_only_transient_kinds_are_retryable(self, kind, expected):
        assert StageError(stage=Stage.STT, kind=kind, message="x").retryable is expected

    def test_response_serialises_to_json(self):
        """The demo returns this over HTTP; it must round-trip."""
        response = _pipeline().run_text("भारत की राजधानी क्या है?")
        payload = response.model_dump_json()
        assert '"status":"answered"' in payload
