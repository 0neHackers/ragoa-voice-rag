"""The orchestrator — the harness the task asks for.

This is the difference between "a function that calls an LLM" and a harness: every stage
runs behind a uniform wrapper that times it, retries it when retrying makes sense,
converts any escaping exception into a typed `StageError`, and hands the orchestrator a
`StageResult` to make a routing decision from. A failed STT call produces a structured
error response with timings intact; it does not unwind the request.

Pipeline order, and why each guardrail sits where it does:

    audio ──▶ STT ──▶ [safety] ──▶ [off-topic] ──▶ retrieve ──▶ [confidence]
                                                                    │
                              answer ◀── [groundedness] ◀── generate ┘

Safety and off-topic run *before* retrieval so a bad query costs nothing downstream.
The confidence gate runs after retrieval but before generation — it is the one that
prevents the classic RAG failure of generating fluently from the least-bad chunk of a
corpus that simply does not contain the answer. Groundedness runs last because it is the
only check that can catch a model which had good context and drifted anyway.

The query vector is computed **once** and threaded through the off-topic guardrail and
retrieval. At ~100ms per embedding (DECISIONS.md D8) embedding twice would put the
pipeline over budget on its own.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

from generation.generator import NoAnswerFromModel
from harness.retry import RetriesExhausted, RetryPolicy, retry_sync
from harness.types import (
    Answer, ErrorKind, GuardrailVerdict, LatencyBreakdown, PipelineResponse,
    ResponseStatus, RetrievalResult, Stage, StageError, StageResult, Timer, Transcript,
)

T = TypeVar("T")


@dataclass(slots=True)
class PipelineConfig:
    top_k: int = 5
    enable_generation: bool = True
    enable_guardrails: bool = True
    #: Wall-clock ceiling for the retrieval half. Exceeding it is reported, not fatal —
    #: a slow answer is more useful than a refused one, and hiding the overrun would
    #: defeat the point of measuring.
    retrieval_budget_ms: float = 200.0
    stage_retries: int = 2

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            enable_generation=os.getenv("ENABLE_GENERATION", "1") not in ("0", "false"),
            enable_guardrails=os.getenv("ENABLE_GUARDRAILS", "1") not in ("0", "false"),
            retrieval_budget_ms=float(os.getenv("RETRIEVAL_BUDGET_MS", "200")),
        )


@dataclass
class VoiceRAGPipeline:
    """Wires the stages together. Every public method returns a `PipelineResponse`."""

    retriever: Any                      # retrieval.retriever.Retriever
    stt: Any = None                     # stt.sarvam_client.SarvamSTT
    generator: Any = None               # generation.generator.Generator
    guardrails: Any = None              # guardrails.suite.GuardrailSuite
    config: PipelineConfig = field(default_factory=PipelineConfig.from_env)

    # ------------------------------------------------------------------ #
    # Stage wrapper
    # ------------------------------------------------------------------ #

    def _run_stage(
        self,
        stage: Stage,
        fn: Callable[[], T],
        *,
        retries: int | None = None,
        deadline_s: float | None = None,
        no_retry: tuple[type[BaseException], ...] = (),
    ) -> StageResult[T]:
        """Run one stage: timed, optionally retried, never allowed to raise.

        `StageResult` returned by a stage is passed through as-is, so a stage that has
        already classified its own failure (the STT client does) keeps that
        classification instead of having it flattened to INTERNAL here.

        `no_retry` names outcomes that are deterministic, so retrying only spends the
        latency budget to reach the same answer — a model refusal, a rejected API key, a
        malformed request.
        """
        attempts = max(1, retries if retries is not None else self.config.stage_retries)
        retry_on: tuple[type[BaseException], ...] = (Exception,)
        if no_retry:
            excluded = no_retry

            class _Retryable(type):
                def __instancecheck__(cls, instance: object) -> bool:
                    return isinstance(instance, Exception) and not isinstance(instance, excluded)

            class RetryableError(Exception, metaclass=_Retryable):
                """Virtual base matching any exception except the excluded ones."""

            retry_on = (RetryableError,)

        policy = RetryPolicy(max_attempts=attempts, base_delay_s=0.1,
                             max_delay_s=1.0, deadline_s=deadline_s, retry_on=retry_on)

        started = time.perf_counter()
        try:
            value = retry_sync(fn, policy) if attempts > 1 else fn()
        except RetriesExhausted as exc:
            # Every failure path is timed too. A stage that took 4 seconds to fail is
            # exactly the thing a latency-graded pipeline needs to be able to see.
            return _timed(
                StageResult(
                    stage=stage,
                    error=StageError(
                        stage=stage, kind=_classify(exc.last), message=str(exc.last),
                        attempts=exc.attempts, detail={"exception": type(exc.last).__name__},
                    ),
                ),
                started,
            )
        except Exception as exc:  # noqa: BLE001 - the point of the harness
            return _timed(
                StageResult(
                    stage=stage,
                    error=StageError(stage=stage, kind=_classify(exc), message=str(exc),
                                     detail={"exception": type(exc).__name__}),
                ),
                started,
            )

        if isinstance(value, StageResult):
            # A stage that timed itself keeps its own number; it excludes our wrapper.
            if not value.elapsed_ms:
                _timed(value, started)
            return value

        return _timed(StageResult(stage=stage, value=value), started)

    # ------------------------------------------------------------------ #
    # Entrypoints
    # ------------------------------------------------------------------ #

    def run_audio(self, audio: Any, **kw: Any) -> PipelineResponse:
        """Full voice path: audio bytes → transcript → answer."""
        trace_id = uuid.uuid4().hex[:12]
        latency = LatencyBreakdown()

        if self.stt is None:
            return PipelineResponse.failed(
                StageError(stage=Stage.STT, kind=ErrorKind.CONFIG,
                           message="No STT client configured on this pipeline."),
                trace_id=trace_id, latency=latency,
            )

        stt_result = self._run_stage(Stage.STT, lambda: self.stt.transcribe(audio), retries=1)
        latency.record(Stage.STT, stt_result.elapsed_ms)
        latency.stt_ms = stt_result.elapsed_ms

        if not stt_result.ok:
            latency.full_ms = stt_result.elapsed_ms
            return PipelineResponse.failed(stt_result.error, trace_id=trace_id, latency=latency)

        transcript: Transcript = stt_result.value
        response = self.run_text(transcript.text, _latency=latency, _trace_id=trace_id, **kw)
        response.transcript = transcript
        return response

    def run_text(
        self,
        query: str,
        *,
        _latency: LatencyBreakdown | None = None,
        _trace_id: str | None = None,
        top_k: int | None = None,
    ) -> PipelineResponse:
        """Post-STT path. Public so the benchmark can measure retrieval without audio.

        This is **not** a voice bypass for the demo: `PipelineResponse.transcript` stays
        `None` here, and the demo surfaces that, so a text-driven run is always
        distinguishable from a spoken one.
        """
        latency = _latency or LatencyBreakdown()
        trace_id = _trace_id or uuid.uuid4().hex[:12]
        verdicts: list[GuardrailVerdict] = []
        errors: list[StageError] = []

        query = (query or "").strip()
        if not query:
            return PipelineResponse.failed(
                StageError(stage=Stage.RETRIEVE, kind=ErrorKind.VALIDATION,
                           message="Empty query — nothing to retrieve."),
                trace_id=trace_id, latency=latency,
            )

        with Timer() as retrieval_timer:
            # ---- pre-retrieval guardrails, BEFORE embedding ---------------
            # Ordering is load-bearing. Embedding is ~90ms of a ~92ms retrieval budget;
            # everything else together is under 2ms. Running these checks first means a
            # refused query costs ~0.05ms instead of ~90ms — which is the entire point of
            # a pre-retrieval guardrail, and was not true while an earlier version needed
            # the query vector to score against the corpus centroid. Neither surviving
            # check needs the vector, so none is computed for a query we will refuse.
            if self.config.enable_guardrails and self.guardrails is not None:
                pre = self.guardrails.run_pre_retrieval(query)
                verdicts.extend(pre)
                for verdict in pre:
                    latency.record(verdict.name, verdict.elapsed_ms)
                blocked = next((v for v in pre if not v.passed), None)
                if blocked is not None:
                    latency.retrieval_ms = _elapsed(retrieval_timer)
                    latency.full_ms = latency.retrieval_ms + latency.stt_ms
                    return PipelineResponse.declined(
                        reason=blocked.name,
                        query=query, guardrails=verdicts, latency=latency, trace_id=trace_id,
                    )

            # ---- embed once, reuse everywhere -----------------------------
            embed_result = self._run_stage(
                Stage.EMBED, lambda: self.retriever.embed_query(query), retries=2
            )
            latency.record(Stage.EMBED, embed_result.elapsed_ms)
            if not embed_result.ok:
                latency.retrieval_ms = _elapsed(retrieval_timer)
                return PipelineResponse.failed(embed_result.error, query=query,
                                               guardrails=verdicts,
                                               trace_id=trace_id, latency=latency)
            query_vector: np.ndarray = embed_result.value

            # ---- retrieval -------------------------------------------------
            retrieve_result = self._run_stage(
                Stage.RETRIEVE,
                lambda: self.retriever.retrieve(query, query_vector=query_vector,
                                                top_k=top_k or self.config.top_k),
                retries=2,
            )
            latency.record(Stage.RETRIEVE, retrieve_result.elapsed_ms)
            for name, ms in getattr(self.retriever, "last_timings", {}).items():
                latency.record(f"retrieve.{name}", ms)

            if not retrieve_result.ok:
                latency.retrieval_ms = _elapsed(retrieval_timer)
                return PipelineResponse.failed(retrieve_result.error, query=query,
                                               guardrails=verdicts, trace_id=trace_id,
                                               latency=latency)
            retrieval: RetrievalResult = retrieve_result.value

            # ---- post-retrieval guardrail (confidence gate) ---------------
            if self.config.enable_guardrails and self.guardrails is not None:
                gate = self.guardrails.run_post_retrieval(query, retrieval)
                verdicts.append(gate)
                latency.record(gate.name, gate.elapsed_ms)
                if not gate.passed:
                    latency.retrieval_ms = _elapsed(retrieval_timer)
                    latency.full_ms = latency.retrieval_ms + latency.stt_ms
                    return PipelineResponse.declined(
                        reason=gate.name, query=query, retrieved=retrieval.chunks,
                        guardrails=verdicts, latency=latency, trace_id=trace_id,
                    )

        latency.retrieval_ms = retrieval_timer.ms

        # ---- generation (outside the retrieval budget, by D5) -------------
        if not self.config.enable_generation or self.generator is None:
            latency.full_ms = latency.retrieval_ms + latency.stt_ms
            return PipelineResponse(
                status=ResponseStatus.ANSWERED,
                answer=None, query=query, retrieved=retrieval.chunks,
                guardrails=verdicts, latency=latency, trace_id=trace_id,
                reason="retrieval_only",
            )

        gen_result = self._run_stage(
            Stage.GENERATE,
            lambda: self.generator.generate(query, retrieval.chunks),
            retries=2, deadline_s=30.0,
            no_retry=(NoAnswerFromModel, PermissionError, ValueError),
        )
        latency.record(Stage.GENERATE, gen_result.elapsed_ms)
        latency.generation_ms = gen_result.elapsed_ms

        # A model that answered `NO_ANSWER` used the refusal its prompt authorises. That
        # is a decline, not a failure, and it must reach the user as the same typed
        # response a guardrail decline produces — the user should not be able to tell
        # which component noticed the corpus doesn't cover their question.
        if gen_result.error is not None and _is_model_refusal(gen_result.error):
            latency.full_ms = latency.retrieval_ms + latency.generation_ms + latency.stt_ms
            verdicts.append(GuardrailVerdict(
                name="model_declined", passed=False,
                reason=gen_result.error.message, elapsed_ms=gen_result.elapsed_ms,
            ))
            return PipelineResponse.declined(
                reason="model_declined", query=query, retrieved=retrieval.chunks,
                guardrails=verdicts, latency=latency, trace_id=trace_id,
            )

        if not gen_result.ok:
            latency.full_ms = latency.retrieval_ms + latency.generation_ms + latency.stt_ms
            errors.append(gen_result.error)
            return PipelineResponse(
                status=ResponseStatus.ERROR, reason=str(gen_result.error), query=query,
                retrieved=retrieval.chunks, guardrails=verdicts, errors=errors,
                latency=latency, trace_id=trace_id,
            )

        answer: Answer = gen_result.value

        # ---- post-generation guardrail (groundedness) ---------------------
        if self.config.enable_guardrails and self.guardrails is not None:
            grounded = self.guardrails.run_post_generation(query, answer, retrieval)
            verdicts.append(grounded)
            latency.record(grounded.name, grounded.elapsed_ms)
            if not grounded.passed:
                latency.full_ms = (latency.retrieval_ms + latency.generation_ms
                                   + grounded.elapsed_ms + latency.stt_ms)
                return PipelineResponse.declined(
                    reason=grounded.name, query=query, retrieved=retrieval.chunks,
                    guardrails=verdicts, latency=latency, trace_id=trace_id,
                    answer_mode=answer.mode,
                )

        latency.full_ms = (latency.retrieval_ms + latency.generation_ms + latency.stt_ms
                           + sum(v.elapsed_ms for v in verdicts if v.name == "groundedness"))

        return PipelineResponse(
            status=ResponseStatus.ANSWERED,
            answer=answer.text, query=query, retrieved=retrieval.chunks,
            guardrails=verdicts, latency=latency, errors=errors,
            answer_mode=answer.mode, trace_id=trace_id,
        )

    # ------------------------------------------------------------------ #

    def health(self) -> dict[str, Any]:
        """What the demo's /health endpoint reports. Cheap; no model calls."""
        return {
            "index_size": getattr(self.retriever.vector_index, "size", 0),
            "index_backend": getattr(self.retriever.vector_index, "backend", "unknown"),
            "lexical_size": getattr(self.retriever.lexical_index, "size", 0),
            "embed_model": getattr(self.retriever.embedder, "model_name", "unknown"),
            "stt_configured": self.stt is not None and bool(
                getattr(getattr(self.stt, "config", None), "api_key", "")
            ),
            "generation_enabled": self.config.enable_generation and self.generator is not None,
            "guardrails_enabled": self.config.enable_guardrails and self.guardrails is not None,
            "retrieval_budget_ms": self.config.retrieval_budget_ms,
        }


def _elapsed(timer: Timer) -> float:
    """Read a Timer mid-block, where `__exit__` has not run yet."""
    import time

    return (time.perf_counter() - timer._t0) * 1000.0  # noqa: SLF001


def _timed(result: StageResult[T], started: float) -> StageResult[T]:
    result.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result


def _is_model_refusal(error: StageError) -> bool:
    return error.detail.get("exception") == "NoAnswerFromModel"


def _classify(exc: BaseException) -> ErrorKind:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(s in text for s in ("401", "403", "unauthorized", "forbidden", "invalid api")):
        return ErrorKind.AUTH
    if "429" in text or "rate limit" in text:
        return ErrorKind.RATE_LIMITED
    if any(s in text for s in ("timeout", "timed out", "connection", "refused",
                               "unreachable", "reset", "502", "503")):
        return ErrorKind.TRANSIENT
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return ErrorKind.VALIDATION
    return ErrorKind.INTERNAL


__all__ = ["VoiceRAGPipeline", "PipelineConfig"]
