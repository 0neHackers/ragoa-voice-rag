"""Structured input/output types shared by every pipeline stage.

The task requires the pipeline to run inside a real harness rather than a raw
prompt-in/text-out call. Everything in this module exists to make that concrete: no
stage returns a bare string, and no stage raises across a stage boundary. A stage
either produces its typed payload or produces a typed `StageError`, and the
orchestrator decides what that means for the request as a whole.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Stage identity + failure taxonomy
# --------------------------------------------------------------------------- #

class Stage(str, Enum):
    """Every point in the pipeline that can succeed, fail, or decline on its own."""

    STT = "stt"
    INPUT_SAFETY = "input_safety"
    OFF_TOPIC = "off_topic"
    EMBED = "embed"
    RETRIEVE = "retrieve"
    CONFIDENCE_GATE = "confidence_gate"
    GENERATE = "generate"
    GROUNDEDNESS = "groundedness"


class ErrorKind(str, Enum):
    """Why a stage failed, in terms the orchestrator can act on.

    The distinction that matters is `TRANSIENT` vs everything else: only transient
    failures are worth retrying. Retrying a `CONFIG` error (missing API key) or a
    `VALIDATION` error (empty audio) just burns latency budget to fail again.
    """

    TRANSIENT = "transient"        # network blip, 5xx, timeout — retry with backoff
    RATE_LIMITED = "rate_limited"  # 429 — retry, but back off harder
    AUTH = "auth"                  # 401/403 — bad or missing credentials, do not retry
    CONFIG = "config"              # missing key/model/file — do not retry
    VALIDATION = "validation"      # caller handed us something unusable — do not retry
    UPSTREAM = "upstream"          # provider returned a well-formed refusal or 4xx
    INTERNAL = "internal"          # our bug — surfaced, never swallowed


class StageError(BaseModel):
    """A failure that crossed a stage boundary without becoming an exception."""

    stage: Stage
    kind: ErrorKind
    message: str
    attempts: int = 1
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def retryable(self) -> bool:
        return self.kind in (ErrorKind.TRANSIENT, ErrorKind.RATE_LIMITED)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.stage.value}/{self.kind.value}] {self.message}"


T = TypeVar("T")


class StageResult(BaseModel, Generic[T]):
    """Either a typed value or a typed error, plus how long the stage took.

    Deliberately not an exception-based API. A failed STT call must not be able to
    unwind the whole request — the orchestrator needs to see the failure as data so
    it can still return a structured response and still report timings.
    """

    stage: Stage
    value: T | None = None
    error: StageError | None = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None

    def unwrap(self) -> T:
        if not self.ok:
            raise RuntimeError(f"unwrap() on failed stage: {self.error}")
        return self.value  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Per-stage payloads
# --------------------------------------------------------------------------- #

class Transcript(BaseModel):
    """Output of the STT stage. `is_real_audio` is not decorative.

    The task explicitly requires voice input rather than typed text. This flag is set
    only by code paths that actually pushed audio bytes to a speech recogniser, and
    the demo surfaces it, so "voice-enabled" is an auditable property of a response
    rather than a claim in a README.
    """

    text: str
    language_code: str
    provider: str
    model: str
    is_real_audio: bool
    audio_duration_s: float | None = None
    transport: Literal["streaming", "batch"] = "streaming"
    partials: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """One indexed unit of the corpus, with the metadata retrieval can filter on."""

    chunk_id: str
    text: str
    strategy: str
    query_id: str | None = None
    passage_idx: int | None = None
    source_lang: str | None = None
    char_len: int = 0
    is_selected_passage: bool | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "strategy": self.strategy,
            "query_id": self.query_id,
            "passage_idx": self.passage_idx,
            "source_lang": self.source_lang,
            "is_selected_passage": self.is_selected_passage,
        }


class ScoredChunk(BaseModel):
    """A retrieved chunk with the score breakdown that produced its rank.

    Dense and lexical scores are kept separately rather than collapsed into one
    number so the confidence gate can threshold on cosine similarity specifically —
    a fused RRF score has no absolute meaning and cannot be thresholded honestly.
    """

    chunk: Chunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    rank: int = 0


class RetrievalResult(BaseModel):
    chunks: list[ScoredChunk]
    strategy: str
    top_score: float = 0.0
    mean_score: float = 0.0
    n_candidates: int = 0


class GuardrailVerdict(BaseModel):
    """One guardrail's decision. `passed=False` means the pipeline must stop here."""

    name: str
    passed: bool
    score: float | None = None
    threshold: float | None = None
    reason: str | None = None
    elapsed_ms: float = 0.0


class Answer(BaseModel):
    text: str
    mode: Literal["generated", "extractive"]
    model: str
    cited_chunk_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# The pipeline's single public response type
# --------------------------------------------------------------------------- #

class ResponseStatus(str, Enum):
    ANSWERED = "answered"
    DECLINED = "declined"   # a guardrail deliberately refused — a designed outcome
    ERROR = "error"         # a stage failed — surfaced, not hidden


class LatencyBreakdown(BaseModel):
    """Per-stage timings, plus the two headline numbers the task grades.

    `retrieval_ms` is the number held to the <200ms bar (see DECISIONS.md D5);
    `full_ms` includes generation and the post-generation groundedness check.
    """

    stages: dict[str, float] = Field(default_factory=dict)
    stt_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    full_ms: float = 0.0

    def record(self, stage: Stage | str, ms: float) -> None:
        key = stage.value if isinstance(stage, Stage) else stage
        self.stages[key] = round(ms, 3)


class PipelineResponse(BaseModel):
    """What the harness always returns — for answers, declines, and failures alike.

    A decline is a first-class response with a machine-readable `reason`, which is
    what makes "the system knows when not to answer" demonstrable rather than
    indistinguishable from a crash.
    """

    status: ResponseStatus
    answer: str | None = None
    reason: str | None = None
    query: str | None = None
    transcript: Transcript | None = None
    retrieved: list[ScoredChunk] = Field(default_factory=list)
    guardrails: list[GuardrailVerdict] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    errors: list[StageError] = Field(default_factory=list)
    answer_mode: str | None = None
    trace_id: str | None = None

    @classmethod
    def declined(cls, reason: str, **kw: Any) -> "PipelineResponse":
        return cls(status=ResponseStatus.DECLINED, reason=reason, **kw)

    @classmethod
    def failed(cls, error: StageError, **kw: Any) -> "PipelineResponse":
        kw.setdefault("errors", []).append(error)
        return cls(status=ResponseStatus.ERROR, reason=str(error), **kw)


# --------------------------------------------------------------------------- #
# Timing helper
# --------------------------------------------------------------------------- #

class Timer:
    """`with Timer() as t: ...` → `t.ms`. Uses perf_counter, not wall clock."""

    __slots__ = ("_t0", "ms")

    def __enter__(self) -> "Timer":
        self.ms = 0.0
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
        return False
