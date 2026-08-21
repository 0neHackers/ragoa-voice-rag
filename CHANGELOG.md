# Changelog

All notable changes to this project. Append-only — entries are never deleted or rewritten.
Versioning scheme: `V{MAJOR}.{MINOR}` per the team's master build prompt.
MAJOR = any behavior/function/UI/bugfix/security change. MINOR = non-behavioral (docs, formatting, comments).

## V0.0 — 2026-08-21
**Type:** Scaffold
**Summary:** Initial project structure and versioning system created.
**Files changed:**
- `master_repo/VERSION` — created, set to `0.0`
- `master_repo/CHANGELOG.md` — created with this entry
- `master_repo/V0.0/VERSION` — created, set to `0.0`
- `master_repo/V0.0/` — full directory scaffold (data, stt, chunking, retrieval, generation, guardrails, harness, benchmarks, demo, tests)
**Details:** `master_repo`, the `VERSION` pointer, `CHANGELOG.md`, and the V{MAJOR}.{MINOR}
directory-per-version system initialized per master prompt section 1/2. Git repository
initialized at `master_repo/`. No functional code yet — this is the starting point, not a bump.
**Supersedes:** —

## V1.0 — 2026-08-21
**Type:** Major
**Summary:** Phase 1 — real speech-to-text via Sarvam's streaming WebSocket endpoint, plus the
structured-I/O and retry primitives every later stage is built on.
**Files changed:**
- `harness/types.py` — new. `Stage`, `ErrorKind`, `StageError`, `StageResult[T]`, `Transcript`,
  `Chunk`, `ScoredChunk`, `RetrievalResult`, `GuardrailVerdict`, `Answer`, `LatencyBreakdown`,
  `PipelineResponse`, `Timer`. Establishes the rule that no stage raises across a boundary —
  it returns a typed value or a typed error.
- `harness/retry.py` — new. `RetryPolicy` (full-jitter exponential backoff + total deadline),
  `retry_sync`, `retry_async`, `RetriesExhausted`.
- `stt/sarvam_client.py` — new. `SarvamSTT.transcribe()` over the streaming WebSocket endpoint
  (`wss://api.sarvam.ai/speech-to-text/ws`, `saarika:v2.5`), with `Audio` (16kHz mono PCM),
  `load_audio_file()` (downmix + linear resample), `record_microphone()` (rejects near-silence
  rather than transcribing an empty demo), and `_extract_transcript()` frame parsing.
- `stt/transcribe.py` — new. Standalone CLI: `--mic` / `--file` / `--list-devices` / `--json`.
  Errors out if given neither audio source; there is no typed-text input.
- `tests/test_stt.py` — new. 15 tests covering the STT contract, frame-shape tolerance, and
  retry/backoff behaviour. All passing.
- `DECISIONS.md` — D3 revised (see Details).
- `.env.example`, `.env` — `EMBED_MODEL` default updated to match the D3 revision.
**Details:** Streaming is the primary transport per D1. The batch REST endpoint is retained as
an explicitly *labelled* fallback (`Transcript.transport == "batch"`) because a `wss://`
handshake is the most environment-fragile call in the pipeline — proxies and some PaaS egress
rules block WebSockets while allowing HTTPS. An auth failure deliberately skips the fallback,
since the same key will fail identically over REST and retrying only burns latency budget.
`Transcript.is_real_audio` is set only by paths that pushed audio bytes to a recogniser, making
"voice-enabled" an auditable property of a response rather than a README claim.
D3 revised: `fastembed` 0.8's ONNX registry does not carry `intfloat/multilingual-e5-small`;
switched to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (also 384-dim, also
multilingual, 0.22GB). The only other multilingual option, `multilingual-e5-large`, is
disqualified on both latency and container size.
Build tooling added at repo root: `bump.sh` (mechanises master-prompt section 1.3 steps 1–4,
6, 7) and a shared `master_repo/.venv` — the virtualenv is a reproducible build artifact and is
excluded from version snapshots rather than copied into all of them.
**Supersedes:** V0.0
