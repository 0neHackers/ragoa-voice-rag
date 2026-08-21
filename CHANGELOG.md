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

## V2.0 — 2026-08-21
**Type:** Major
**Summary:** Phase 2 — MSMARCO-XI loading against the dataset's real schema, plus all four
required chunking strategies as independently testable modules.
**Files changed:**
- `data/loader.py` — new. `Example`/`Passage` dataclasses, `load_examples()`, `_coerce_passages()`,
  `corpus_stats()`, `_read_parquet()`, JSONL cache with atomic writes.
- `data/build_corpus.py` — new. CLI that loads the corpus and profiles every strategy over it
  (chunk counts, chars p50/p90/max, build seconds).
- `chunking/base.py` — new. `ChunkingStrategy` ABC, `split_sentences()` (Devanagari danda aware),
  `split_paragraphs()`, script-aware `approx_tokens()`.
- `chunking/fixed_size.py` — new. Strategy 1: character-window chunking with overlap.
- `chunking/semantic.py` — new. Strategy 2: embedding-similarity breakpoints with a
  percentile threshold; `chunk_examples()` overridden to embed the whole corpus in one pass.
- `chunking/recursive.py` — new. Strategy 3: paragraph → sentence → clause → hard-cut descent,
  with greedy repacking of runt pieces.
- `chunking/metadata_aware.py` — new. Strategy 4: passage-as-chunk, metadata propagated.
- `chunking/registry.py` — new. `STRATEGIES`, `get_strategy()`, `DEFAULT_ENSEMBLE`.
- `retrieval/embedder.py` — new. `Embedder` (fastembed/ONNX, L2-normalised output),
  `get_embedder()` process-wide cache, `warmup()`.
- `tests/test_chunking.py` — new. 35 tests. Total suite now 50, all passing.
- `DECISIONS.md` — D8 added (embedding latency floor).
**Details:** Three schema facts were established by inspection rather than assumption, and each
changed the code. (1) `load_dataset("ai4bharat/MSMARCO-XI", "hi")` fails — the repo has no named
configs, only per-language parquet files, so the loader addresses `validation/hinval.parquet`
directly. (2) The `passages` column is a struct of three parallel lists —
`Translated_passages`, `English_passages`, `is_selected` (int 0/1) — not the `passage_text`
shape plain MS MARCO uses; both are now handled. (3) The Hindi train parquet is 3.7GB in a
single row group against 462MB for validation, with identical schema, so validation is used
(D2 amended in effect). Loaded corpus: 2000 examples / 19,987 passages / mean 330 chars.
The JSONL cache is keyed on (lang, split) only, *not* on limit — an earlier per-limit key meant
every corpus-size change re-read the parquet at ~5 minutes a time. It now extracts a 5000-example
slice once and serves smaller limits by prefix, taking reload from 335s to 0.36s.
Measured chunk behaviour over 400 examples / 3995 passages: fixed_size 4138 chunks (p50 296
chars), recursive 4070 (p50 298), metadata_aware 4029 (p50 297); semantic over 100 examples
produces 1.49 chunks per passage against ~1.02 for the others, confirming it is the only
strategy that meaningfully re-segments this corpus.
`is_selected` is carried into chunk metadata but deliberately excluded from ranking — it is
ground-truth answer labelling, and boosting on it would leak the answer key into retrieval and
inflate every benchmark number this project reports.
**Supersedes:** V1.0
