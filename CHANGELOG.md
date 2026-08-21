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

## V3.0 — 2026-08-21
**Type:** Major
**Summary:** Phase 3 — in-process hybrid retrieval: exact dense vector search fused with a
hand-rolled inverted-index BM25 via reciprocal rank fusion.
**Files changed:**
- `retrieval/vector_index.py` — new. `VectorIndex` over `faiss.IndexFlatIP` with `_NumpyFlatIP`,
  an exact fallback implementing identical math; `centroid()` for the off-topic guardrail;
  `save()`/`load()` persistence.
- `retrieval/lexical.py` — new. `LexicalIndex`: Okapi BM25 over a hand-built inverted index,
  Unicode-aware `tokenize()`, bilingual `STOPWORDS`.
- `retrieval/retriever.py` — new. `Retriever.retrieve()`, `_rrf()` fusion, `_materialise()`,
  `RetrievalConfig`, per-stage timings in `last_timings`.
- `retrieval/build_index.py` — new. Build CLI with `dedupe_chunks()` and build statistics.
- `retrieval/embedder.py` — `embed_texts()` gains build-time process parallelism;
  `embed_query()` pinned to `parallel=0`; `build_parallelism` capped at 8.
- `tests/test_retrieval.py` — new. 33 tests. Suite now 83, all passing.
- `.gitattributes` — new, normalises line endings.
**Details:** `IndexFlatIP` (exact) rather than an ANN index: at ~20k vectors an exact search is
a single matmul at 0.3–0.5ms, so ANN buys no measurable time while costing recall — and that
recall loss would move the score distribution the confidence guardrail thresholds against,
making the guardrail a function of the index's approximation error.
RRF rather than a weighted score blend, because cosine is bounded [-1,1] and BM25 is unbounded
and corpus-dependent, so any `alpha*dense + (1-alpha)*bm25` needs per-corpus renormalisation.
The tradeoff is that fused scores have no absolute meaning, so `ScoredChunk.dense_score` is
kept separately and the confidence gate thresholds on raw cosine only. Chunks surfaced by BM25
alone fall outside the dense candidate list and so have no dense score; `_materialise()` now
computes it with a single dot product rather than leaving `None`, which would have silently
excluded exactly those chunks from the check that decides whether we may answer.
**BM25 was rewritten rather than taken from `rank_bm25`.** `BM25Okapi.get_scores` evaluates
`[doc.get(term, 0) for doc in self.doc_freqs]` — a Python-level pass over every document per
query term — measured at 3.25ms/query on 3k chunks and scaling linearly, which would have cost
~20ms+ at full corpus size. The inverted index touches only documents containing a query term:
**0.12ms/query, a 27x improvement.** The IDF form also differs deliberately —
`log(1 + (N-df+0.5)/(df+0.5))` instead of the textbook `log((N-df+0.5)/(df+0.5))`, which is
zero for a term in one of two documents and *negative* for terms in more than half the corpus,
letting a common word push documents down the ranking.
Measured end-to-end retrieval on 3039 real Hindi chunks: **124–136ms total** — query embed
110–130ms, dense search 0.3–0.5ms, BM25 6–13ms (now 0.12ms), fusion 0.1ms. Correct passages
retrieved for the sampled queries. Comfortably inside the 200ms bar, with embedding as the
dominant term exactly as D8 predicted.
Build-time embedding now fans across worker processes: 294ms/chunk single-process vs 82ms at
parallel=8 (3.6x), taking a 15k-chunk build from ~74 minutes to ~20. Parallelism is capped at
8 because each worker loads its own ~250MB ONNX session — an unbounded `cpu_count - 2` spawned
14 workers and died with a bad allocation. The query path is pinned to `parallel=0`, where
process spawn would be pure overhead.
Fixed: a literal NUL byte in the BM25 empty-document placeholder made `lexical.py` unimportable
("source code string cannot contain null bytes").
**Supersedes:** V2.0

## V4.0 — 2026-08-21
**Type:** Major
**Summary:** Phases 4–6 — the orchestration harness, four guardrails, and grounded
generation. The pipeline now runs end to end from query to structured answer.
**Files changed:**
- `harness/orchestrator.py` — new. `VoiceRAGPipeline` with `run_audio()`/`run_text()`,
  the `_run_stage()` wrapper (timing, selective retry, exception→`StageError`), `_timed()`,
  `_is_model_refusal()`, `PipelineConfig`, `health()`.
- `harness/factory.py` — new. `build_pipeline()` — the single assembly point every
  entrypoint uses, with embedder warmup.
- `guardrails/base.py` — new. `Guardrail` ABC; timing, error containment, fail-open/closed.
- `guardrails/input_safety.py` — new. Harm + prompt-injection patterns, length cap, NFKC folding.
- `guardrails/language_match.py` — new. `LanguageMatchGuardrail`, `from_chunks()` script
  inference, `ROMANISED_HINDI_MARKERS`.
- `guardrails/confidence.py` — new. Absolute top-score threshold plus margin-over-tail test.
- `guardrails/groundedness.py` — new. Stopword-stripped content-token overlap, unsupported-number
  penalty, optional LLM entailment path.
- `guardrails/suite.py` — new. Groups guardrails by pipeline position.
- `guardrails/off_topic.py` — **deleted**, see Details.
- `generation/generator.py` — new. `Generator` (Claude Haiku 4.5), extractive fallback,
  `NoAnswerFromModel`, `_resolve_citations()`.
- `generation/prompts.py` — new. Grounding system prompt, numbered-passage rendering.
- `retrieval/embedder.py` — `DEFAULT_THREADS = 1`; `EMBED_THREADS` override.
- `tests/test_guardrails.py`, `tests/test_harness.py` — new, 64 tests. Suite now 147, all passing.
**Details:**
**The off-topic guardrail was removed because it was measured and found inverted.** It
scored queries against the corpus centroid. On the 3,039-chunk index, real Hindi questions
scored -0.035 and 0.252 while gibberish scored 0.184 and 0.234 — real questions ranked
*below* nonsense. The cause is structural, not a bad threshold: a centroid points in the
"average passage" direction, so similarity to it measures genericness, not topicality, and
a specific question is nearly orthogonal to the mean. The signal that does separate them is
max-similarity over the whole corpus — which is exactly the top dense score the confidence
gate already thresholds, so a second semantic gate would have been a less precise copy of
one the pipeline already runs, not an independent check. Semantic off-topic detection is
therefore `low_confidence`'s job, and it does it with the real score rather than a proxy.
Replaced with `language_mismatch`, which catches a failure the confidence gate provably
cannot: the English query "what is the capital of france" scores **0.628** against the Hindi
corpus — well above the 0.42 confidence threshold — because a multilingual embedder maps it
near Hindi passages about countries, and MS MARCO genuinely contains them. Unguarded, that
query gets a confident answer synthesised from passages that were never about it. The script
check catches it in ~0.02ms with no embedding. Romanised Hindi is exempted so code-mixed
transcripts still work.
Fixed during that work: `the` (थे, "were") was in the romanised-Hindi marker list, so
*every* English query exempted itself as Hinglish and the guardrail never fired. Words that
are also ordinary English (`the`, `me`, `par`, `to`, `is`, `us`) are now excluded — a marker
that fires on English costs the guardrail its entire purpose, while a missing marker costs
only a little Hinglish recall.
**ONNX threads pinned to 1**, measured not assumed: one query embed takes 112ms median /
121ms max with unbounded threads, and 104ms / 107ms at `threads=1`. A 12-layer forward pass
over a ~15-token query is too small to parallelise, so extra threads buy nothing and cost
synchronisation — visible mostly in the tail, which is what P100 measures. It also fixed the
index build: `parallel=N` re-instantiates the embedder per worker, so unbounded threads meant
6–8 workers x 16 intra-op threads with separate arenas, and the ONNX session load died with
"bad allocation". Build throughput improved as a side effect, 82 → 51 ms/chunk.
Guardrail routing verified against nine probe queries: Hindi questions answered (96–105ms),
English declined `language_mismatch`, Hinglish answered, gibberish declined, injection and
harm-seeking declined `input_safety`, an unanswerable Hindi question declined `low_confidence`.
Phases 4, 5 and 6 landed in one bump rather than three because they are mutually dependent —
an orchestrator cannot be verified without guardrails and a generator to orchestrate, and
bumping to a version whose pipeline cannot run end to end would have frozen a snapshot that
never worked. Later phases return to one bump per phase.
**Supersedes:** V3.0

## V5.0 — 2026-08-21
**Type:** Major
**Summary:** Phases 7–8 — benchmarks, threshold calibration, the deployable demo (web +
CLI), and two guardrail corrections that only measurement could have found.
**Files changed:**
- `benchmarks/latency.py` — new. P50/P70/P100 over real dataset queries; nearest-rank
  `percentile()`; excluded-and-reported warmup; declines counted, not dropped; detects
  extractive mode and labels the full-pipeline row as not-an-LLM-measurement.
- `benchmarks/calibrate_thresholds.py` — new. Scores answerable vs held-out vs
  out-of-domain vs gibberish populations on **both** dense and lexical signals; reports
  whether they separate at all rather than always emitting a confident threshold.
- `benchmarks/chunking_comparison.py` — new. recall@5 and MRR per strategy against the
  dataset's own `is_selected` labels, with chunk count and embed cost alongside.
- `guardrails/confidence.py` — rewritten. Added the lexical-support signal; `min_top_score`
  0.42 → **0.45**; documented the calibration data inline.
- `harness/orchestrator.py` — pre-retrieval guardrails now run **before** the embed.
- `guardrails/suite.py` — `run_pre_retrieval()` `query_vector` is now optional.
- `demo/app.py` — new. FastAPI: `/ask/voice`, `/ask/text`, `/health`, `/guardrails`;
  lifespan startup that records failures instead of crashing the container.
- `demo/index.html` — new. Browser UI; Web Audio API WAV capture; full guardrail/timing trace.
- `demo/cli.py` — new. `--mic`/`--file`/`--text`/`--demo-suite`.
- `stt/sarvam_client.py` — `load_audio_file()` accepts file-like objects.
- `Dockerfile`, `render.yaml`, `Procfile`, `.dockerignore` — new.
- `README.md`, `DECISIONS.md` — rewritten/corrected (see Details).
- `tests/test_demo_api.py` — new; `tests/test_guardrails.py`, `tests/test_harness.py` extended.
  Suite now **163**, all passing.
**Details:**
**Measured latency, 50 real queries, 15,449-chunk index: retrieval pipeline P50 99.17ms,
P70 102.39ms, P100 127.53ms — 100% of queries within the 200ms budget.** Query embedding is
95.4ms of the 99ms P50; dense search 1.03ms, BM25 0.29ms, fusion 0.09ms, all three
guardrails together under 0.1ms. 50/50 answered, no false refusals. The full-pipeline
number is **not** a valid end-to-end figure in this run — no `ANTHROPIC_API_KEY` is set, so
generation ran extractive and collapsed onto retrieval; the benchmark now detects and
labels that rather than reporting 99ms as "including generation".
**The confidence gate was rebuilt after calibration showed it barely worked.** Dense cosine
alone: answerable queries median 0.675, but *gibberish* median **0.774** — nonsense scored
higher than real questions, because a multilingual encoder maps arbitrary Devanagari into
the same region as Hindi prose. The best dense-only threshold (0.67) reached 66% balanced
accuracy only by refusing 47% of real questions. The lexical signal separates them
completely: every one of 6 gibberish queries scored **0.00** BM25 while all 60 answerable
queries scored 8.54–43.33. Requiring BM25 > 0 blocks 7/42 unanswerable queries at **zero
false refusals**. `min_top_score` moved to 0.45, just below the observed answerable minimum
of 0.472, because the dense populations overlap too heavily for a higher bar to be worth
its false-refusal cost. Documented plainly that plausible-but-absent questions
(out-of-domain: cosine 0.576–0.662, BM25 12.8–19.4) sit inside the answerable range on both
signals and are the generator's `NO_ANSWER` and the groundedness check's job, not this gate's.
Fixed while doing so: `lexical_score` is `None` on a chunk BM25 never matched, so the first
version's `if lexical_scores:` guard skipped the check exactly when *no* term matched —
letting every gibberish query through the signal added to catch it. An all-`None` list is
the strongest absence of evidence, not missing data.
**Pre-retrieval guardrails now run before the query is embedded.** They ran after, which
made the "rejects bad input before paying downstream cost" claim false: a refused injection
still spent ~90ms embedding. Measured after the fix, `input_safety` and `language_mismatch`
declines cost **0.0–0.1ms** instead of ~90ms. The ordering was a leftover constraint from
the deleted centroid guardrail, the only pre-retrieval check that ever needed the vector.
**Corrected two documentation claims that did not match the code.** `DECISIONS.md` D2 said
the `train` split; the loader reads `validation`, deliberately — the Hindi train parquet is
3.7GB against 462MB for validation with an identical schema, and nothing here is trained.
It also claimed `load_dataset("ai4bharat/MSMARCO-XI", "hi")`, which fails: the repo is flat
per-language parquet files, not named configs. D3's `intfloat/multilingual-e5-small` was
already unavailable in fastembed's registry and had been replaced by
`paraphrase-multilingual-MiniLM-L12-v2`; D3 now says so and explains the prefix consequence.
The demo suite's "answerable" examples were replaced with **real indexed corpus queries**.
The obvious-looking "भारत की राजधानी क्या है?" is not in this corpus — retrieval returns
passages about the weather in Bokaro and flights to Bangalore, and the gate correctly
refuses at 0.440. It is kept, relabelled, because a refusal that can be *shown* correct
demonstrates the guardrail better than a plausible one.
The browser records via the Web Audio API rather than `MediaRecorder`: Chrome emits
WebM/Opus, which libsndfile cannot demux, and the alternative was an ffmpeg dependency on
the deploy host plus a transcode in the request path. The client encodes 16kHz mono WAV
directly and rejects silence and sub-0.4s clips before spending an STT call.
**Supersedes:** V4.0

## V6.0 — 2026-08-21
**Type:** Major
**Summary:** Rebuilt the web UI (boxy minimal, fluid 320–3840px, micro-animations), added
deployment docs covering the Vercel constraint, recorded the chunking-comparison results,
and did a prose pass across the docs and docstrings.
**Files changed:**
- `demo/index.html` — rewritten. Cal Sans / JetBrains Mono / Disket Mono (Space Mono
  fallback) / Noto Sans Devanagari type system; `clamp()` fluid scale; bouncy easing on
  interactive elements; `prefers-reduced-motion` and light-scheme support; configurable
  `__API_BASE__`; per-stage timing shares; chip clipping fix.
- `demo/config.js` — new. Retargets the UI at a non-same-origin backend.
- `demo/fonts/README.md` — new. Disket Mono drop-in instructions.
- `demo/app.py` — mounts `/fonts` via `StaticFiles`.
- `../DEPLOY.md` — new. Render / Vercel+Render / HF Spaces paths, GitHub push, pre-submit
  checklist.
- `../HANDOFF.md` — rewritten; consolidated "what I need from you" list.
- `README.md` — rewritten; chunking-comparison results and UI section added.
- `DECISIONS.md`, `chunking/*.py`, `data/loader.py`, `generation/generator.py`,
  `guardrails/groundedness.py`, `harness/types.py`, `stt/sarvam_client.py` — prose pass.
- `tests/test_demo_api.py` — `test_root_serves_the_ui` now asserts the Devanagari font link
  and the API-base hook rather than the old heading text. Suite still 163, all passing.
**Details:**
**Chunking comparison, finally measured** (250 examples, 60 queries, recall@5 against the
dataset's own `is_selected` labels): semantic 0.967 / MRR 0.770, fixed_size 0.950 / 0.752,
recursive 0.950 / 0.757, metadata_aware 0.950 / 0.759, shipped ensemble 0.950 / 0.752, and
**all four together 0.917** — worse than any single strategy. Stacking splitters produces
near-duplicate chunks that occupy top-5 slots the relevant passage needed. That result is
now in the README, because "chunking should be vast" is easy to misread as "use more
splitters" and the data says otherwise.
Semantic won and is still not the shipped default. Tried to rebuild the production index on
it; the corpus-wide sentence-embedding pass died inside an ONNX MatMul with a bad allocation
at 1,500 examples, though it is fine at the 250-example benchmark scale. Fixing it means
windowing that pass, which is a real change and not one to make hours before a deadline, so
the shipped ensemble stays `metadata_aware + recursive` and gives up 1.7 points of recall@5
for something that builds reliably. Both the finding and the gap are documented rather than
quietly dropped.
**Vercel cannot host this backend, and the numbers are in `DEPLOY.md`.** Measured payload:
onnxruntime 45MB + numpy 34MB + faiss 15MB + fastembed 2MB, the embedding model ~240MB, the
index 38MB. Serverless functions cap at 250MB unzipped; even dropping pandas/pyarrow/datasets
(build-time only) leaves ~375MB. Cold starts would also load a 240MB ONNX session per
invocation, which is untenable for a project whose headline is a 99ms P50. The frontend can
live on Vercel — `demo/config.js` sets `window.__API_BASE__` and CORS is already open — but
the pipeline needs a container host. Render is the recommended single-URL path.
**UI.** Fluid type and spacing via `clamp()` interpolated across 320–3840px, so the
stylesheet has essentially one breakpoint. Verified by measurement at 320 / 768 / 3840: no
page-level horizontal overflow at any of them, 44px minimum tap targets at the small end,
shell capped at 1728px at the large end. Wide tables scroll inside their own container
instead of pushing the page sideways.
Fixed while building it: the embedding-model chip renders
"paraphrase-multilingual-MiniLM-L12-v2" at 343px with `white-space: nowrap`, which forced the
entire page into horizontal scroll at 320px. One label was setting the site's minimum width.
**Noto Sans Devanagari was added and is not cosmetic.** Neither JetBrains Mono nor Cal Sans
has Devanagari glyphs, and every passage and answer in this system is Hindi — without it the
actual content falls back to whatever the OS supplies. Disket Mono is genuinely unavailable:
Fontfabric doesn't redistribute through npm or Google Fonts, so `@font-face` points at
`/fonts/` for a local drop-in and Space Mono stands in until then, rather than letting the
stack collapse silently to a default sans.
**Process note — a versioning deviation, recorded rather than hidden.** This work was
authored inside `V5.0/` after V5.0 had already been committed, and only bumped to `V6.0/`
afterwards. Section 1.3 says a version folder freezes at bump time and is never edited
again, so `V5.0/` is *not* a clean snapshot of what its own changelog entry describes — it
contains this entry's changes too. `V6.0/` onward is correct. Noting it because a version
history that quietly papers over its own violations is worth less than one that doesn't.
**Supersedes:** V5.0
