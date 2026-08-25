# Voice-Enabled RAG — Team 0neHackers

**HH Goa 2026, Task 2.** Live at **https://ragoa-voice-rag.up.railway.app/**

Ask a question out loud in Hindi. Get an answer grounded in
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — or a refusal
that tells you why.

Shanzal Firoz · Aditya Vishwakarma · Kanishka Rajput

```
 voice ──▶ Sarvam STT ──▶ [safety] ──▶ [language] ──▶ (optional en→hi) ──▶ embed
                                                                              │
                                                              dense (FAISS) ◀─┤
                                                              BM25 (inverted) ◀┘
                                                                    │
                                                                 RRF fusion
                                                                    │
                                                          [confidence gate]
                                                                    │
                       answer ◀── [groundedness] ◀── Sarvam 105b-conversations
```

Every stage runs inside a harness that times it, retries only what's worth retrying, and
turns failures into typed values instead of exceptions. Every refusal comes back as a
structured response with a machine-readable reason.

---

## Numbers

50 real MSMARCO-XI queries, 15,449-chunk index, **live generation**
(`benchmarks/results_latest.json`):

| Retrieval pipeline — the number held to the 200ms bar | |
|---|---|
| **P50** | **102.09 ms** |
| **P70** | **106.01 ms** |
| **P100** | **117.85 ms** |
| mean / min / stdev | 103.10 / 86.88 / 7.59 ms |
| inside budget | **100% of queries** |

| Full pipeline — retrieval + LLM generation + groundedness | |
|---|---|
| **P50** | **2801.65 ms** |
| **P70** | **3229.50 ms** |
| **P100** | **6139.86 ms** |
| outcomes | 45/50 answered, 5 declined on groundedness |

Where the time actually goes — embedding is the retrieval floor, generation dominates
everything else:

| stage | P50 | P100 |
|---|---|---|
| generation (Sarvam) | 2686.33 ms | 6051.76 ms |
| query embed | 98.60 ms | 116.19 ms |
| dense search (FAISS) | 1.11 ms | 2.43 ms |
| BM25 lexical | 0.31 ms | 0.65 ms |
| RRF fusion | 0.10 ms | 6.98 ms |
| groundedness check | 0.52 ms | 1.05 ms |
| `input_safety` | 0.03 ms | 0.08 ms |
| `low_confidence` | 0.03 ms | 0.04 ms |
| `language_mismatch` | 0.01 ms | 0.05 ms |

| | |
|---|---|
| Corpus | 15,449 unique chunks from 1,500 MSMARCO-XI Hindi examples (`validation` split) |
| Index | FAISS `IndexFlatIP`, in-process, exact |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2`, 384-dim, ONNX on CPU, `threads=1` |
| Generation | `sarvam-105b-conversations` |
| Chunking | 4 strategies, compared on recall@5 and MRR |
| Guardrails | 4, across 3 pipeline positions |
| Tests | 177 |

### Which number is the 200ms one

Two latencies, both reported, neither hidden.

**Retrieval-pipeline latency** — guardrails, query embedding, dense + lexical search,
fusion. This is the one held to the task's bar, and it clears it on every query.

**Full-pipeline latency** adds LLM generation and the groundedness check, and it is
roughly 2.8 seconds. Almost all of that is generation.

The split isn't a dodge, and now that generation is live the gap is plain to see:
generation is ~2680 ms of a ~2800 ms full-pipeline P50. That figure is a third-party
provider's time-to-first-token — nothing we write moves it, so folding it into the headline
would mean reporting Sarvam's inference infrastructure as if it were our pipeline. Both
numbers are in the benchmark output and on screen in the demo for every request. Worth
flagging because it's the one place this build interprets the task rather than just
following it. (`DECISIONS.md` D5.)

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
cp .env.example .env                                 # then add your keys
python -m retrieval.build_index --limit 1500 --out index_store
python -m demo.app                                   # http://localhost:8000
```

Building the index takes about 13 minutes for 15k chunks. If you just want to poke at it:

```bash
python -m retrieval.build_index --limit 200 --out index_dev
python -m demo.cli --demo-suite --index index_dev
```

### Keys

**One key drives everything.**

| Variable | Powers | If it's missing |
|---|---|---|
| `SARVAM_API_KEY` | speech-to-text, generation, translation, text-to-speech | STT returns a typed `CONFIG` error — **never** a fake transcript. Generation drops to labelled `extractive` mode. Translation and TTS are unavailable. |

There used to be a second provider here — Claude Haiku for generation. It's gone.
Consolidating onto Sarvam removed two keys, two auth schemes, two rate limits and two
independent ways for the demo to fail in front of judges. See `DECISIONS.md` D6.

The key isn't needed to run the retrieval benchmark or the guardrail tests.

Deploying is its own thing — see [`LIVE_LINK.md`](LIVE_LINK.md) and the deployment section
of the [root README](../README.md). Short version: it runs on Railway, Vercel cannot host it
at all (250 MB function limit against a ~375 MB payload), and no 512 MB free tier can either
— the process measures ~717 MB resident and the weight is the embedding model, not the
corpus.

---

## How it works

### The dataset

`load_dataset("ai4bharat/MSMARCO-XI", "hi")` doesn't work. The repo isn't exposed as named
language configs — it's a flat set of per-language parquet files — so `data/loader.py` goes
straight at `validation/hinval.parquet` and caches an extracted slice as JSONL.

We read `validation` rather than `train` because the Hindi train parquet is a single 3.7 GB
file against 462 MB for validation, with an identical schema. Downloading 3.7 GB to slice
1,500 examples off the front is a bad trade, and nothing here is trained anyway — this is a
retrieval corpus, so the train/validation split carries no methodological weight.

### Speech-to-text — Sarvam, streaming

`stt/sarvam_client.py` prefers Sarvam's **WebSocket streaming** endpoint and falls back to
batch — and on our account, the fallback is what actually runs. More on that below, because
it's a real deviation from the plan rather than a detail.

Sarvam over ElevenLabs because the dataset is AI4Bharat's Indic retrieval benchmark and the
demo gets spoken by an India-based team. Sarvam is tuned for Indic phonetics and code-mixed
Hindi-English, which is exactly what it's going to hear.

**Tested against the live API, and streaming doesn't work on this key.** The WebSocket
endpoint connects, authenticates, validates the model — it rejects `saarika:v2` and
`saarika:flash` as deprecated, so it's clearly parsing our requests — accepts every audio
frame without complaint, and then returns **zero transcript frames**. We tried 100ms, 200ms
and 500ms chunk cadences, with and without the model parameter, and the
`speech-to-text-translate` endpoint too. The batch REST endpoint transcribes the identical
audio perfectly, first try.

Two genuine bugs surfaced while establishing that, both invisible without a real key:

- The stream terminator was wrong. We sent `{"event": "stop"}`, which several streaming APIs
  document. Sarvam validates *every* frame as an audio request, so it answered
  `Invalid request: 'audio' must not be None` — and our parser was quietly swallowing
  `type: "error"` frames, reporting "no transcript" over the top of the one useful
  diagnostic.
- The receive loop never terminated. Sarvam doesn't close the socket, so `async for raw in ws`
  waited until the outer 34-second timeout fired, at which point the batch fallback rescued
  the request. Streaming looked broken when it was really just never being allowed to finish.

So the shipped path is **batch, labelled as batch**. `Transcript.transport` reads `"batch"`,
the demo displays it, and this README says it — a demo should never claim to be streaming when
it isn't. The client still probes streaming once per process; the moment that probe comes back
empty it latches and stops paying for it, so the cost is one probe rather than ~1.5s on every
request. `STT_TRANSPORT=streaming` re-exercises the path with no code change if Sarvam turns it
on for the account.

Measured round trip on 2.1s of Hindi audio: **~0.9–1.2s**, batch.

**There's no typed-text shortcut anywhere in the voice stage.** A missing key gets you a
typed error, not a substitute. `Transcript.is_real_audio` is set only by code paths that
pushed actual audio bytes at a recogniser, and the UI shows it.

In the browser we encode 16 kHz mono WAV ourselves through the Web Audio API instead of using
`MediaRecorder`. Chrome's `MediaRecorder` emits WebM/Opus and libsndfile has no WebM demuxer,
so the choice was either installing ffmpeg on the deploy host and transcoding inside the
request, or just encoding the right format where the samples already live. The client also
throws out silence and sub-0.4s clips before spending an STT call — a recorder that captured
nothing is the single most common way a "voice-enabled" demo turns out silent.

### Chunking — four strategies, and we measured them

| Strategy | What it does | Why it's here |
|---|---|---|
| `fixed_size` | 256 tokens, ~18% overlap | Baseline. Cheap recall floor. |
| `semantic` | Splits at embedding-similarity breakpoints between sentences | Chunks don't get cut mid-idea |
| `recursive` | Paragraph → sentence → token fallback | Respects natural boundaries |
| `metadata_aware` | Each MSMARCO passage as a chunk, carrying `query_id`, `passage_idx`, `source_lang`, `is_selected` | The corpus is *already* segmented by humans — re-splitting throws that away |

`python -m benchmarks.chunking_comparison` scores all four against the dataset's own
`is_selected` relevance labels. Measured over 250 examples and 60 queries:

| strategy | chunks | recall@5 | MRR | embed time |
|---|---|---|---|---|
| **semantic** | 3,771 | **0.967** | **0.770** | 181.9 s |
| fixed_size | 2,557 | 0.950 | 0.752 | 117.0 s |
| recursive | 2,505 | 0.950 | 0.757 | 116.6 s |
| metadata_aware | 2,505 | 0.950 | 0.759 | 123.2 s |
| ensemble (shipped) | 2,530 | 0.950 | 0.752 | 124.4 s |
| ensemble (all four) | 5,110 | **0.917** | 0.743 | 218.9 s |

Two things fall out of that table, and neither is what we expected going in.

**Throwing everything in makes it worse.** The all-four ensemble scores 0.917 — below every
individual strategy. More chunks means near-duplicate versions of the same passage from
different splitters, and those duplicates eat top-5 slots the actually-relevant passage
needed. "Vast chunking" doesn't mean stacking splitters.

**Semantic wins, and the shipped index doesn't use it.** That's a real gap, so here's the
honest reason: semantic chunking embeds every sentence in the corpus in one batched pass, and
at 1,500 examples that allocation dies on this machine — we tried, it OOM'd inside an ONNX
MatMul. It's fine at the 250-example benchmark scale. Closing the gap means windowing that
pass, which is a genuine fix and not one to rush the night before a deadline. So the shipped
index runs `metadata_aware + recursive` (`chunking/registry.py:DEFAULT_ENSEMBLE`) and gives up
1.7 points of recall@5 for something that reliably builds.

Cross-strategy duplicates get deduplicated at build time regardless — running several
splitters over one corpus produces heavy overlap, and indexing the same text five times just
inflates the index and returns five copies of one passage.

### Retrieval — hybrid, in-process

Dense search over FAISS `IndexFlatIP`, fused with BM25 by reciprocal rank fusion.

**Exact, not approximate.** At ~15k vectors an exact search is one matmul in well under a
millisecond, so an ANN index buys no measurable time and costs recall. Worse, that recall loss
would shift the score distribution the confidence guardrail thresholds against — making a
safety property depend on the index's approximation error.

**In-process, not hosted.** Pinecone or Qdrant Cloud add 20–100 ms of round-trip before any
compute happens. A 200 ms budget doesn't survive that.

**RRF, not a weighted blend.** Cosine lives in [-1, 1]; BM25 is unbounded and
corpus-dependent. Any `α·dense + (1-α)·bm25` needs a normalisation you have to re-tune per
corpus. RRF fuses on rank and needs none. The cost is that fused scores mean nothing in
absolute terms — which is exactly why `ScoredChunk` keeps `dense_score` separately and the
confidence gate reads raw cosine.

**BM25 is hand-written, not `rank_bm25`.** `BM25Okapi.get_scores` walks every document for
every query term. Measured at 3.25 ms per query on 3k chunks and scaling linearly, that's
~20 ms at full corpus size. An inverted index only touches documents that actually contain a
query term: **0.12 ms, 27× faster.** The IDF form differs too, and not by accident —
`log(1 + (N-df+0.5)/(df+0.5))` instead of the textbook `log((N-df+0.5)/(df+0.5))`, which goes
*negative* for terms appearing in more than half the corpus and lets a common word drag
documents down the ranking.

### The harness

`harness/orchestrator.py`. Not a function that calls an LLM:

- **Typed I/O everywhere.** Every stage returns `StageResult[T]` — a value or a typed
  `StageError`, plus elapsed time. Nothing raises across a stage boundary.
- **Retries that discriminate.** `ErrorKind` splits `TRANSIENT`/`RATE_LIMITED` (retry) from
  `AUTH`/`CONFIG`/`VALIDATION` (don't — retrying a rejected API key just spends latency to
  fail the same way). Backoff uses **full jitter**, because the benchmark fires 50 queries in
  a tight loop and un-jittered retries collide on every attempt. There's a **deadline** too,
  since three attempts against a hung socket is a correct retry policy and a dead demo at the
  same time.
- **Per-stage error handling.** Failed STT gives you a structured error with timings intact.
  A failed generator still returns what retrieval found.
- **Failure paths get timed too** — a stage that took four seconds to fail is precisely what a
  latency-graded pipeline needs to be able to see.
- **The query gets embedded once** and threaded through the guardrails and both retrievers. At
  ~104 ms a pop, embedding twice would blow the budget on its own.

### Guardrails — four of them, three positions

| # | Guardrail | Position | Catches |
|---|---|---|---|
| 1 | `input_safety` | pre-retrieval | Harm-seeking phrasing, prompt injection, absurd transcript length |
| 2 | `language_mismatch` | pre-retrieval | Queries in a script the corpus isn't written in |
| 3 | `low_confidence` | post-retrieval, pre-generation | The corpus doesn't contain the answer |
| 4 | `groundedness` | post-generation | Model had good context and drifted anyway |

Every trip returns `{status: "declined", reason: "<guardrail>", ...}` with the score, the
threshold, and a human-readable explanation. A decline is a designed outcome — never a crash,
never a silently empty answer.

They fail **open** if they break internally, because these are quality gates on a public Q&A
demo, not a security boundary. The exception is `input_safety`, which fails closed —
withholding is the entire point of that one.

Guardrails 1 and 2 both run **before** the query is embedded. That ordering matters more than
it sounds: embedding is ~99 ms of the ~102 ms retrieval P50, so a refusal that happens after
the embed saves almost nothing. Measured, a blocked injection costs **0.01–0.03 ms** instead
of ~99 ms.

#### One of these was built, measured, and thrown away

Guardrail 2 started life as an embedding-based off-topic detector scoring queries against the
corpus centroid. On the real index it turned out to be **inverted**:

| query | centroid sim | max-sim to corpus |
|---|---|---|
| "भारत की राजधानी क्या है?" (real) | **−0.035** | 0.440 |
| "मधुमेह के लक्षण क्या हैं?" (real) | 0.252 | 0.678 |
| "asdkjh qwe zxcvbn" (gibberish) | **0.184** | 0.381 |
| "aaaaa bbbbb ccccc" (gibberish) | 0.234 | 0.398 |

Real questions scored *below* gibberish. That's structural, not a bad threshold: a centroid
points in the direction of the average passage, so similarity to it measures how *generic*
something is, not how on-topic — and a specific question is nearly orthogonal to the mean. No
threshold repairs a signal pointing the wrong way.

The column that does separate them is max-similarity over the corpus, which is exactly the top
dense score guardrail 3 already thresholds. So semantic off-topic detection belongs to
guardrail 3, working from the real score instead of a proxy. A second semantic gate would have
been a blurrier copy of one we already run, not an independent check.

What replaced it catches something guardrail 3 provably can't. The English query *"what is the
capital of france"* scores **0.628** against the Hindi corpus — comfortably above the
confidence threshold — because a multilingual encoder maps it near Hindi passages about
countries, and MS MARCO genuinely has those. Left alone it gets a confident answer built from
passages that were never about it. A script check catches it in ~0.02 ms with no embedding at
all. Romanised Hindi (*"bharat ki rajdhani kya hai"*) is exempted, since Sarvam transcribes
code-mixed speech and Hinglish isn't a mismatch.

#### The confidence gate needed rebuilding too

It thresholded on dense cosine alone. Calibrated against the real index, that was close to
useless:

| population | n | min | p05 | median | p95 | max |
|---|---|---|---|---|---|---|
| answerable | 60 | 0.472 | 0.547 | 0.675 | 0.837 | 0.860 |
| held-out | 30 | 0.450 | 0.513 | 0.624 | 0.702 | 0.747 |
| out-of-domain | 6 | 0.576 | 0.583 | 0.611 | 0.652 | 0.662 |
| gibberish | 6 | 0.655 | 0.672 | **0.774** | 0.831 | 0.845 |

Gibberish — random Devanagari syllables — scored a **higher median cosine than real
questions**. The best dense-only threshold (0.67) blocks 78.6% of unanswerable queries by also
refusing **47% of real ones**. Unusable.

The lexical signal separates them outright:

| population | n | top BM25: min | median | max | scoring zero |
|---|---|---|---|---|---|
| answerable | 60 | 8.54 | 20.28 | 43.33 | **0/60** |
| held-out | 30 | 0.00 | 13.26 | 23.38 | 1/30 |
| out-of-domain | 6 | 12.76 | 16.47 | 19.38 | 0/6 |
| gibberish | 6 | 0.00 | 0.00 | 0.00 | **6/6** |

Every gibberish query scores exactly zero, because none of its tokens exist anywhere in the
corpus. Every one of the 60 answerable queries matched something. Requiring BM25 > 0 blocks 7
of 42 unanswerable queries at **zero false refusals** — and it's free, since hybrid retrieval
already computes the number.

So the gate wants three things: lexical support, a top cosine above 0.45 (just under the
observed answerable minimum of 0.472 — deliberately permissive, since the dense populations
overlap too much for a higher bar to be worth its false-refusal cost), and a **margin** over
the mean of the remaining chunks. A query the corpus answers gives a peaked score
distribution; one it can't gives a flat one, and a flat distribution at a respectable absolute
level is the signature of "vaguely on topic, no actual answer."

**What it still can't do**, stated rather than buried: plausible Hindi questions the corpus
has no answer to ("मेरा पासवर्ड क्या है?") score 0.576–0.662 cosine and 12.8–17.9 BM25 —
inside the answerable range on both. They're built from ordinary Hindi words that really do
appear across a web corpus, so no retrieval-score threshold separates them without refusing
real questions too. Catching those is what the generator's `NO_ANSWER` refusal and the
groundedness check are for. This gate is one layer of three.

Thresholds get regenerated by `python -m benchmarks.calibrate_thresholds`, which reports
whether the populations actually separate instead of always emitting a confident-looking
number. A decisive threshold over overlapping distributions is a guardrail that fires at
random.

### Generation

`sarvam-105b-conversations`, `temperature=0`, prompted to ground its answer, allowed to
refuse via `NO_ANSWER`, required to cite `[n]`, and pinned to the passages' language.

**Model choice was measured.** Sarvam offers two chat models and they're not
interchangeable here — `sarvam-105b` is a reasoning model that took **25.5s and 865 tokens**
on a task where the conversations variant took **2.3s and 21**. On grounded extraction the
answer is already sitting in the context; reasoning about it is pure latency. Worse, it
spent the whole token budget thinking and returned `content: null`.

Retrieved passages are numbered and delimited, and the model is told they're reference
material. Corpus text is scraped from the web and untrusted — a passage containing "ignore
the above" has to read as data, not instruction.

**Two things broke the moment a real model started writing real prose**, neither visible
while generation was extractive:

- **It answered in English.** Hindi question, Hindi passages, fluent English answer — every
  time — despite the prompt asking it to match the question's language. That breaks a Hindi
  voice demo outright and it zeroes the groundedness check, which compares answer tokens
  against Hindi context. The instruction now comes last, as an explicit override, and keys
  off the *passages* rather than the question.
- **The groundedness threshold was calibrated for copies, not prose.** At 0.45 it refused
  37.5% of faithful generated answers. Re-measured over 40 samples and moved to 0.20. Full
  table in `DECISIONS.md` D9.

And one guardrail bug that only a citing model could expose: the unsupported-number check
was reading `[1]`, `[2]` citation markers as numeric claims, so every correctly-cited answer
got flagged for "fabricated figures" — rejecting precisely the behaviour the prompt asks
for. Citations are stripped before the number scan now.

`NO_ANSWER` isn't an error. It's the model using the refusal the prompt authorises, and the
orchestrator turns it into the same typed decline a guardrail produces, so a user can't tell
which component noticed the corpus fell short. It's matched anywhere in the response, not
just at the start — models routinely explain themselves first and append it.

Without `SARVAM_API_KEY`, generation drops to `extractive` — top passage, verbatim, labelled
`mode: "extractive"` in the response and on screen. A copied passage never gets presented as
a generated answer.

### Translation — English → Hindi, opt-in

The corpus is Hindi and `language_mismatch` refuses Latin-script queries for a measured
reason. Ticking **"Translate my question from English to Hindi first"** runs the query
through Sarvam's translate endpoint before anything else, so retrieval compares like with
like instead of leaning on cross-lingual alignment that has already been shown to mislead.

It stays opt-in deliberately. Auto-translating every Latin-script query would silently
disable a guardrail that exists because of a real failure, and it would mangle romanised
Hindi — "bharat ki rajdhani kya hai" is already a Hindi question, and translating it *as
English* produces nonsense. Both strings are kept in the response, and the UI shows the
original alongside the translation, so a translated query is never passed off as what the
user typed. Measured at ~860 ms.

### Speak aloud — both halves of the loop

Sarvam's `bulbul` TTS reads back **the question and the answer**, in Hindi. For a Hindi
demo that matters more than it sounds: a viewer who doesn't read Devanagari can still tell
the pipeline understood and answered.

Citation markers are stripped before synthesis — `[1][2]` is useful on screen and becomes
"one two" mid-sentence when spoken. Inputs over Sarvam's 1500-character limit are split on
sentence boundaries, including the Devanagari danda, and the returned clips are joined into
one WAV rather than concatenated raw, which would leave a RIFF header in the middle of the
stream. One shared `<audio>` element means pressing the second button stops the first —
two Hindi voices talking over each other is worse than no audio at all.

---

## Benchmarks

```bash
python -m benchmarks.latency --n 50                    # P50/P70/P100, both numbers
python -m benchmarks.latency --n 50 --no-generation    # retrieval only
python -m benchmarks.chunking_comparison               # recall@5 + MRR per strategy
python -m benchmarks.calibrate_thresholds              # confidence-gate calibration
```

Three deliberate choices keep the latency benchmark honest:

- **Queries come from the dataset**, sampled from the corpus that was actually indexed — not a
  hand-picked list of questions known to work.
- **Warmup runs are excluded and reported.** The first ONNX inference pays graph
  initialisation; leaving it in makes P100 a cold-start measurement, dropping it quietly hides
  a real cost.
- **Declines are counted, not discarded.** A guardrail decline is a fast path, so removing
  those runs would drag every percentile down by deleting the cheapest requests. They stay in,
  and the decline rate sits next to the percentiles — so a suspiciously fast P50 shows up as a
  high decline rate instead of looking like speed.

P100 is the true maximum (nearest-rank), not an interpolated percentile reporting a number no
request ever took.

---

## The web UI

`demo/index.html`, served by FastAPI at `/`. Boxy and minimal on purpose — 2px corners, 1px
rules, a faint grid, and type doing most of the work.

- **Cal Sans** for display headings, **JetBrains Mono** for body and data, **Disket Mono**
  (falling back to Space Mono) for labels and numbers, **Noto Sans Devanagari** for all Hindi
  content. That last one isn't optional: neither JetBrains Mono nor Cal Sans has Devanagari
  glyphs, and every passage in this system is Hindi.
- **Fluid from 320px to 3840px.** Every type step and spacing unit is a `clamp()` interpolating
  across exactly that range, which is why there's essentially one breakpoint in the whole
  stylesheet. Verified at 320, 768 and 3840 — no horizontal overflow, 44px tap targets at the
  small end, content capped at 1728px at the large end so it doesn't stretch into nonsense.
- **Micro-animations** on anything interactive: a bouncy overshoot easing
  (`cubic-bezier(0.34, 1.56, 0.64, 1)`) reserved for things you pressed, a flatter curve for
  ambient motion, staggered reveals on results. All of it collapses under
  `prefers-reduced-motion`.
- **Wide content scrolls inside its own box.** Tables get a horizontal scroller rather than
  pushing the page sideways.

The response pane shows the whole trace — every guardrail verdict with score and threshold,
per-stage timings with percentage share, retrieved passages with both scores, and the raw JSON.
A voice RAG demo that only shows its answer is indistinguishable from one that made it up.

Fonts are documented in `demo/fonts/README.md`. `demo/config.js` retargets the UI at a
different backend origin if you split the frontend off.

---

## Layout

```
V{X.Y}/
├── data/          MSMARCO-XI loading + normalisation
├── chunking/      four strategies + registry
├── retrieval/     embedder, FAISS index, BM25, hybrid retriever, build CLI
├── generation/    grounding prompt + Sarvam chat client + extractive fallback
├── translation/   opt-in English→Hindi query translation
├── tts/           Hindi speech synthesis for question and answer
├── guardrails/    input safety, language match, confidence gate, groundedness, suite
├── harness/       orchestrator, typed I/O, retry policy, assembly factory
├── benchmarks/    latency, chunking comparison, threshold calibration
├── demo/          FastAPI app + browser UI + CLI
└── tests/         177 tests
```

## Versioning

Every phase of work is a numbered snapshot under `master_repo/V{MAJOR}.{MINOR}/`, with
`master_repo/VERSION` pointing at the current one and `master_repo/CHANGELOG.md` carrying a
dated, file-by-file entry per bump. Old version folders get frozen on bump and never touched
again. `bump.sh` handles the mechanical part.

## What this doesn't do well

- The groundedness check is lexical overlap by default, not entailment. It catches invented
  entities and fabricated figures. It won't catch a wrong *inference* drawn from words that
  were genuinely in the context. There's an opt-in LLM entailment path (`use_llm=True`) that
  costs a second round-trip.
- `input_safety` is pattern-based. Overt harm-seeking and injection phrasing, yes. Obfuscated
  or adversarially-worded attacks, no. It's a first-line filter for a public demo, not a
  moderation system.
- The shipped index uses `metadata_aware + recursive`, giving up 1.7 points of recall@5 to
  `semantic`. Semantic chunking's corpus-wide sentence-embedding pass used to OOM at
  production scale; it is windowed now (`chunking/semantic.py`, `EMBED_WINDOW = 4096`), but
  the shipped ensemble was calibrated and benchmarked without it and was not re-cut before
  the deadline.
- Query embedding is ~99 ms of the ~102 ms retrieval P50. Everything else together is under
  2 ms. Making this meaningfully faster means a smaller embedding model, not
  micro-optimising search.

## Also see

- `DECISIONS.md` — every design call and its reasoning, including the ones revised after
  measuring
- `../CHANGELOG.md` — full version history
- `../README.md` — architecture, measured numbers, and the deployment write-up
- `LIVE_LINK.md` — hosting options, measured memory, and why the free tiers don't fit
