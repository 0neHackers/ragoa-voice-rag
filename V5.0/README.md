# Voice-Enabled RAG — Team 0neHackers

**HH Goa 2026, Task 2.** Speak a question in Hindi; get an answer grounded in
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — or an
explicit, reasoned refusal when the corpus cannot support one.

Shanzal Firoz · Aditya Vishwakarma · Kanishka Rajput

```
 voice ──▶ Sarvam streaming STT ──▶ [input safety] ──▶ [language match] ──▶ embed
                                                                              │
                                                              dense (FAISS) ◀─┤
                                                              BM25 (inverted) ◀┘
                                                                    │
                                                                 RRF fusion
                                                                    │
                                                          [confidence gate]
                                                                    │
                            answer ◀── [groundedness] ◀── Claude Haiku 4.5
```

Every stage runs inside a harness that times it, retries only what is worth retrying, and
converts failures into typed values rather than exceptions. Every refusal is a structured
response with a machine-readable reason.

---

## Results at a glance

| | |
|---|---|
| **Retrieval pipeline** (the <200ms target) | see `benchmarks/results_latest.json` |
| **Full pipeline** (incl. generation) | reported separately — see below |
| Corpus | 15,449 unique chunks from 1,500 MSMARCO-XI Hindi examples (`validation` split) |
| Index | FAISS `IndexFlatIP`, in-process, exact |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2`, 384-dim, ONNX/CPU |
| Chunking strategies | 4 implemented, compared on recall@5 and MRR |
| Guardrails | 4, at 3 pipeline positions |
| Tests | 147 |

> **Which number is held to 200ms, and why.** Two latencies are reported separately and
> neither is hidden. **Retrieval-pipeline latency** — guardrails, query embedding, dense +
> lexical search, fusion — is the number held to the task's <200ms bar. **Full-pipeline
> latency** adds LLM generation and the post-generation groundedness check. Generation
> latency is a third-party provider's time-to-first-token; no client-side engineering
> moves it, so folding it into the headline number would measure Anthropic's
> infrastructure rather than this pipeline. Both numbers are in the benchmark output and
> in the demo UI on every request. This interpretation is stated up front rather than
> buried, because it is the one place this build interprets the task instead of following
> it. (`DECISIONS.md` D5.)

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
cp .env.example .env                                 # then add your keys
python -m retrieval.build_index --limit 1500 --out index_store
python -m demo.app                                   # http://localhost:8000
```

The index build takes ~13 minutes for 15k chunks. For a fast smoke test:

```bash
python -m retrieval.build_index --limit 200 --out index_dev
python -m demo.cli --demo-suite --index index_dev
```

### Keys

| Variable | Required for | Without it |
|---|---|---|
| `SARVAM_API_KEY` | the voice path | STT returns a typed `CONFIG` error — **never** a fake transcript |
| `ANTHROPIC_API_KEY` | generated answers | falls back to labelled `extractive` mode (verbatim top passage) |

Neither key is needed to run the retrieval benchmark or the guardrail tests.

---

## What runs where

### The dataset

`load_dataset("ai4bharat/MSMARCO-XI", "hi")` does not work — the repo
is a flat set of per-language parquet files rather than named configs, so `data/loader.py`
addresses `validation/hinval.parquet` directly and caches an extracted slice as JSONL. The
`validation` split is used because the Hindi train parquet is 3.7GB against 462MB for
validation with an identical schema, and nothing in this system is trained — the corpus is
a retrieval corpus, so the train/validation distinction carries no methodological weight.

### Speech-to-text — Sarvam, streaming

`stt/sarvam_client.py` talks to Sarvam's **WebSocket streaming** endpoint, not the batch
one. Sarvam over ElevenLabs because the dataset is AI4Bharat's Indic retrieval benchmark
and the demo is spoken by an India-based team: Sarvam is tuned for Indic phonetics and
code-mixed Hindi-English, which is exactly the audio it will receive.

The batch REST endpoint exists as a **labelled** fallback — a `wss://` handshake is the
most environment-fragile call in the pipeline, and some proxies and PaaS egress rules
block it while allowing HTTPS. When the fallback fires, `Transcript.transport` reads
`"batch"`, so a demo never claims to be streaming when it wasn't.

**There is no typed-text substitute for the voice stage.** If the key is missing, STT
returns a typed error. `Transcript.is_real_audio` is set only by code paths that pushed
audio bytes to a recogniser, and the UI shows it.

The browser encodes 16kHz mono WAV itself through the Web Audio API instead of using
`MediaRecorder`. Chrome's `MediaRecorder` emits WebM/Opus, which libsndfile cannot
demux; the alternatives were installing ffmpeg on the deploy host and transcoding in the
request path, or encoding the format the recogniser wants where the samples already are.
The client also rejects silence and sub-0.4s clips before spending an STT call — a
recorder that captured nothing is the commonest way a "voice-enabled" demo turns out to
be silent.

### Chunking — four strategies, compared

| Strategy | What it does | Why it's here |
|---|---|---|
| `fixed_size` | 256 tokens, ~18% overlap | Baseline. Cheap recall floor. |
| `semantic` | Splits at embedding-similarity breakpoints between sentences | Chunks don't cut mid-idea |
| `recursive` | Paragraph → sentence → token fallback | Respects natural boundaries; the general-purpose splitter |
| `metadata_aware` | Treats each MSMARCO passage as a natural chunk, attaching `query_id`, `passage_idx`, `source_lang`, `is_selected` | The corpus is *already* segmented by humans; re-splitting it discards that |

`python -m benchmarks.chunking_comparison` scores all four — plus the shipped ensemble —
on **recall@5** and **MRR** against the dataset's own `is_selected` relevance labels, and
reports chunk count and embedding cost next to the quality numbers so a strategy that
wins recall by tripling the index is visible as such. Results in
`benchmarks/chunking_comparison.json`.

The shipped index uses `metadata_aware + recursive` (`chunking/registry.py:DEFAULT_ENSEMBLE`).
Cross-strategy duplicates are deduplicated at build time — running several splitters over
the same corpus produces heavy overlap, and indexing the same text five times inflates the
index while making retrieval return five copies of one passage.

### Retrieval — hybrid, in-process

Dense search over FAISS `IndexFlatIP` fused with BM25 by **reciprocal rank fusion**.

*Exact, not approximate.* At ~15k vectors an exact search is one matmul in well under a
millisecond, so an ANN index buys no measurable time while costing recall — and that
recall loss would shift the score distribution the confidence guardrail thresholds
against, making the guardrail a function of the index's approximation error.

*In-process, not hosted.* Pinecone or Qdrant Cloud add 20–100ms of network round-trip
before any compute. The <200ms budget does not survive that.

*RRF, not a weighted score blend.* Cosine is bounded in [-1, 1]; BM25 is unbounded and
corpus-dependent. Any `α·dense + (1-α)·bm25` needs a normalisation that must be re-tuned
per corpus. RRF fuses on rank and needs none. The cost is that fused scores have no
absolute meaning — which is why `ScoredChunk` keeps `dense_score` separately, and the
confidence gate thresholds on raw cosine only.

**BM25 is hand-rolled rather than `rank_bm25`.** `BM25Okapi.get_scores` scans every
document for every query term — measured at 3.25ms per query on 3k chunks and scaling
linearly, which would be ~20ms at full corpus size. An inverted index touches only
documents containing a query term: **0.12ms, a 27× improvement.** The IDF form also
differs deliberately: `log(1 + (N-df+0.5)/(df+0.5))` rather than the textbook
`log((N-df+0.5)/(df+0.5))`, which goes *negative* for terms appearing in more than half
the corpus and lets a common word push documents down the ranking.

### Harness

`harness/orchestrator.py`. Not a function that calls an LLM:

- **Structured I/O everywhere.** Every stage returns `StageResult[T]` — a typed value or a
  typed `StageError`, plus its elapsed time. Nothing raises across a stage boundary.
- **Retries that discriminate.** `ErrorKind` separates `TRANSIENT`/`RATE_LIMITED` (retry)
  from `AUTH`/`CONFIG`/`VALIDATION` (don't — retrying a rejected API key spends latency to
  fail identically). Backoff uses **full jitter**, because the benchmark fires 50 queries
  in a loop and un-jittered retries re-collide on every attempt. Retries also honour a
  **deadline**, since three attempts against a hung socket is a correct retry policy and
  a failed demo simultaneously.
- **Per-stage error handling.** A failed STT call yields a structured error response with
  timings intact. A failed generator still returns the retrieved context.
- **Failure paths are timed too** — a stage that took four seconds to fail is exactly what
  a latency-graded pipeline needs to see.
- **The query is embedded once** and threaded through the guardrails and both retrievers.
  At ~104ms per embedding, embedding twice would blow the budget on its own.

### Guardrails — four, at three positions

| # | Guardrail | Position | Catches |
|---|---|---|---|
| 1 | `input_safety` | pre-retrieval | Harm-seeking phrasing, prompt injection, absurd transcript length |
| 2 | `language_mismatch` | pre-retrieval | Queries in a script the corpus isn't written in |
| 3 | `low_confidence` | post-retrieval, pre-generation | The corpus does not contain the answer |
| 4 | `groundedness` | post-generation | The model had good context and drifted anyway |

Every trip returns `{status: "declined", reason: "<guardrail>", ...}` with the score, the
threshold, and a human-readable explanation. A decline is a designed outcome, never a
crash and never a silent empty answer. Guardrails fail **open** if they error internally —
they are quality gates over a public Q&A demo, not a security boundary — except
`input_safety`, which fails closed, since withholding is its entire purpose.

**One guardrail was built, measured, and deleted.** Guardrail 2 was originally an
embedding-based off-topic detector scoring queries against the corpus centroid. Measured
on the real index, it was *inverted*:

| query | centroid sim | max-sim to corpus |
|---|---|---|
| "भारत की राजधानी क्या है?" (real) | **−0.035** | 0.440 |
| "मधुमेह के लक्षण क्या हैं?" (real) | 0.252 | 0.678 |
| "asdkjh qwe zxcvbn" (gibberish) | **0.184** | 0.381 |
| "aaaaa bbbbb ccccc" (gibberish) | 0.234 | 0.398 |

Real questions scored *below* gibberish. That is structural, not a bad threshold: a
centroid points in the "average passage" direction, so similarity to it measures how
generic a text is, not how on-topic — and a specific question is nearly orthogonal to the
mean. No threshold repairs an inverted signal.

The column that *does* separate them is max-similarity over the corpus, which is exactly
the top dense score guardrail 3 already thresholds. So semantic off-topic detection is
guardrail 3's job, done with the real score rather than a proxy, and a second semantic
gate would have been a less precise duplicate rather than an independent check.

What replaced it catches a failure guardrail 3 provably cannot. The English query *"what
is the capital of france"* scores **0.628** against the Hindi corpus — above the 0.42
confidence threshold — because a multilingual embedder maps it near Hindi passages about
countries, and MS MARCO genuinely contains them. Unguarded, it receives a confident answer
synthesised from passages that were never about it. The script check catches it in ~0.02ms
with no embedding at all. Romanised Hindi (*"bharat ki rajdhani kya hai"*) is exempted,
because Sarvam transcribes code-mixed speech and a Hinglish transcript is not a mismatch.

The confidence gate requires **two** signals, not one: an absolute top cosine score, and a
**margin** over the mean of the remaining chunks. A query the corpus answers produces a
peaked score distribution; one it cannot produces a flat one. A flat distribution at a
respectable absolute level is the signature of "this corpus is vaguely about the topic but
holds no answer" — which an absolute threshold alone waves straight through.

Thresholds are calibrated against real data, not guessed:
`python -m benchmarks.calibrate_thresholds` scores answerable dataset queries against
held-out, out-of-domain, and gibberish queries, and reports whether the populations
actually separate. If they overlap it says so rather than picking a decisive-looking
number — a confident threshold over overlapping distributions is a guardrail that fires at
random.

### Generation

Claude Haiku 4.5, `temperature=0`, with a prompt that forces grounding, authorises
refusal (`NO_ANSWER`), demands `[n]` citations, and pins the answer's language to the
question's. Retrieved passages are numbered and delimited and the model is told they are
reference material — corpus text is web-scraped and untrusted, and a passage containing
"ignore the above" must read as data.

A `NO_ANSWER` reply is **not** an error. It is the model using the refusal the prompt
authorises, and the orchestrator converts it into the same typed decline a guardrail
produces, so a user cannot tell which component noticed the corpus fell short.

Without `ANTHROPIC_API_KEY`, generation degrades to `extractive` mode — the top passage,
verbatim, labelled `mode: "extractive"` in the response and in the UI. A copied passage is
never presented as a generated answer.

---

## Benchmarks

```bash
python -m benchmarks.latency --n 50                    # P50/P70/P100, both numbers
python -m benchmarks.latency --n 50 --no-generation    # retrieval only
python -m benchmarks.chunking_comparison               # recall@5 + MRR per strategy
python -m benchmarks.calibrate_thresholds              # confidence-gate calibration
```

Three choices keep the latency benchmark honest:

- **Queries come from the dataset**, sampled from the corpus that was actually indexed —
  not a hand-picked list of questions known to work.
- **Warmup runs are excluded and reported.** The first ONNX inference pays graph
  initialisation; including it makes P100 a cold-start measurement, and dropping it
  silently would hide a real cost.
- **Declines are counted, not discarded.** A guardrail decline is a fast path — removing
  those runs would lower every percentile by deleting the cheapest requests. They stay in
  the sample, and the decline rate is reported next to the percentiles, so a suspiciously
  fast P50 is visible as a high decline rate rather than mistaken for speed.

P100 is the true maximum (nearest-rank), not an interpolated percentile that reports a
number no request ever took.

---

## Repo layout

```
V{X.Y}/
├── data/          MSMARCO-XI loading + normalisation
├── chunking/      four strategies + registry
├── retrieval/     embedder, FAISS index, BM25, hybrid retriever, build CLI
├── generation/    grounding prompt + Claude client + extractive fallback
├── guardrails/    input safety, language match, confidence gate, groundedness, suite
├── harness/       orchestrator, typed I/O, retry policy, assembly factory
├── benchmarks/    latency, chunking comparison, threshold calibration
├── demo/          FastAPI app + browser UI + CLI
└── tests/         147 tests
```

## Versioning

Every phase of work is a numbered snapshot under `master_repo/V{MAJOR}.{MINOR}/`, with
`master_repo/VERSION` pointing at the current one and `master_repo/CHANGELOG.md` carrying
a dated, file-by-file entry for each bump. Older version folders are frozen on bump and
never edited again. `bump.sh` performs the mechanical part.

## Deployment

`Dockerfile` builds the index **into the image** rather than at container start —
embedding 15k chunks takes ~13 minutes, and doing it on boot means serving 503s until a
platform health check kills the container. `render.yaml` is a ready blueprint; the
`starter` plan is specified deliberately, as 512MB is not enough for the index plus the
ONNX session.

## Honest limitations

- The groundedness check is lexical-overlap by default, not entailment. It catches
  invented entities and fabricated figures; it will not catch a wrong *inference* drawn
  from words that are present in the context. An opt-in LLM entailment path exists
  (`use_llm=True`) and costs a second round-trip.
- `input_safety` is pattern-based. It catches overt harm-seeking and injection phrasing,
  not obfuscated or adversarially-worded attacks. It is a first-line filter for a public
  demo, not a moderation system.
- Semantic off-topic rejection happens *after* retrieval, not before. Since retrieval is
  ~0.5ms of a ~105ms budget, the early-exit saving would have been negligible — and as
  measured above, the cheap pre-retrieval proxy did not work.
- Query embedding is ~104ms of the retrieval budget. Everything else together is under
  2ms. Making this pipeline meaningfully faster means a smaller embedding model, not
  micro-optimising search.

## References

- `DECISIONS.md` — every design call with its rationale, including the ones revised after
  measurement
- `../CHANGELOG.md` — full version history
