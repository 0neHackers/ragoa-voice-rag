<p align="center">
  <img src="V10.0/demo/assets/0neHackers.svg" alt="0neHackers" width="380">
</p>

<h1 align="center">Voice-Enabled RAG</h1>

<p align="center">
  Ask a question out loud in Hindi. Get an answer grounded in
  <a href="https://huggingface.co/datasets/ai4bharat/MSMARCO-XI"><code>ai4bharat/MSMARCO-XI</code></a>,
  or a refusal that tells you why.
</p>

<p align="center">
  <a href="https://ragoa-voice-rag.up.railway.app/"><b>Live app</b></a>
  &nbsp;·&nbsp;
  HH Goa 2026, Task 2
  &nbsp;·&nbsp;
  <code>#RAGInGoa</code>
</p>

---

```
 voice ──▶ Sarvam STT ──▶ [input safety] ──▶ [language match] ──▶ (optional en→hi) ──▶ embed
                                                                                        │
                                                                        dense (FAISS) ◀─┤
                                                                        BM25 (inverted) ◀┘
                                                                              │
                                                                          RRF fusion
                                                                              │
                                                                    [confidence gate]
                                                                              │
                              answer ◀── [groundedness] ◀── sarvam-105b-conversations
```

Speak, and the audio goes to Sarvam for transcription. The transcript passes two cheap
guardrails before anything expensive happens, gets embedded once, and is searched two ways
at the same time — dense vectors through FAISS and keywords through a hand-written BM25.
The two result lists are fused by rank. A confidence gate then decides whether the corpus
actually contains an answer; if it doesn't, the system says so instead of inventing one. If
it does, the passages go to an LLM with a prompt that forbids outside knowledge, and the
answer it returns gets checked for groundedness before anyone sees it.

Every stage is timed. Every failure is a typed value rather than an exception. Every refusal
carries a machine-readable reason, a score, and the threshold it missed.

---

## Numbers

Measured over 50 real MSMARCO-XI queries against the 15,449-chunk index, with generation
live. Raw output in [`V10.0/benchmarks/results_latest.json`](V10.0/benchmarks/results_latest.json).

| Retrieval pipeline — the number held to the 200 ms bar | |
|---|---|
| **P50** | **102.09 ms** |
| **P70** | **106.01 ms** |
| **P100** | **117.85 ms** |
| mean / min / stdev | 103.10 / 86.88 / 7.59 ms |
| inside budget | **100 % of queries** |

| Full pipeline — retrieval + LLM generation + groundedness | |
|---|---|
| **P50** | **2801.65 ms** |
| **P70** | **3229.50 ms** |
| **P100** | **6139.86 ms** |
| outcomes | 45/50 answered, 5 declined on groundedness |

Where the time goes. Embedding is the floor of the retrieval half; generation dominates
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

### Which number is held to 200 ms, and why

Two latencies are reported, and neither is hidden.

**Retrieval-pipeline latency** covers guardrails, query embedding, dense and lexical search,
and fusion. That is the number held to the task's bar, and it clears it on every query.

**Full-pipeline latency** adds the LLM call and the groundedness check, and lands around
2.8 seconds. Generation is 2686 ms of that at P50.

The split is stated up front because it is the one place this build interprets the task
rather than following it literally. Generation time is a third-party provider's
time-to-first-token; no client-side engineering moves it, so folding it into the headline
would report Sarvam's inference infrastructure as though it were this pipeline. Both numbers
appear in the benchmark output and on screen in the app for every request, so nothing
is concealed by the choice.

---

## What it does

**Speak, type, or upload.** The microphone path records raw PCM through the Web Audio API
and encodes 16 kHz mono WAV in the browser. Typed questions work too, and come back flagged
`voice_input: false` — a typed question is never presented as a spoken one.

**Translate from English.** An opt-in toggle sends the question through Sarvam's translation
endpoint first, so an English speaker can use a Hindi corpus. Roughly 860 ms. Both the
original and the translation are shown, so a translated query is never passed off as what
the user typed.

**Hear it back.** Both the question and the answer can be read aloud in Hindi. For a Hindi
demo that matters more than it sounds: someone who does not read Devanagari can still tell
the pipeline understood and answered.

**See the whole trace.** Every guardrail verdict with its score and threshold, per-stage
timings with percentage shares, the retrieved passages with both their dense and lexical
scores, and the raw JSON. A RAG demo that shows only its answer is indistinguishable from
one that made the answer up.

---

## Running it locally

```bash
cd V9.0
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
cp .env.example .env                                 # add SARVAM_API_KEY
python -m demo.app                                   # http://localhost:7860
```

`V9.0` rather than the current `V10.0` because `V9.0` is the snapshot carrying the committed
index, so there is nothing to build. The application code is identical between the two. To
work in `V10.0` instead, carry the index across first
(`cp -r V9.0/index_store/. V10.0/index_store/`), or rebuild it from scratch:

```bash
python -m retrieval.build_index --limit 1500 --out index_store
```

That takes about 13 minutes on a multi-core machine and considerably longer on two cores.

### The one key

| Variable | Powers | If it is missing |
|---|---|---|
| `SARVAM_API_KEY` | speech-to-text, generation, translation, text-to-speech | STT returns a typed `CONFIG` error, never a fake transcript. Generation drops to a labelled extractive mode. Translation and TTS are unavailable. Retrieval and every guardrail still work. |

There used to be a second provider here for generation. Consolidating onto Sarvam removed
two keys, two auth schemes, two rate limits, and two independent ways for the demo to fail
in front of a judge.

---

## Deployment

Live at **https://ragoa-voice-rag.up.railway.app/**, on Railway.

**The deployed snapshot is `V9.0`, not `V10.0`.** Railway's Root Directory is pinned to
`V9.0`, and that is deliberate: `V9.0` is the version that carries the committed
15,449-chunk index, so its image build is a file copy rather than a 48-minute re-embed on a
2-core builder. `V10.0` differs from it only in `verify_task.py` and documentation — no
application code changed between them — so nothing being served is stale. To move the
deployment forward, carry the index across and repoint the dashboard:

```bash
cp -r V9.0/index_store/. V10.0/index_store/
```

then set Root Directory to `V10.0`. There is no reason to do this urgently.

The image builds the index at **build time**, not on boot. Embedding 15k chunks takes about
13 minutes; doing that on container start means a cold deploy serves 503s for a quarter of an
hour and the platform health check kills the container long before it finishes. The 38 MB
index is committed, so the normal build path is a file copy and the rebuild branch only fires
on a checkout without one. The embedding model is pulled into the image too — otherwise it
downloads 225 MB during somebody's first request.

The container runs as UID 1000 rather than root, one uvicorn worker on purpose (each worker
would load its own index and ONNX session, and the pipeline is CPU-bound on embedding, so a
second competes for the same cores instead of adding throughput).

### The one thing that is easy to get wrong on Railway

The first build failed on the very first `COPY`:

```
failed to compute cache key: "/requirements.txt": not found
```

The Dockerfile was found correctly. The build context was not. Railway has **no build-context
field** — not in `railway.json`, not in `railway.toml`. `dockerContext` is not part of its
Config-as-Code schema at all, and an unrecognised key is silently ignored rather than
rejected, so adding one changes nothing and looks like it should have worked.

What actually scopes the build context is **Settings → Source → Root Directory**, which is
dashboard-only. Setting it to `V9.0` scopes the whole checkout, and `COPY requirements.txt .`
resolves. `dockerfilePath` then has to come *out* of `railway.json` — with the root already
scoped, a leftover `"V9.0/Dockerfile"` makes Railway look for `V9.0/V9.0/Dockerfile` and fail
a second, more confusing way.

Watch Paths is unrelated to any of this. It only gates whether a push triggers a redeploy.

So [`railway.json`](railway.json) stays deliberately small, and the setting that matters is
not in it:

```json
{
  "$schema": "https://railway.com/railway.schema.json",
  "build": { "builder": "DOCKERFILE" },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3,
    "numReplicas": 1
  }
}
```

`healthcheckTimeout` is 300 because loading 15k vectors and the ONNX session on a cold
container is not instant, and the default would restart-loop a container that was going to
become healthy.

### Why not a free tier

The app measures **687 MB resident** loaded and **743 MB** under query. Every 512 MB free
tier is therefore out, and this was tested rather than assumed: dropping the index and BM25
entirely still leaves ~546 MB, and ONNX allocator tuning moves 14 MB against a 170 MB gap.
The weight is the embedding model's 250,000 × 384 vocabulary table, and it is already int8.
Vocabulary pruning would plausibly get 60–70 % of it back, but it means re-exporting the
model, re-embedding the corpus, re-benchmarking, and re-calibrating both cosine-thresholded
guardrails — not a thing to start against a no-resubmission deadline.

`SARVAM_API_KEY` is set as a Railway service variable. It is not in the repo, and
`.gitignore` covers `.env` with only `.env.example` committed.

### Codespaces

[`.devcontainer/`](.devcontainer/devcontainer.json) reuses the same Dockerfile, so a
Codespace runs what a container host runs. It starts the app from `/app` rather than the
mounted workspace — the image keeps the index there, and starting from `/workspaces/<repo>`
finds no `index_store` and 503s on every request.

It opens a Cloudflare tunnel instead of using Codespaces' own port forwarding, because that
forwarding is not usable as a link you hand to someone: visibility reverts to private on its
own, non-browser clients get a flat 401 regardless of what the visibility says, and anonymous
visitors hit an "about to access a development port" interstitial GitHub gives no way to
disable.

---

## How each piece works

### The dataset

`load_dataset("ai4bharat/MSMARCO-XI", "hi")` does not work — the repository is not exposed
as named language configs but as a flat set of per-language parquet files. So
[`V10.0/data/loader.py`](V10.0/data/loader.py) addresses `validation/hinval.parquet` directly
and caches an extracted slice as JSONL.

The `validation` split is used rather than `train` because the Hindi train parquet is a
single 3.7 GB file against 462 MB for validation, with an identical schema. Downloading
3.7 GB to slice 1,500 examples off the front is a bad trade, and nothing here is trained —
this is a retrieval corpus, so the train/validation distinction carries no methodological
weight.

### Speech-to-text

Sarvam over ElevenLabs, because the dataset is AI4Bharat's Indic retrieval benchmark and the
demo is spoken by an India-based team. Sarvam is tuned for Indic phonetics and code-mixed
Hindi-English, which is exactly the audio it receives.

**Streaming is attempted first and does not work on this account.** The WebSocket endpoint
connects, authenticates, validates the model — it rejects `saarika:v2` and `saarika:flash`
as deprecated, so it is genuinely parsing the requests — accepts every audio frame without
complaint, and returns zero transcript frames. Tested at 100 ms, 200 ms and 500 ms chunk
cadences, with and without the model parameter, and against the `speech-to-text-translate`
endpoint. The batch REST endpoint transcribes the identical audio perfectly on the first
attempt.

So the shipped path is batch, labelled as batch. `Transcript.transport` reads `"batch"`, the
app displays it, and this README says it. The client probes streaming once per process and
latches the result, so the cost is a single probe rather than roughly 1.5 seconds wasted on
every request. `STT_TRANSPORT=streaming` re-exercises the path with no code change if Sarvam
enables it.

Two genuine bugs surfaced while establishing that, neither visible without a live key. The
stream terminator was wrong — `{"event": "stop"}`, which several streaming APIs document,
but Sarvam validates every frame as an audio request and answers
`Invalid request: 'audio' must not be None`. And the receive loop never terminated, because
Sarvam does not close the socket, so it blocked until a 34-second timeout after which the
batch fallback quietly rescued the request.

In the browser, WAV is encoded client-side rather than using `MediaRecorder`, which emits
WebM/Opus that libsndfile cannot demux. The alternative was installing ffmpeg on the deploy
host and transcoding inside the request. The client also rejects silence and sub-0.4-second
clips before spending an STT call, because a recorder that captured nothing is the most
common way a "voice-enabled" demo turns out silent.

### Chunking — four strategies, and they were measured against each other

| Strategy | What it does | Why it is here |
|---|---|---|
| `fixed_size` | 256 tokens, ~18 % overlap | Baseline. A cheap recall floor. |
| `semantic` | Splits at embedding-similarity breakpoints between sentences | Chunks do not get cut mid-idea |
| `recursive` | Paragraph → sentence → token fallback | Respects natural boundaries |
| `metadata_aware` | Each MSMARCO passage as a chunk, carrying `query_id`, `passage_idx`, `source_lang`, `is_selected` | The corpus is already segmented by humans; re-splitting throws that away |

`python -m benchmarks.chunking_comparison` scores all four against the dataset's own
`is_selected` relevance labels. Over 250 examples and 60 queries:

| strategy | chunks | recall@5 | MRR | embed time |
|---|---|---|---|---|
| **semantic** | 3,771 | **0.967** | **0.770** | 181.9 s |
| fixed_size | 2,557 | 0.950 | 0.752 | 117.0 s |
| recursive | 2,505 | 0.950 | 0.757 | 116.6 s |
| metadata_aware | 2,505 | 0.950 | 0.759 | 123.2 s |
| ensemble (shipped) | 2,530 | 0.950 | 0.752 | 124.4 s |
| ensemble (all four) | 5,110 | **0.917** | 0.743 | 218.9 s |

Two things fall out of that table, and neither was expected.

**Stacking every strategy makes retrieval worse.** The all-four ensemble scores 0.917, below
every individual strategy. More chunks means near-duplicate versions of the same passage
produced by different splitters, and those duplicates occupy top-5 slots the genuinely
relevant passage needed. "Chunking should be vast" is easy to read as "use more splitters",
and the data says otherwise.

**Semantic wins, and the shipped index does not use it.** That gap is real, so here is the
reason: semantic chunking embeds every sentence in the corpus in one batched pass, and at
1,500 examples that allocation died inside an ONNX MatMul. It is fine at the 250-example
benchmark scale. The pass is windowed now, but the shipped ensemble remains
`metadata_aware + recursive` and gives up 1.7 points of recall@5 for something that builds
reliably.

Cross-strategy duplicates are deduplicated at build time regardless.

### Retrieval

Dense search over a FAISS `IndexFlatIP`, fused with BM25 by reciprocal rank fusion.

**Exact, not approximate.** At roughly 15k vectors an exact search is a single matmul in well
under a millisecond, so an ANN index buys no measurable time and costs recall. Worse, that
recall loss would shift the score distribution the confidence guardrail thresholds against,
making a safety property depend on the index's approximation error.

**In-process, not hosted.** Pinecone or Qdrant Cloud add 20–100 ms of round-trip before any
compute. A 200 ms budget does not survive that.

**RRF rather than a weighted blend.** Cosine lives in [-1, 1]; BM25 is unbounded and
corpus-dependent. Any `α·dense + (1-α)·bm25` needs a normalisation re-tuned per corpus. RRF
fuses on rank and needs none. The cost is that fused scores mean nothing in absolute terms,
which is exactly why `ScoredChunk` keeps `dense_score` separately and the confidence gate
reads raw cosine.

**BM25 is hand-written rather than `rank_bm25`.** `BM25Okapi.get_scores` walks every document
for every query term — 3.25 ms per query on 3k chunks, scaling linearly, so roughly 20 ms at
full corpus size. An inverted index touches only documents containing a query term:
**0.12 ms, 27× faster.** The IDF form differs deliberately too,
`log(1 + (N-df+0.5)/(df+0.5))` instead of the textbook version, which goes negative for terms
appearing in more than half the corpus and lets a common word drag documents down the
ranking.

### The harness

Not a function that calls an LLM. [`V10.0/harness/orchestrator.py`](V10.0/harness/orchestrator.py):

- **Typed I/O everywhere.** Every stage returns `StageResult[T]` — a value or a typed
  `StageError`, plus elapsed time. Nothing raises across a stage boundary.
- **Retries that discriminate.** `ErrorKind` separates `TRANSIENT` and `RATE_LIMITED`
  (retry) from `AUTH`, `CONFIG` and `VALIDATION` (do not — retrying a rejected API key
  spends latency to fail identically). Backoff uses full jitter, because the benchmark fires
  50 queries in a tight loop and un-jittered retries collide on every attempt. There is a
  wall-clock deadline too, since three attempts against a hung socket is a correct retry
  policy and a dead demo simultaneously.
- **Failure paths are timed.** A stage that took four seconds to fail is precisely what a
  latency-graded pipeline needs to be able to see.
- **The query is embedded once** and threaded through the guardrails and both retrievers. At
  ~99 ms per embedding, embedding twice would blow the budget on its own.

### Guardrails

| # | Guardrail | Position | Catches |
|---|---|---|---|
| 1 | `input_safety` | pre-retrieval | Harm-seeking phrasing, prompt injection, absurd transcript length |
| 2 | `language_mismatch` | pre-retrieval | Queries in a script the corpus is not written in |
| 3 | `low_confidence` | post-retrieval, pre-generation | The corpus does not contain the answer |
| 4 | `groundedness` | post-generation | The model had good context and drifted anyway |

Every refusal returns `{status: "declined", reason: "<guardrail>", ...}` with the score, the
threshold, and a human-readable explanation. A decline is a designed outcome, never a crash
and never a silently empty answer.

They fail **open** if they break internally, because these are quality gates over a public
Q&A demo rather than a security boundary. The exception is `input_safety`, which fails
closed — withholding is the entire point of that one.

Guardrails 1 and 2 run **before** the query is embedded. That ordering matters more than it
sounds: embedding is ~99 ms of the ~102 ms retrieval P50, so a refusal after the embed saves
almost nothing. Measured, a blocked injection costs **0.01–0.03 ms** instead of ~99 ms.

#### One guardrail was built, measured, and deleted

Guardrail 2 began as an embedding-based off-topic detector scoring queries against the corpus
centroid. On the real index it was **inverted**:

Measured against the shipped 15,449-chunk index — reproduce with
[`V10.0/benchmarks/offtopic_centroid_probe.json`](V10.0/benchmarks/offtopic_centroid_probe.json):

| query | centroid similarity | max-similarity to corpus |
|---|---|---|
| "भारत की राजधानी क्या है?" (real) | **−0.017** | 0.585 |
| "मधुमेह के लक्षण क्या हैं?" (real) | 0.269 | 0.739 |
| "asdkjh qwe zxcvbn" (gibberish) | **0.239** | 0.524 |
| "aaaaa bbbbb ccccc" (gibberish) | 0.279 | 0.561 |

Both gibberish strings scored *above* a real question on centroid similarity. That is structural rather than a bad threshold: a
centroid points in the direction of the average passage, so similarity to it measures how
generic a text is, not how on-topic, and a specific question is nearly orthogonal to the
mean. No threshold repairs a signal pointing the wrong way.

The column that does separate them is max-similarity over the corpus — real questions at
0.585 and 0.739, gibberish at 0.524 and 0.561 — and that is exactly the top dense score
guardrail 3 already thresholds. So semantic off-topic detection belongs to guardrail 3,
working from the real score instead of a proxy. Note that both gibberish scores still sit
above guardrail 3's 0.45 cosine bar, which is precisely why that gate needed a second signal
as well.

What replaced it catches something guardrail 3 provably cannot. The English query *"what is
the capital of france"* scores **0.628** against the Hindi corpus, comfortably above the
confidence threshold, because a multilingual encoder maps it near Hindi passages about
countries and MS MARCO genuinely contains them. Unguarded, it receives a confident answer
built from passages that were never about it. A script check catches it in ~0.02 ms with no
embedding at all. Romanised Hindi is exempted, since Sarvam transcribes code-mixed speech
and Hinglish is not a mismatch.

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
questions**. The best dense-only threshold blocks 78.6 % of unanswerable queries by also
refusing **47 % of real ones**.

The lexical signal separates them outright. Every gibberish query scores exactly **0.00**
BM25, because none of its tokens exist anywhere in the corpus, while all 60 answerable
queries scored 8.5–43.3. Requiring BM25 > 0 blocks 7 of 42 unanswerable queries at **zero
false refusals**, and it is free, since hybrid retrieval already computes the number.

**What the gate still cannot do**, stated rather than buried: plausible Hindi questions the
corpus has no answer to score inside the answerable range on both signals. They are built
from ordinary Hindi words that really do appear across a web corpus, so no retrieval-score
threshold separates them without refusing real questions too. Catching those is the job of
the generator's authorised `NO_ANSWER` refusal and the groundedness check.

#### Groundedness combines two signals with OR

Cosine between the answer's embedding and the best-matching retrieved chunk, and the
fraction of the answer's content tokens present in the context. An answer passes if
**either** clears its bar.

Calibrated over 30 answers, each paired with a deliberately hallucinated counterpart
generated against mismatched context. `f_` columns are the faithful answer, `h_` the
hallucinated one; the raw rows are in
[`V10.0/benchmarks/semantic_groundedness_calibration.json`](V10.0/benchmarks/semantic_groundedness_calibration.json)
if you want to recompute this.

| rule | faithful answers refused | hallucinations caught |
|---|---|---|
| lexical ≥ 0.20 alone | 3.3 % | 86.7 % |
| lexical ≥ 0.25 alone | 10.0 % | 90.0 % |
| semantic ≥ 0.30 alone | 10.0 % | 96.7 % |
| semantic ≥ 0.30 **and** lexical ≥ 0.25 | 16.7 % | 100 % |
| **semantic ≥ 0.30 or lexical ≥ 0.25** — shipped | **3.3 %** | **86.7 %** |

Read that table honestly and OR is not free. It catches *less* than either signal does alone
at the same bars — a hallucination has two chances to slip through instead of one, and 86.7 %
is below semantic-alone's 96.7 %.

What it buys is the ability to run **both** bars stricter than either could sustain by itself.
Lexical alone at 0.25 refuses three faithful answers; semantic alone at 0.30 refuses three
different ones; the union refuses only the single answer that fails both. Their false
refusals barely overlap, because the two signals fail on opposite inputs — a heavy but
faithful paraphrase scores low lexically and high semantically, a copied-but-irrelevant span
does the reverse. So OR at the strict bars lands on the same operating point as lexical alone
at the loose bar (3.3 % / 86.7 %), while depending far less on either number being right for
any particular answer.

That is the trade this build chose, and the reason is the failure cost. On a public Q&A demo
a wrongly-refused correct answer is visible and cheap; a confidently-delivered hallucination
is invisible and expensive — but a demo that refuses one answer in six, which is what the AND
rule does, stops being a demo. AND is the right rule if you can tolerate that refusal rate.

Two caveats worth stating rather than hiding. **n = 30**, so one row moves any of these
figures by 3.3 points and the ordering between neighbouring rules is not statistically solid.
And the hallucinations here are synthetic — generated against mismatched context, which makes
them more clearly wrong than a model drifting slightly on correct context would be. The real
catch rate against subtle drift is lower than 86.7 %.

An unsupported figure is a hard block regardless of either score, because a fabricated
statistic is the highest-damage and most common hallucination in retrieval QA. Citation
markers are stripped before that scan — `[1]` and `[2]` were being read as fabricated
figures, which rejected every correctly-cited answer the generator produced.

### Generation

`sarvam-105b-conversations` at `temperature=0`, prompted to ground its answer, allowed to
refuse via `NO_ANSWER`, required to cite `[n]`, and pinned to the passages' language.

**The model choice was measured.** Sarvam offers two chat models and they are not
interchangeable here — `sarvam-105b` is a reasoning model that took **25.5 s and 865 tokens**
on a question the conversations variant answered in **2.3 s and 21**. On grounded extraction
the answer is already in the context; reasoning about it is pure latency. Worse, it spent the
entire token budget thinking and returned `content: null`.

Retrieved passages are numbered and delimited, and the model is told they are reference
material. Corpus text is scraped from the web and untrusted — a passage containing "ignore
the above" has to read as data, not instruction.

`NO_ANSWER` is not an error. It is the model using the refusal the prompt authorises, and the
orchestrator converts it into the same typed decline a guardrail produces, so a user cannot
tell which component noticed the corpus fell short.

---

## Layout

```
master_repo/
├── README.md             this file
├── CHANGELOG.md          dated, file-by-file history of every version bump
├── VERSION               the current version
├── bump.sh               the version-bump procedure
├── railway.json          deployment config
├── .devcontainer/        Codespaces setup
└── V0.0/ … V10.0/        one frozen snapshot per phase of work
                          V10.0 is current; V9.0 is what Railway serves

V10.0/
├── data/                 MSMARCO-XI loading and normalisation
├── chunking/             four strategies plus a registry
├── retrieval/            embedder, FAISS index, BM25, hybrid retriever, build CLI
├── stt/                  Sarvam speech-to-text, batch and streaming transports
├── generation/           grounding prompt, Sarvam chat client, extractive fallback
├── guardrails/           input safety, language match, confidence gate, groundedness
├── harness/              orchestrator, typed I/O, retry policy, assembly factory
├── translation/          opt-in English→Hindi query translation
├── tts/                  Hindi speech synthesis for question and answer
├── benchmarks/           latency, chunking comparison, threshold calibration
├── demo/                 FastAPI app, browser UI, CLI
├── tests/                177 tests
├── audio_samples/        Hindi WAV clips for exercising the voice path
└── verify_task.py        18 checks against the task requirements
```

Every phase of work is a numbered snapshot, with `VERSION` pointing at the current one and
`CHANGELOG.md` carrying a dated entry per bump. Old version folders are frozen and never
edited again.

---

## Verifying it yourself

```bash
cd V10.0
python -m pytest tests/ -q      # 177 tests
python verify_task.py           # 18 checks, including a live hit on the deployed service
python -m demo.cli --demo-suite # walks every guardrail path in order
```

`verify_task.py` reads artefacts rather than prose — benchmark JSON for the latency claims,
the filesystem for strategy and guardrail counts, git for the secrets check, a constructed
`PipelineResponse.declined()` for the structured-refusal claim, and an HTTP request to the
live deployment.

---

## What this does not do well

- The groundedness check cannot detect a confident wrong *inference* drawn entirely from
  words and ideas present in the context. That answer is, by both signals, grounded.
- `input_safety` is pattern-based. It catches overt harm-seeking and injection phrasing, not
  obfuscated or adversarially-worded attacks. It is a first-line filter, not a moderation
  system.
- The shipped index gives up 1.7 points of recall@5 to `semantic` chunking, for build
  reliability.
- Query embedding is ~99 ms of the ~102 ms retrieval P50. Everything else together is
  under 2 ms. Making this meaningfully faster means a smaller embedding model, not
  micro-optimising search.

---

<p align="center">
  <img src="V10.0/demo/assets/madeby0nehackers.svg" alt="Made by 0neHackers" width="300">
</p>
