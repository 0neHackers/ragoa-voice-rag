# DECISIONS.md — Team 0neHackers, HH Goa 2026 Task 2

Design calls that gate the rest of the build. Once a decision is recorded here it is
**not silently re-decided** — edit this file when you mean to change something, then re-run.

---

## D1 — STT provider: **Sarvam AI** (Saarika v2.5 / streaming)

**Decision:** Sarvam AI, using the **streaming (WebSocket) endpoint**, not the batch
transcription endpoint.

**Rationale:** The provided dataset (`ai4bharat/MSMARCO-XI`) is AI4Bharat's *Indic-language*
retrieval benchmark — its entire reason for existing is Indian-language RAG. Sarvam is
purpose-built for exactly that: Indic phonetics, Indian-accented speech, and code-mixed
Hindi-English utterances, which is how the demo will actually be spoken by an India-based
team. ElevenLabs Scribe has a marginally lower raw realtime floor, but it is not tuned for
Indic accents or code-mixing, so it would be the faster-but-less-accurate choice on precisely
the audio we will feed it. The streaming endpoint is mandatory: batch transcription is designed
for long files and its round-trip alone would consume most of the pipeline budget.

**Consequence:** requires `SARVAM_API_KEY`. There is no "typed text pretending to be a
transcript" path anywhere in this codebase — if the key is missing, the STT stage returns a
typed `StageError`, it does not silently substitute text.

**Verified against the live API on 2026-08-21, and the streaming half does not work on our
account.** Three things came out of that, all of which only a real key could surface:

1. The original terminator was wrong. This client ended a stream with `{"event": "stop"}`,
   which several other streaming APIs document. Sarvam validates *every* frame against its
   audio-request schema, so the sentinel came back as
   `Invalid request: 'audio' must not be None`. The stream ends by closing the socket.
2. The audio frame shape was right —
   `{"audio": {"data": <b64>, "encoding": "audio/wav", "sample_rate": 16000}}` — confirmed
   by probing six candidate shapes; `audio/x-raw` is rejected, `audio/wav` accepted.
3. **The WebSocket endpoint accepts audio and returns nothing.** It connects, authenticates,
   validates the model (it rejects `saarika:v2` and `saarika:flash` as deprecated), takes
   every frame without complaint, and emits zero transcript frames — tested at 100ms, 200ms
   and 500ms chunk cadences, with and without the model parameter, and against
   `speech-to-text-translate` too. The batch REST endpoint transcribes the identical audio
   perfectly.

So the shipped default is **batch, honestly labelled**, not streaming. The client still tries
streaming first, and the moment a streaming attempt opens cleanly and yields nothing it
latches `_streaming_known_dead` so the rest of the process goes straight to the endpoint that
works — one wasted probe per process instead of ~1.5s wasted per request. `STT_TRANSPORT`
forces `batch` or `streaming` explicitly.

This is a deviation from the plan and it is visible rather than hidden: `Transcript.transport`
reads `"batch"`, the demo shows it, and the README says so. Measured round trip on 2.1s of
audio: **~0.9–1.2s batch**. If Sarvam enables streaming on the account, `STT_TRANSPORT=streaming`
exercises the path with no code change.

---

## D2 — Dataset subset: **`hi` (Hindi), `validation` split, first 1500 examples**

**Decision:** read `validation/hinval.parquet` from the repo directly, capped at
`--limit 1500` examples, which expands to 30,759 raw chunks and **15,449 unique** after
cross-strategy deduplication.

**Two corrections to the original plan, both forced by what the dataset actually is.**
First, `load_dataset("ai4bharat/MSMARCO-XI", "hi", ...)` fails — the repo is not exposed
as named language configs but as a flat set of per-language parquet files, so the loader
addresses the file directly. Second, the split is `validation`, not `train`: the Hindi
train parquet is a single 3.7GB file, against 462MB for validation with an identical
schema. Downloading 3.7GB to slice 1,500 examples off the front is a bad trade on a
hackathon timeline, and since the corpus here is a *retrieval corpus* rather than
training data, the train/validation distinction carries no methodological weight — nothing
in this system is trained. `--split train` remains available.

**Rationale:** Must agree with D1 — a Hindi STT front-end retrieving over an English corpus
would put a translation hop in the hot path and break the latency budget. The cap keeps index
build time and memory sane for a hackathon-timeline demo box while still leaving a corpus
large enough that retrieval is a real problem (a 50-passage toy corpus makes every guardrail
threshold meaningless). Raise `--limit` if the deploy target has the RAM; the extracted slice
is cached locally as JSONL, so the parquet download is paid once.

---

## D3 — Embedding model: **`paraphrase-multilingual-MiniLM-L12-v2`** via `fastembed` (ONNX, CPU)

**Decision:** 384-dim multilingual MiniLM-L12, run through `fastembed`'s ONNX runtime rather
than `sentence-transformers` + PyTorch.

**Rationale:** Three requirements at once — must handle Devanagari (rules out `bge-small-en`),
must run on CPU in the low tens of milliseconds for a <200ms budget (rules out any 768-dim or
1024-dim model), and must not drag a ~2GB PyTorch install into a deployment container.

**Revised 2026-08-21:** originally specified `intfloat/multilingual-e5-small`. `fastembed` 0.8
does not carry that model in its ONNX registry — the only multilingual options it offers are
this MiniLM (384-dim, 0.22GB) and `multilingual-e5-large` (1024-dim, 2.24GB). E5-large is
disqualified on both latency and container size, so the MiniLM is the choice that keeps the
CPU/ONNX constraint intact. One consequence worth recording: E5 requires `query: ` / `passage: `
prefixes and this model does not, so the retrieval layer applies no prefixes.

## D4 — Vector store: **FAISS in-process (`IndexFlatIP`)**, with an exact-NumPy fallback

**Decision:** in-process FAISS, no hosted vector DB. If `faiss-cpu` has no wheel for the
runtime Python, the retrieval layer transparently falls back to an exact NumPy cosine index
implementing the identical math and the identical interface.

**Rationale:** the task's <200ms bar is not survivable with a network hop — Pinecone/Qdrant
Cloud cost 20–100ms of round-trip before any compute. At corpus sizes in the ~10k-vector range
an exact flat index is *also* the accuracy-correct choice: no ANN recall loss, and the search
is a single matmul in the low single-digit milliseconds. Vectors are L2-normalized so inner
product **is** cosine similarity, which lets guardrail thresholds be stated in cosine terms.

---

## D5 — Latency: two separately reported numbers; **<200ms targets the retrieval pipeline**

**Decision:** report `retrieval_ms` (guardrail → embed → dense+BM25 search → fuse → rerank) and
`full_ms` (that, plus LLM generation and the post-generation groundedness check) as two
clearly-labeled numbers. The **retrieval-pipeline number is the one held to <200ms**.

**Rationale:** stated openly rather than buried, because it is the one place this build
interprets the task rather than following it. Generation latency belongs to a third-party LLM
provider's TTFT, and no client-side engineering moves it; folding it into the headline number
would measure their infrastructure, not our pipeline. Both numbers are published in the README
and the benchmark output, so nothing is hidden by the choice.

---

## D6 — Generation LLM: **Sarvam `sarvam-105b-conversations`**, one provider for everything

**Decision:** answers are generated by Sarvam's chat API, on the same key that drives
speech-to-text, translation and text-to-speech. No second vendor.

**Revised — this used to be Claude Haiku 4.5.** Two reasons it changed, one practical and
one architectural.

The practical one: the Anthropic key available to this project had no credits, so every
generation call returned `credit balance is too low` and the pipeline ran permanently in
extractive fallback. A demo that never actually generates is not demonstrating generation.

The architectural one is the better argument, and it would hold even with credits. STT,
translation and TTS were already Sarvam. Keeping generation on a second provider meant two
keys, two auth schemes, two rate limits, two billing relationships and two independent ways
for the demo to break in front of judges. Consolidating onto one credential removes a whole
class of failure from a system that has to work on the first try.

**The model choice inside Sarvam was measured, not assumed.** Sarvam exposes two chat
models and they are not interchangeable here:

| model | latency | completion tokens | behaviour |
|---|---|---|---|
| `sarvam-105b` | 25.5s | 865 | reasoning model — emits `reasoning_content` |
| `sarvam-105b-conversations` | 2.3s | 21 | answers directly |

`sarvam-105b` is a reasoning model, and on grounded extraction — where the answer is sitting
in the context and the job is to state it — that reasoning is pure latency. Worse, it spends
the token budget thinking: at `max_tokens=160` it returned `finish_reason: "length"` with
`content: null` and the whole budget in `reasoning_content`. `reasoning_effort: "low"` did
not fix it. So the conversations variant is the default, and `_extract_content` still
inspects `reasoning_content` so a misconfiguration produces a clear error rather than a
mysterious empty answer.

**Two things had to be fixed once a real model started writing real prose**, neither of
which showed up while generation was extractive:

1. The model answered in **English** for a Hindi question over Hindi passages, despite the
   prompt asking it to match the question's language. That breaks a Hindi voice demo and
   it zeroes the groundedness check, which compares answer tokens against Hindi context.
   The instruction is now stated last, as an explicit override, and keyed to the *passages*
   rather than the question.
2. The groundedness threshold was calibrated against extractive answers — literal copies of
   the context — so it sat at 0.45. Against genuine paraphrase it refused **6 of 18 faithful
   answers**. Re-measured and moved to 0.30. See D9.

Without a key, generation still degrades to labelled `extractive` mode, so the retrieval
pipeline, the guardrails and the benchmark all stay runnable and honestly measurable.

---

## D6b — Translation and text-to-speech: Sarvam, same key

**Decision:** English→Hindi query translation (`/translate`) and Hindi speech synthesis
(`bulbul` TTS) are both available, both on the same Sarvam key.

**Rationale.** Translation exists because the corpus is Hindi and the `language_mismatch`
guardrail correctly refuses Latin-script queries — measured: "what is the capital of france"
scores 0.628 against this corpus, above the confidence threshold, because a multilingual
encoder maps it near Hindi passages about countries. Translating first makes an English
speaker's question genuinely Hindi, so retrieval compares like with like instead of leaning
on cross-lingual alignment that has already been shown to mislead.

**It is opt-in, and that is the important part.** Auto-translating every Latin-script query
would silently disable a guardrail that exists because of a real measured failure, and it
would mangle romanised Hindi — "bharat ki rajdhani kya hai" is already a Hindi question, and
translating it *as English* produces nonsense. The user ticking the box is the signal that
the input really is English. Both strings are kept in the response so the UI can show what
was asked and what was actually searched for.

TTS closes the loop: you speak a question and the system speaks its answer back. For a Hindi
demo that matters more than it sounds — a viewer who does not read Devanagari can still tell
the pipeline understood and answered. `anushka` is the default speaker, verified against the
live API, which rejects unrecognised speaker names outright.

## D7 — Guardrails: four, at three distinct pipeline positions

**Decision:** (1) input safety/moderation and (2) corpus-language match run *pre-retrieval*,
(3) a retrieval-confidence gate runs *post-retrieval, pre-generation*, and (4) a groundedness
check runs *post-generation*.

**Rationale:** the task asks the system to know when *not* to answer, and each position catches
a failure the others structurally cannot. The confidence gate prevents the classic RAG failure —
retrieving the least-bad chunk from a corpus that does not contain the answer, then confidently
generating from it. The groundedness check is the only one that can catch a model that had good
context and still drifted. Every trip returns a typed `{status: "declined", reason: ...}`
response, never a crash and never a silent empty answer.

**Revised: guardrail 2 was an embedding-based off-topic detector; it was removed after
measurement.** It scored the query against the corpus centroid. On the 3,039-chunk index it was
*inverted* — real Hindi questions scored -0.035 and 0.252, gibberish scored 0.184 and 0.234, so
genuine questions ranked below nonsense. That is structural rather than a tuning problem: a
centroid points in the "average passage" direction, so similarity to it measures how generic a
text is, not how on-topic, and a specific question is nearly orthogonal to the mean.

The signal that *does* separate them is max-similarity over the corpus — which is the top dense
score the confidence gate already thresholds. So semantic off-topic detection belongs to
`low_confidence`, where it runs on the real score instead of a proxy, and a second semantic gate
would have been a less precise duplicate rather than an independent check.

Guardrail 2 is now a **corpus-language match**, which catches a failure the confidence gate
provably cannot: the English query "what is the capital of france" scores **0.628** against the
Hindi corpus — above the 0.42 confidence threshold — because a multilingual embedder maps it
near Hindi passages about countries and MS MARCO genuinely contains them. Without this gate that
query receives a confident answer synthesised from passages that were never about it. Romanised
Hindi ("bharat ki rajdhani kya hai") is exempted, since Sarvam transcribes code-mixed speech and
a Hinglish transcript is not a language mismatch.

---

## D8 — Query embedding is the latency floor; the budget is built around it

**Decision:** accept ~100ms per query embedding as the dominant cost of the retrieval
pipeline, warm the model at startup, and reuse the single query vector across the off-topic
guardrail and dense retrieval rather than embedding the query more than once.

**Measured, on the dev box (16 CPUs, onnxruntime 1.29, CPU execution provider):**

| model | layers | vocab | single-query embed |
|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` (int8) | 12 | 250k | **~100 ms** |
| `all-MiniLM-L6-v2` (int8, English-only) — control | 6 | 30k | ~16 ms |

Thread count makes almost no difference (99–108ms across `intra_op_num_threads` 1/2/4/8), so
this is the model's cost on this hardware, not a misconfiguration. The 6x gap against the
control is explained by 2x the layers and an 8x larger embedding matrix.

**Rationale:** the honest options were (a) keep the multilingual model and spend half the
budget on it, or (b) switch to the 6-layer English model and retrieve over the corpus's
`English_passages` column. (b) is faster but incoherent: the STT front-end is Hindi (D1), so
an English index would need a translation hop in the hot path — which would cost far more than
the 84ms saved and would make the pipeline's own language handling a fiction. (a) still lands
the retrieval pipeline under the 200ms bar with headroom, so it wins on both honesty and
architecture.

**Consequences:**
- `Embedder.warmup()` is called at startup by the demo and the benchmark. The first ONNX
  `run()` pays graph init and weight paging; without warm-up the reported P100 would measure a
  one-off cold start rather than steady-state retrieval.
- The fastembed ONNX cache is pinned to `model_cache/` under the repo rather than the OS temp
  directory, so a deployed instance cannot lose the model on a restart and re-download 220MB
  on a user's first query.
- Semantic chunking's cost is now explicit: ~4.8s per 100 examples at index build time. It is
  a build-time cost only and never touches the query path.

---

## D9 — Groundedness threshold, re-measured against generated prose

> **Revised — see D9a below.** This entry records the lexical-only calibration and the
> reasoning that led to a 0.20 floor. The shipped guardrail is no longer lexical-only: it
> is `semantic >= 0.30 OR lexical >= 0.25`. The measurements here still stand for what they
> measured; D9a explains what replaced them and why.

**Decision (superseded):** the lexical-overlap floor in the groundedness guardrail is 0.20,
down from 0.45.

**Rationale.** 0.45 was calibrated when generation was extractive — answers that were
literal copies of the retrieved context, so overlap was trivially high and the threshold
cost nothing. The moment a real model started producing Hindi paraphrase, it became a
false-refusal machine. Measured over 40 faithful generated answers:

```
min 0.091 · p10 0.214 · median 0.633

threshold   faithful answers refused
0.45        37.5%
0.40        25.0%
0.35        20.0%
0.30        17.5%
0.25        12.5%
0.20         7.5%   <- shipped
0.15         2.5%
```

0.45 was rejecting well over a third of correct answers. An intermediate 0.30 still cost
17.5%, which showed up in a 50-query benchmark as 8 declines. 0.20 keeps a genuine floor —
an answer sharing almost no vocabulary with its context still trips it — at roughly 1 false
refusal in 13.

**The residual 7.5% is real and not hidden.** A faithful answer scoring 0.09 is genuinely
indistinguishable from a hallucination by lexical overlap alone, and dropping further would
stop it measuring anything. That is the honest limit of a cheap metric against paraphrase.

This check is therefore one layer of three, and not the load-bearing one. The
unsupported-number check is the sharp instrument — fabricated figures are the
highest-damage, most common hallucination in retrieval QA — and the retrieval-confidence
gate upstream prevents most ungrounded answers from ever being generated.

**Fixed while measuring this:** the unsupported-number check was reading `[1]`, `[2]`
citation markers as numeric claims and flagging every correctly-cited answer as containing
fabricated figures — rejecting exactly the behaviour the prompt asks for. Citations are now
stripped before the number scan.

---

## D9a — Groundedness is two signals combined with OR

**Decision:** an answer passes if `semantic >= 0.30` **or** `lexical >= 0.25`, where
semantic is the cosine between the answer's embedding and the best-matching single retrieved
chunk, and lexical is the fraction of the answer's content tokens present in the context.
Shipped in `guardrails/groundedness.py` as `DEFAULT_MIN_SEMANTIC` and `DEFAULT_MIN_OVERLAP`.

**What forced the change.** D9's lexical-only floor was reported as always declining once
real paraphrase was being generated. Two distinct causes: a stale server process still
running the old 0.30 threshold, and — the real one — lexical overlap being a poor proxy for
faithfulness against a model that genuinely paraphrases rather than extracts. D9 already
said as much in its own residual-7.5% paragraph; the fix was to stop asking one cheap metric
to carry the whole check.

**Calibration.** 30 answers, each paired with a hallucinated counterpart generated against
mismatched context. Raw rows in `benchmarks/semantic_groundedness_calibration.json`.

```
rule                                      faithful refused    hallucinations caught
lexical >= 0.20 alone                           3.3%                  86.7%
lexical >= 0.25 alone                          10.0%                  90.0%
semantic >= 0.30 alone                         10.0%                  96.7%
semantic >= 0.30 AND lexical >= 0.25           16.7%                 100.0%
semantic >= 0.30 OR  lexical >= 0.25   <-ship   3.3%                  86.7%
```

**Read honestly, OR is not free.** It catches less than either signal alone at the same bars
— a hallucination gets two chances to pass instead of one, and 86.7% is below semantic-alone's
96.7%.

What it buys is running **both** bars stricter than either could sustain alone. Lexical at
0.25 alone refuses three faithful answers; semantic at 0.30 alone refuses three *different*
ones; the union refuses only the one answer that fails both (f_sem 0.076, f_lex 0.091). The
two signals fail on opposite inputs — heavy-but-faithful paraphrase scores low lexically and
high semantically, a copied-but-irrelevant span does the reverse — so their false refusals
barely overlap. OR at the strict bars therefore lands on the same operating point as lexical
alone at the loose bar, while depending far less on either number being right for any one
answer.

AND is the correct rule if a 1-in-6 refusal rate is acceptable. On a public demo it is not.

**Two caveats, stated rather than buried.** n=30, so one row moves any figure by 3.3 points
and the ordering between adjacent rules is not statistically solid. And the hallucinations
are synthetic, generated against mismatched context, which makes them more obviously wrong
than a model drifting slightly on correct context — so the real catch rate against subtle
drift is below 86.7%.

**Two implementation details that mattered more than the thresholds.** The semantic score is
the best cosine against any *single* chunk, not against the concatenated context — comparing
against the concatenation lifted the faithful floor from 0.005 to 0.076 and made the two
populations separable at all. And the unsupported-number check strips `[n]` citation markers
before scanning, because it was reading them as fabricated figures and rejecting exactly the
citation behaviour the prompt asks for.
