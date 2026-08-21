# DECISIONS.md — Team 0neHackers, HH Goa 2026 Task 2

Design calls that gate the rest of the build. Once a decision is recorded here it is
**not silently re-decided** — change this file deliberately, then re-run.

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

## D6 — Generation LLM: **Anthropic Claude Haiku 4.5**, with a typed extractive fallback

**Decision:** `claude-haiku-4-5-20251001` for grounded answer synthesis. If `ANTHROPIC_API_KEY`
is absent, generation degrades to a clearly-labeled **extractive** mode that returns the
highest-scoring retrieved span verbatim.

**Rationale:** Haiku is the fastest model in the Claude 5 family, which matters when generation
sits inside a latency-graded pipeline. The fallback exists so the retrieval pipeline, the
guardrails, and the benchmark remain runnable and honestly measurable without an LLM key — the
answer is *marked* `mode: "extractive"` in the structured response, never passed off as
generated.

---

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
