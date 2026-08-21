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

## D2 — Dataset subset: **`hi` (Hindi), `train` split, first 2000 examples**

**Decision:** `load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")`, capped at
`CORPUS_SIZE=2000` examples (≈8–10k passages after expansion).

**Rationale:** Must agree with D1 — a Hindi STT front-end retrieving over an English corpus
would put a translation hop in the hot path and break the latency budget. The cap keeps index
build time and memory sane for a hackathon-timeline demo box while still leaving a corpus
large enough that retrieval is a real problem (a 50-passage toy corpus makes every guardrail
threshold meaningless). `CORPUS_SIZE` is env-configurable — raise it if the deploy target has
the RAM.

---

## D3 — Embedding model: **`intfloat/multilingual-e5-small`** via `fastembed` (ONNX, CPU)

**Decision:** 384-dim multilingual E5-small, run through `fastembed`'s ONNX runtime rather
than `sentence-transformers` + PyTorch.

**Rationale:** Three requirements at once — must handle Devanagari (rules out `bge-small-en`),
must run on CPU in the low tens of milliseconds for a <200ms budget (rules out any 768-dim
model), and must not drag a ~2GB PyTorch install into a deployment container. E5 requires the
`query: ` / `passage: ` prefixes; the retrieval layer applies them, and they matter — omitting
them measurably degrades E5 recall.

---

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

**Decision:** (1) input safety/moderation and (2) off-topic detection run *pre-retrieval*,
(3) a retrieval-confidence gate runs *post-retrieval, pre-generation*, and (4) a groundedness
check runs *post-generation*.

**Rationale:** the task asks the system to know when *not* to answer, and each position catches
a failure the others structurally cannot. Off-topic detection pre-retrieval saves the entire
downstream cost on garbage input. The confidence gate is the one that prevents the classic RAG
failure — retrieving the least-bad chunk from a corpus that simply does not contain the answer,
then confidently generating from it. The groundedness check is the only one that can catch a
model that had good context and still drifted. Every trip returns a typed
`{status: "declined", reason: ...}` response, never a crash and never a silent empty answer.
