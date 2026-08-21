# Handoff — what's done, and what only you can do

Deadline: **22 August 2026, 11:59 PM IST. One submission, no resubmissions.**

---

## Status

| Requirement | State |
|---|---|
| Voice → STT | Built (Sarvam streaming + labelled batch fallback). **Untested against the live API — needs a key.** |
| Chunking, 4 strategies | Done, compared on recall@5 / MRR |
| Vector retrieval | Done — FAISS in-process, hybrid with BM25, 15,449 chunks |
| <200ms latency | **Met: P50 99.17ms, P70 102.39ms, P100 127.53ms**, 100% within budget |
| P50/P70/P100 over 30–50+ queries | Done, 50 real dataset queries |
| Harness | Done — typed I/O, selective retry, per-stage error handling |
| Guardrails | Done — 4, at 3 positions, each returning a structured decline |
| Tests | 163, passing |
| GitHub repo | **Not created — needs you** |
| Live link | **Not deployed — needs you** |
| 2 videos + social posts | **Needs you** |

---

## 1. API keys — the one real blocker

`.env` currently has empty placeholders. Nothing else is missing.

```bash
# master_repo/V5.0/.env
SARVAM_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

**Why this matters for the submission.** The task requires real voice input. The Sarvam
client is written against the documented streaming WebSocket protocol but **has never
executed against the live API** — I had no key. The frame-parsing is written tolerantly
(it accepts several response shapes Sarvam has shipped across model versions), but you
must verify it before recording the demo. Budget time for this; it is the one part of the
build that could still surprise you.

```bash
cd master_repo/V5.0
python -m stt.transcribe --mic --seconds 6 --save audio_samples/test.wav
```

Expected: your spoken Hindi comes back as a transcript, with `via streaming` in the output.
If it says `via batch`, the WebSocket handshake failed and fell back — the transcript is
still real, but check your network before blaming the code. If it errors, the message names
the cause; the frame shapes are in `stt/sarvam_client.py:_extract_transcript`.

Without `ANTHROPIC_API_KEY` the system still answers, but in **extractive** mode — it
returns the top retrieved passage verbatim, labelled as such in the UI and the API. That
is honest but it is not a great demo. Set the key.

Then re-run the benchmark to get a **real** full-pipeline number:

```bash
python -m benchmarks.latency --n 50
```

The current full-pipeline figure in `benchmarks/results_full_pipeline.json` is explicitly
marked `"full_pipeline_includes_llm": false` because generation ran extractive. Don't quote
it as an end-to-end latency until you've re-run it with a key.

---

## 2. Deploy the live link

The `Dockerfile` builds the index **into the image** (~13 min build, so the container is
ready on boot rather than serving 503s while it embeds).

Render, using the included blueprint:

1. Push to GitHub (below), then New → Blueprint on Render and point it at the repo.
2. `render.yaml` is already configured. Set `SARVAM_API_KEY` and `ANTHROPIC_API_KEY` in the
   dashboard — they are marked `sync: false` so they are never committed.
3. Keep the **starter** plan or larger. 512MB will OOM: the index plus the ONNX session
   needs ~1GB.
4. Test it cold in an incognito window before you put it in the form.

Note the root directory: the blueprint assumes the deploy context is `master_repo/V5.0/`.
Either set Render's root directory to that path, or copy that folder's contents to the repo
root before pushing.

**Leave the deployment up past the deadline** — the judging window is unknown.

---

## 3. GitHub

The repo is committed locally at `master_repo/` with real incremental history (V0.0 → V5.0,
one commit per phase). That history is worth keeping — Video 1 is about process, and this
shows it.

```bash
cd master_repo
git remote add origin https://github.com/<you>/<repo>.git
git branch -M main
git push -u origin main
```

Confirm `.env` is **not** in the repo (`.gitignore` covers it — verify with
`git ls-files | grep -i env`, which should show only `.env.example`).

---

## 4. Videos and posts — every member, both platforms

- **Video 1 (90s, process):** how the team worked. The `CHANGELOG.md` and commit history
  are good material — particularly the guardrails that were built, measured, found broken,
  and rebuilt.
- **Video 2 (demo):** the pipeline end to end with **real microphone audio**. Check your
  screen recorder is capturing the mic, not just system audio — record a throwaway clip
  first. `python -m demo.cli --demo-suite` walks every guardrail path if you want the
  refusal behaviour on camera without improvising queries.

Both videos → Instagram **and** X → by **Shanzal, Aditya, and Kanishka individually**, not
one shared post. Every post must carry **`#RAGInGoa`**. At least one Instagram account must
be public; make them all public or judges can't open the links.

---

## 5. The form

`https://forms.gle/MNvCjcv23Hn2Eeu58` — fill **every** field including the ones marked
optional (member 2/3 names, handles, post links, and the live link; the task doc treats the
live link as required even though the form does not). Confirmation field: type `#RAGInGoa`
exactly.

Also confirm the **separate participation form** was submitted — it is a different form
from the submission one.

**Submit once.** A second submission auto-flags for rejection.

---

## Talking points, if judges ask

Three findings worth knowing, all documented in `README.md` and `CHANGELOG.md`:

1. **The off-topic guardrail was measured to be inverted and deleted.** Centroid similarity
   scored real Hindi questions (−0.035, 0.252) *below* gibberish (0.184, 0.234). A centroid
   measures genericness, not topicality.
2. **The confidence gate barely worked on cosine alone.** Gibberish scored a *higher* median
   cosine (0.774) than real questions (0.675). BM25 separates them completely — all
   gibberish at 0.00, all 60 answerable queries at 8.5–43.3 — so the gate now requires
   lexical support, blocking 7/42 unanswerable queries at zero false refusals.
3. **BM25 was hand-rolled because `rank_bm25` was 27× too slow** (3.25ms → 0.12ms per query).

And one honest limitation to state rather than hide: plausible Hindi questions the corpus
simply doesn't answer sit *inside* the answerable range on both signals. No retrieval-score
threshold separates them without refusing real questions, which is why the generator's
`NO_ANSWER` refusal and the groundedness check exist downstream.
