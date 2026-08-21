# What's done, and what I need from you

**Deadline: 22 August 2026, 11:59 PM IST. One submission — no resubmissions.**

---

## Where the build stands

| Requirement | Status |
|---|---|
| Voice → speech-to-text | **Working and verified end to end against the live API.** Runs on batch, labelled as batch — streaming returns nothing on this account (details below). |
| 4 chunking strategies | Done, and benchmarked against each other on recall@5 / MRR |
| Vector retrieval | Done — FAISS in-process, 15,449 chunks, hybrid with BM25 |
| Under 200ms | **Met. P50 99.17ms, P70 102.39ms, P100 127.53ms** — every query inside budget |
| P50/P70/P100 across 30–50+ queries | Done, 50 real dataset queries |
| Harness | Done — typed I/O, selective retries, per-stage error handling |
| Guardrails | Done — 4 of them, 3 positions, every refusal structured |
| Tests | 167, all passing |
| Web UI | Rebuilt — boxy minimal, fluid 320px → 3840px, all four fonts + both team marks live |
| GitHub repo | **Done** — https://github.com/0neHackers/ragoa-voice-rag (public, 14 commits) |
| Live link | **Yours to do** |
| 2 videos + 6 social posts | **Yours to do** |

---

## 1. Keys — both in, both tested

Your keys are in `master_repo/V7.0/.env`, which is git-ignored. I verified nothing secret
reached the remote before pushing.

> **Rotate both after the hackathon.** They were pasted into a chat transcript, so treat them
> as exposed regardless of how carefully the repo is handling them.

**Sarvam works. Streaming doesn't, on this account.** I tested it properly and found two real
bugs in my own client along the way — details in `README.md`, but the summary is that the
WebSocket endpoint connects, authenticates, validates the model, accepts every audio frame,
and returns **zero transcript frames**, across every chunk cadence and endpoint I tried. The
batch REST endpoint transcribes the same audio perfectly on the first try.

So the pipeline runs on batch, labelled `transport: "batch"` everywhere it's reported. It's a
deviation from the plan and it's visible rather than buried. If Sarvam enables streaming on
your account, `STT_TRANSPORT=streaming` exercises that path with no code change.

Verified end to end by using Sarvam's own TTS to synthesise Hindi speech and feeding it back
through the whole pipeline:

```
spoken     : ईमानदारी या सच्चाई की परिभाषा क्या है
transcript : ईमानदारी या सच्चाई की परिभाषा क्या है?     (exact)
STT 1146ms · retrieval 121.8ms · full 2430ms · all 4 guardrails passed
```

**The Anthropic key has no credits.** The API returns `credit balance is too low`. That
arrives as an HTTP 400, which was taking the whole request down with it, so I made it degrade
to extractive mode with the reason stated in the response instead. The system answers, and it
says plainly that the answer is a retrieved passage rather than generated text.

**If you want generated answers in the demo video, you need to put credits on that account.**
Roughly $5 is far more than enough — Haiku at these token counts costs fractions of a cent per
query. Everything else works without it; you'd just be showing extractive answers.

Once credits are on, re-run this for a real end-to-end latency figure:

```bash
cd master_repo/V7.0
python -m benchmarks.latency --n 50
```

The current full-pipeline number is flagged `"full_pipeline_includes_llm": false` for exactly
this reason. Don't quote it as end-to-end latency until it's been re-run. The retrieval number
(99ms P50) is real and stands on its own.

---

## 2. Deployment — the one thing still open

Full detail in **[DEPLOY.md](DEPLOY.md)**.

- **Vercel can't run the backend.** Serverless functions cap at 250 MB; the embedding model
  alone is 240 MB and the whole payload is ~375 MB. Not a config problem — it doesn't fit.
- **Recommended: Render only, one URL.** FastAPI serves the page and the API together, so
  it's one link for the form and one thing to keep alive.
- If you want Vercel for the frontend anyway, set `window.__API_BASE__` in `demo/config.js`
  to the Render URL and deploy `demo/` as a static site. CORS is already open and the route
  is already wired.

What I need from you:

- A **Render** account on the **starter** plan or higher. The free tier's 512 MB will OOM,
  and sleeping instances make a terrible first impression on a cold judge link.
- Both keys pasted into the Render dashboard (never the repo).
- Render's root directory set to **`master_repo/V7.0`** — the Dockerfile lives one level
  down from the repo root.

Budget **15–20 minutes** for the first build. The Dockerfile bakes the index into the image
so the container is ready the moment it boots rather than serving 503s while it embeds.

---

## 3. Fonts and branding — done

All four faces load and are confirmed applied in the browser:

- **Cal Sans** — display headings (jsDelivr)
- **JetBrains Mono** — body, data, numbers (Google Fonts)
- **Disket Mono** — labels and metrics. Converted your TTFs to woff2, 82KB → 18KB each.
- **Noto Sans Devanagari** — every Hindi passage and answer. Not optional: neither JetBrains
  Mono nor Cal Sans has Devanagari glyphs, so without it the actual content falls back to
  whatever the OS supplies.

Both SVGs are in the footer — the `0neHackers` wordmark and the `made by` lockup — served
from `/assets`, rendering at their natural aspect ratios, with no page overflow down to
320px. The made-by mark ships as flat `#6B6B75`, which sits too close to the panel colour in
dark mode, so it's lifted optically with a CSS filter rather than by editing your file.

---

## 4. Videos

- **Video 1 — 90 seconds, process.** How the team worked, not the product. `CHANGELOG.md`
  and the commit history are good material — especially the guardrails that got built,
  measured, found broken, and rebuilt. That's a real engineering story and it's all dated.
- **Video 2 — the demo.** End to end, with **real microphone audio**.

Two things that bite people here:

- **Check your recorder is capturing the mic**, not just system audio. Most screen recorders
  don't by default. Record ten throwaway seconds and play it back before the real take.
- `python -m demo.cli --demo-suite` walks every guardrail path in order — answered,
  correctly refused, wrong language, gibberish, injection, harmful. Good for showing refusal
  behaviour on camera without improvising queries live.

---

## 5. Posts — all three of you, both platforms

Both videos → **Instagram and X** → posted by **Shanzal, Aditya and Kanishka individually**.
Not one shared team post. Six posts total.

Every single one needs **`#RAGInGoa`**.

At least one Instagram account has to be public — make them all public, or judges can't open
the links you submit.

---

## 6. The form

`https://forms.gle/MNvCjcv23Hn2Eeu58`

Fill **every** field, including the optional ones — member 2 and 3 names, handles, post
links, and the live link. The form marks the live link optional; the task doc treats it as
required. Trust the doc.

Confirmation field: type `#RAGInGoa` exactly.

Also make sure the **separate participation form** got submitted — different form, easy to
miss.

**Submit once.** A second submission auto-flags for rejection, so have every row filled and
verified before you click.

---

## Everything I need from you, in one list

Everything I can do without you is done. What's left:

1. **A Render account**, starter plan or above, with both keys set in its dashboard and the
   root directory set to `master_repo/V7.0`. This is the only thing standing between you and
   a live link.
2. **Credits on the Anthropic account** — only if you want *generated* rather than extractive
   answers in the demo. ~$5 is plenty. Everything works without it.
3. **The two videos**, with mic audio verified before the real take.
4. **Six posts** — three people, two platforms each, every one tagged `#RAGInGoa`, all
   accounts public.
5. **The submission form**, filled completely and submitted once.
6. **Confirmation the participation form is already in** — it's a separate form.
7. **Rotate both API keys** after judging closes.

Nothing here is blocked on me. Send me a Render URL and I'll verify it cold.

---

## Things worth saying if a judge asks

Three findings, all documented in `README.md` and `CHANGELOG.md` with the numbers:

1. **A guardrail was built, measured, and deleted.** Off-topic detection by corpus-centroid
   similarity scored real Hindi questions (−0.035, 0.252) *below* gibberish (0.184, 0.234).
   A centroid points at the average passage, so similarity to it measures how generic a text
   is, not how on-topic. No threshold fixes a signal pointing the wrong way.
2. **The confidence gate barely worked on cosine alone.** Gibberish scored a *higher* median
   cosine (0.774) than real questions (0.675). BM25 separates them completely — every
   gibberish query scores 0.00, all 60 answerable queries score 8.5–43.3 — so the gate now
   demands lexical support. Blocks 7 of 42 unanswerable queries at zero false refusals.
3. **BM25 is hand-written because `rank_bm25` was 27× too slow** — 3.25 ms → 0.12 ms per
   query.

And one limitation worth owning rather than hiding: plausible Hindi questions the corpus
simply doesn't answer land *inside* the answerable range on both signals. No retrieval-score
threshold separates those without also refusing real questions, which is exactly why the
generator's `NO_ANSWER` refusal and the groundedness check exist downstream.
