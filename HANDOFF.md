# What's done, and what I need from you

**Deadline: 22 August 2026, 11:59 PM IST. One submission — no resubmissions.**

---

## Where the build stands

| Requirement | Status |
|---|---|
| Voice → speech-to-text | Built (Sarvam streaming, batch fallback labelled). **Never run against the live API — no key.** |
| 4 chunking strategies | Done, and benchmarked against each other on recall@5 / MRR |
| Vector retrieval | Done — FAISS in-process, 15,449 chunks, hybrid with BM25 |
| Under 200ms | **Met. P50 99.17ms, P70 102.39ms, P100 127.53ms** — every query inside budget |
| P50/P70/P100 across 30–50+ queries | Done, 50 real dataset queries |
| Harness | Done — typed I/O, selective retries, per-stage error handling |
| Guardrails | Done — 4 of them, 3 positions, every refusal structured |
| Tests | 163, all passing |
| Web UI | Rebuilt — boxy minimal, fluid 320px → 3840px, verified at each end |
| GitHub repo | **Yours to do** |
| Live link | **Yours to do** |
| 2 videos + 6 social posts | **Yours to do** |

---

## 1. API keys — the only thing actually blocking the build

`.env` has empty placeholders right now. Nothing else is missing.

```bash
# master_repo/V5.0/.env
SARVAM_API_KEY=...
ANTHROPIC_API_KEY=...
```

**Please treat the Sarvam one as a real risk.** I wrote the client against the documented
streaming WebSocket protocol, but I've never been able to run it against the live API. The
frame parsing is deliberately tolerant — Sarvam has shipped more than one response shape
across model versions, so it accepts several — but *tolerant* isn't *tested*. Verify it
before you record anything:

```bash
cd master_repo/V5.0
python -m stt.transcribe --mic --seconds 6 --save audio_samples/test.wav
```

You want your spoken Hindi back as text, with `via streaming` in the output. If it says
`via batch`, the WebSocket handshake failed and it fell back — the transcript's still real,
but check your network before blaming the code. If it errors outright, the message names the
cause, and the frame shapes live in `stt/sarvam_client.py:_extract_transcript`.

Without `ANTHROPIC_API_KEY` the system still answers, but in **extractive** mode — it hands
back the top retrieved passage word for word, labelled as such in the UI. That's honest, but
it's a weak demo. Set the key.

Then re-run the benchmark to get a real end-to-end number:

```bash
python -m benchmarks.latency --n 50
```

The current full-pipeline figure is flagged `"full_pipeline_includes_llm": false` because
generation ran extractive. **Don't quote it as end-to-end latency until you've re-run it.**
The retrieval number (99ms P50) is real and stands on its own.

---

## 2. Deployment

Full detail in **[DEPLOY.md](DEPLOY.md)**. The short version:

- **Vercel can't run the backend.** Serverless functions cap at 250 MB; the model alone is
  240 MB and the whole payload is ~375 MB. Not a config problem — it doesn't fit.
- **Recommended: Render only, one URL.** FastAPI serves the page and the API together, so
  you get one link for the form and one thing to keep alive.
- If you do want Vercel for the frontend, set `window.__API_BASE__` in `demo/config.js` to
  the Render URL and deploy `demo/` as a static site. CORS is already open.

What I need from you either way:

- A **GitHub** account and repo (must be public)
- A **Render** account, on the **starter** plan or higher — free tier's 512 MB will OOM,
  and sleeping instances make for a bad first impression on a cold judge link
- Both API keys entered in the Render dashboard, not in the repo

Budget **15–20 minutes** for the first build. The Dockerfile bakes the index into the image
on purpose, so the container is ready the moment it boots.

---

## 3. Fonts — one optional file

Three of the four faces load from CDNs automatically. **Disket Mono doesn't exist on any
package CDN** — Fontfabric releases it directly and doesn't redistribute through npm or
Google Fonts.

If you have it, drop these two files in and they'll be picked up on the next reload:

```
master_repo/V5.0/demo/fonts/DisketMono-Regular.woff2
master_repo/V5.0/demo/fonts/DisketMono-Bold.woff2
```

Nothing breaks without them — the stack falls back to Space Mono, which has the same
squared-off terminal character. See `demo/fonts/README.md` for converting from `.ttf`.

I also added **Noto Sans Devanagari**, which you didn't ask for but the app can't do
without: neither JetBrains Mono nor Cal Sans has Devanagari glyphs, and every passage and
answer in this system is Hindi. Without it the actual content renders in whatever the OS
falls back to.

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

1. `SARVAM_API_KEY` — and please test the voice path with it before recording
2. `ANTHROPIC_API_KEY` — then re-run the benchmark for a real end-to-end number
3. GitHub account + a public repo to push to
4. Render account, starter plan or above
5. Both keys pasted into the Render dashboard
6. *(Optional)* the two Disket Mono `.woff2` files
7. Twitter/X and Instagram handles for all three of you
8. The two videos recorded, with mic audio verified
9. Six posts published, all tagged `#RAGInGoa`, accounts public
10. Confirmation the participation form is already in

Items 1 and 2 are the only ones that block me. Give me those and I can verify the voice path
end to end and produce a real full-pipeline latency figure. The rest need your accounts.

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
