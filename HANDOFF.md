# What's done, and what I need from you

**Deadline: 22 August 2026, 11:59 PM IST. One submission — no resubmissions.**

---

## Where the build stands

| Requirement | Status |
|---|---|
| Voice → speech-to-text | **Working, verified against the live API.** Sarvam batch, labelled as batch — streaming returns nothing on this account. |
| 4 chunking strategies | Done, benchmarked against each other on recall@5 / MRR |
| Vector retrieval | Done — FAISS in-process, 15,449 chunks, hybrid with BM25 |
| Under 200ms | **Met. P50 102.09ms, P70 106.01ms, P100 117.85ms** — 100% of queries inside budget |
| P50/P70/P100 across 30–50+ queries | Done, 50 real dataset queries, live generation |
| Harness | Done — typed I/O, selective retries, per-stage error handling |
| Guardrails | Done — 4 of them, 3 positions, every refusal structured |
| Answer generation | **Live** — `sarvam-105b-conversations`, real Hindi answers |
| English→Hindi translation | Done, opt-in toggle, ~860ms |
| Speak aloud (question + answer) | Done, Sarvam TTS, Hindi |
| Tests | 171, all passing |
| Web UI | Boxy minimal, fluid 320→3840px, four fonts, both team marks placed |
| GitHub repo | **Done** — https://github.com/0neHackers/ragoa-voice-rag |
| Live link | **Needs your Render account** — see below |
| 2 videos + 6 social posts | **Yours to do** |

---

## 1. One key now — Anthropic is gone

Everything runs on `SARVAM_API_KEY`: speech-to-text, answer generation, translation, and
text-to-speech. Your key is in `master_repo/V8.0/.env`, which is git-ignored.

> **Rotate it once judging closes.** It went through a chat transcript, so treat it as
> exposed regardless of how carefully the repo handles it.

**Why Claude was removed.** The immediate reason is the one you gave — the key had no
credits, so generation ran permanently in extractive fallback. The better reason is
architectural: STT, translation and TTS were already Sarvam, so a second provider meant two
keys, two auth schemes, two rate limits, and two independent ways for the demo to break in
front of judges. One credential is one failure mode.

Generation now runs on `sarvam-105b-conversations`, and **you get real generated Hindi
answers** rather than extracted passages — which is a better demo than what Claude was
giving you, since Claude was giving you nothing.

Picking that model was worth measuring. Sarvam's other chat model, `sarvam-105b`, is a
reasoning model: 25.5 seconds and 865 tokens on a question the conversations variant
answered in 2.3 seconds and 21 tokens — and at a normal token budget it spent everything
thinking and returned an empty answer.

---

## 2. Deployment — the only thing left that I can't do

**I cannot deploy for you.** Render's API returns 401 without a token, and I have no
credentials for your account. If you want me to run it and verify end to end, generate a key
at **Render Dashboard → Account Settings → API Keys → Create API Key** and send it. Your
Render login is `0xshanzal@gmail.com`.

Otherwise it's five steps, and I've removed the fiddly one:

1. [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**
2. Connect `0neHackers/ragoa-voice-rag` — there's now a `render.yaml` **at the repo root**
   with `rootDir: V8.0` baked in, so you don't have to set a root directory by hand
3. Paste `SARVAM_API_KEY` when prompted (marked `sync: false`, never touches git)
4. **Apply.** First build is **15–25 minutes** — it embeds all 15,449 chunks into the image
   so the container is ready the moment it boots
5. Check `/health` returns `"ok": true` with `"index_size": 15449`

**Stay on `starter` or above.** Free's 512 MB will OOM, and sleeping instances make a bad
first impression on a cold judge link.

> **One thing I could not verify:** I tried to build the Docker image locally to de-risk
> this, and Docker Desktop wouldn't start on your machine, so the container build is
> **untested**. Every step inside it is individually verified — the dependency install, the
> index build, the app boot — but not the assembled image on Linux. Watch that first Render
> build rather than walking away from it. If it fails, the log will name the line.

Full detail, including why Vercel can't host the backend (250 MB function limit against a
~375 MB payload), is in **[DEPLOY.md](DEPLOY.md)**.

---

## 3. What's new in the app since you last saw it

- **Team wordmark** sits top-right in the masthead, baseline-aligned with "Voice RAG." and
  sized to its cap height. Verified by measurement: 0px from the container edge, 0px
  baseline delta.
- **Made-by mark** is centred in the footer, below enlarged meta text — 11.5px against
  22.1px, so it reads as a signature rather than a banner.
- **Translate toggle** — "Translate my question from English to Hindi first". Off by
  default, and deliberately so: auto-translating would silently disable a guardrail that
  exists because of a measured failure, and it would mangle romanised Hindi like
  "bharat ki rajdhani kya hai", which is already a Hindi question. The UI shows the original
  and the translation side by side, so a translated query is never passed off as what you
  typed.
- **Hear it** buttons on both the question and the answer. Hindi TTS, one shared player so
  pressing the second stops the first. Citation markers are stripped before synthesis —
  `[1][2]` reads as "one two" out loud otherwise.

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

1. **Deploy on Render** — starter plan or above, `SARVAM_API_KEY` in the dashboard. Either
   do the five steps above, or send me a Render API key and I'll do it and verify it.
2. **The two videos**, with mic audio verified before the real take.
3. **Six posts** — three people, two platforms each, every one tagged `#RAGInGoa`, all
   accounts public.
4. **The submission form**, filled completely and submitted once.
5. **Confirmation the participation form is already in** — it's a separate form.
6. **Rotate the Sarvam key** after judging closes.

Only item 1 is blocked on me having something I don't have. Send me a Render URL — or a
Render API key — and I'll verify the deployment cold.

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
