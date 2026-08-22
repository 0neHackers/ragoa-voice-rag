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
| Tests | 177, all passing |
| Web UI | Boxy minimal, fluid 320→3840px, four fonts, both team marks placed |
| GitHub repo | **Done** — https://github.com/0neHackers/ragoa-voice-rag |
| Live link | **GitHub Codespaces** — free, secret set, see below |
| 2 videos + 6 social posts | **Yours to do** |

---

## 1. One key now — Anthropic is gone

Everything runs on `SARVAM_API_KEY`: speech-to-text, answer generation, translation, and
text-to-speech. Your key is in `master_repo/V9.0/.env`, which is git-ignored.

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

## 2. Deployment — GitHub Codespaces, free

**Your Codespace is created and `SARVAM_API_KEY` is set** as a Codespaces user secret
scoped to this repo. Nothing else is needed, and nothing is being paid for.

Codespaces gives every personal account 120 core-hours and 15GB a month free — far more
than a hackathon demo consumes — and it runs on GitHub's infrastructure, so unlike the
Cloudflare tunnel it survives your laptop being closed.

**Getting the public URL:**

1. Open the Codespace — **Code → Codespaces → RAGoa-Voice-RAG**
2. It was created before my devcontainer fix landed, so run **Rebuild Container** from the
   command palette (`Ctrl+Shift+P`). The earlier config started the app from the mounted
   workspace instead of `/app`, where the index is actually baked, and would have returned
   503 on every request.
3. Wait for the build. **The first one is 15–25 minutes** — it embeds all 15,449 chunks
   into the image so the container is ready the instant it boots.
4. Open the **PORTS** tab, find port **7860**, set Visibility to **Public**. A private port
   serves a GitHub login page to everyone else, which reads as a broken link.
5. Copy the `https://…-7860.app.github.dev` URL. That's your live link.

Check `<url>/health` returns `"ok": true` with `"index_size": 15449` before it goes in the
form.

**One caveat.** A Codespace stops after 30 minutes idle by default. It restarts in seconds
on the same URL, but a judge hitting it cold waits for that. Raise the timeout in
Settings → Codespaces, and re-check the URL shortly before you submit.

**Fallback:** `bash V9.0/serve_public.sh` starts the app plus a Cloudflare tunnel on your
machine and prints a public URL in about ten seconds — free, no account. `V9.0/LIVE_LINK.md`
explains why no 512MB free tier can host this (the ONNX session alone is 536MB) and what the
always-on free options are.

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

1. **Open the Codespace and Rebuild Container**, then set port 7860 to Public and copy the
   URL. That's your live link — free, already has the key.
2. **The two videos**, with mic audio verified before the real take.
3. **Six posts** — three people, two platforms each, every one tagged `#RAGInGoa`, all
   accounts public.
4. **The submission form**, filled completely and submitted once.
5. **Confirmation the participation form is already in** — it's a separate form.
6. **Rotate the Sarvam key** after judging closes. It went through a chat transcript, as did
   the Render and Hugging Face keys — revoke all three.

Nothing here is blocked on me. Send me the Codespace URL and I'll verify it cold.

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
