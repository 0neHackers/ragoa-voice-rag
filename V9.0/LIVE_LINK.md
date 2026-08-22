# Getting a live link without paying for hosting

## Why this needed its own file

The app needs about **700MB of RAM**, and the reason is not the corpus — it's the
embedding model. Measured breakdown:

| component | resident memory |
|---|---|
| Python + numpy + faiss + FastAPI | 63 MB |
| **ONNX embedding session** | **536 MB** |
| index (15,449 vectors + chunk text) | 86 MB |
| BM25 inverted index | 31 MB |
| **total** | **~717 MB** |

That matters because it kills the obvious workarounds, both of which were measured rather
than assumed.

**Shrinking the corpus doesn't help.** The index is 86MB and BM25 is 31MB of a 200MB
overshoot. Delete both entirely and the process still sits at ~546MB — a 512MB tier cannot
host this *with no corpus at all*.

**Nor does tuning the ONNX allocator.** Every session option that plausibly affects memory,
measured:

| configuration | session | latency |
|---|---|---|
| default | 497 MB | 103.7 ms |
| `enable_cpu_mem_arena = False` | 484 MB | 106.3 ms |
| device allocator for initializers | 484 MB | 109.6 ms |
| all combined | 485 MB | 104.6 ms |
| + `ORT_DISABLE_ALL` graph optimisation | 483 MB | 106.9 ms |

14MB, against a 170MB gap. The memory is resident model weights, not allocator overhead.

**Why the model is large, and what would actually shrink it.** It is already int8-quantised
— that is what the `-onnx-Q` suffix means — and it is 225MB on disk anyway, because the bulk
of a multilingual encoder is not the transformer but the vocabulary: ~250,000 tokens x 384
dimensions is roughly 96M parameters in the embedding table alone.

So the technique that would work here is **vocabulary pruning**: keep only the tokens that
occur in this Hindi corpus and in plausible queries, slice the embedding matrix down, and
re-export. A 60-70% reduction is realistic and would bring the whole process under 512MB.

It is deliberately not done here. It means rebuilding the tokenizer, re-exporting the ONNX
graph, re-embedding all 15,449 chunks, re-running the latency and chunking benchmarks, and
recalibrating every guardrail threshold against a changed score distribution — because the
confidence gate and the semantic groundedness check both threshold on cosine values that a
different embedding space would move. That is a day of work and revalidation, not an
afternoon, and it trades a working system for a smaller one.

(Note on generative-model quantisation work such as Gemma QAT: it targets decoder LLMs and
does not apply to a sentence-embedding model that is already quantised.)

So: **free 512MB tiers (Render free, Koyeb free) cannot host this.** That's a measured
conclusion, not an assumption.

---

## What's running now: Cloudflare Tunnel

```bash
./serve_public.sh
```

That starts the app, waits for the pipeline to finish loading, opens the tunnel, and prints
the URL. It fetches `cloudflared` on first run (~55MB) — no account, no card, no signup.

It prints a public `https://<random>.trycloudflare.com` URL in about ten seconds. Real
HTTPS, so the microphone works — browsers only allow `getUserMedia` in a secure context.

**Be honest with yourself about the tradeoff.** The tunnel runs on your machine, so:

- Your PC has to stay awake and online for the whole judging window
- The URL changes every time you restart the tunnel
- If your connection drops, so does the link

For a hackathon with an unknown judging window that is a real risk. It is a genuine,
working, free live link — it is not an always-on deployment.

Mitigations if you use it:
- Disable sleep: **Settings → System → Power → Screen and sleep → Never**
- Re-run the tunnel and re-check the URL right before you submit the form
- Keep the terminal window open; closing it kills the tunnel

---

## Railway free trial — 1 GB, no card *(best free option found)*

Checked against Railway's current pricing rather than remembered, because these tiers move:

| plan | RAM | card |
|---|---|---|
| **Free Trial** | **1 GB**, 2 vCPU, 4 GB image | **no card required**, $5 one-time credit |
| Free | 0.5 GB | — |
| Hobby | 48 GB | $5/mo |

1 GB clears the measured ~717 MB, and the ~1 GB image clears the 4 GB cap. No sleep, no
interstitial, a normal `*.up.railway.app` URL, and it runs on Railway's machines rather than
yours.

**The honest limit:** the $5 is a one-time grant, not monthly. A 1 GB service running
continuously costs roughly $10/month on Railway's rates, so expect the credit to cover
somewhere around two weeks. That is comfortably longer than a judging window, and it is not
permanent hosting.

```
railway.json at the repo root already points at V9.0/Dockerfile and sets the /health check.
1. railway.com -> New Project -> Deploy from GitHub -> 0neHackers/ragoa-voice-rag
2. Variables -> add SARVAM_API_KEY
3. Settings -> Networking -> Generate Domain
```

Build takes roughly 5 minutes: the index is committed, so it is copied rather than rebuilt,
and the embedding model is baked in during the build rather than downloaded on the first
request.

---

## Free options that survive your PC being off

### GitHub Codespaces — **no card, recommended**

Free tier gives **120 core-hours a month** (60 hours on the 2-core machine) and 15GB
storage, on a personal account, **with no payment method on file**. The machine is
2 cpu / 8GB — ten times the headroom this needs — and it runs on GitHub's infrastructure,
so your laptop can be closed.

`.devcontainer/devcontainer.json` is already committed and does the work: it builds the
production Dockerfile, bakes the index, forwards port 7860, and marks that port **public**
so a judge who isn't signed into GitHub can open it.

**This is the one step I could not do for you.** Creating a Codespace needs the
`codespace` OAuth scope, which only an interactive browser grant can give, and the CLI on
this machine is signed in as `shanzalfiroz` while the repo belongs to `0neHackers`. Two
clicks in your browser:

1. Add the secret first: **github.com/settings/codespaces → New secret** →
   name `SARVAM_API_KEY`, value your key, repository access `0neHackers/ragoa-voice-rag`
2. Open **github.com/0neHackers/ragoa-voice-rag** → green **Code** button → **Codespaces**
   tab → **Create codespace on main**
3. First creation takes **15–25 minutes** (it embeds all 15,449 chunks into the image).
   Later starts are instant.
4. When it finishes, the **Ports** tab shows 7860. Confirm visibility is **Public**, then
   copy the `https://…app.github.dev` URL — that's your live link.

Watch the budget: 60 hours on 2-core is about two and a half days of continuous running.
The devcontainer sets a 240-minute idle timeout, so it suspends itself rather than burning
hours while nobody is looking. Restarting it is instant and gives the same URL.

### Kaggle Notebooks / Google Colab — no card, session-limited

Both give far more RAM than this needs, free, with no card. Both also cap sessions
(Kaggle ~12h, Colab shorter and idle-sensitive) and neither exposes a public port on its
own, so you'd still run `cloudflared` inside the notebook. Workable as a fallback; more
moving parts than Codespaces for the same result.

### Oracle Cloud Always Free / Google Cloud Run — card for verification

Both are genuinely free within their allowances and both want a card on file purely to
verify identity. Oracle's Always Free tier is the most generous thing on this list — 4 ARM
cores and 24GB RAM, permanently — and Cloud Run's free tier easily covers a demo's traffic.
Neither is an option without a card, which is why Codespaces is the recommendation above.

---

## Paid, for completeness

| host | cost | note |
|---|---|---|
| Hugging Face PRO | $9/mo | Docker Spaces now require PRO; free CPU is static-only |
| Render Standard | $25/mo | `starter` is 512MB — same as free — so it does **not** work |
