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

That matters because it kills the obvious workaround. Shrinking the corpus to fit a free
512MB tier would save at most ~90MB of the 200MB overshoot, and the model still wouldn't
fit. Disabling onnxruntime's CPU memory arena (`enable_cpu_mem_arena=False`) saves another
40MB and slightly *improves* latency (93.8ms vs 104ms median), but 496MB for the embedder
alone is still over the line.

Nor can we swap in a smaller model. `fastembed`'s only lighter option is
`all-MiniLM-L6-v2`, which is English-only — it has no Devanagari, and the entire corpus and
every answer here is Hindi.

So: **free 512MB tiers (Render free, Koyeb free) cannot host this.** That's a measured
conclusion, not an assumption.

---

## What's running now: Cloudflare Tunnel

```bash
# 1. start the app
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8600

# 2. expose it (downloads once, no account, no card, no signup)
cloudflared tunnel --url http://127.0.0.1:8600
```

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

## Free options that survive your PC being off

Both require a card **for identity verification only** — neither charges you within the
free allowance. If you have access to a card, either beats the tunnel.

### Oracle Cloud Always Free *(most headroom)*

Genuinely free forever, not a trial: 4 ARM cores and **24GB RAM**. Enormously more than
this needs.

1. Sign up at [oracle.com/cloud/free](https://www.oracle.com/cloud/free/)
2. Create an **Ampere A1** compute instance (ARM), Ubuntu 22.04, 1–2 cores / 6–12GB
3. Open port 80/443 in the security list
4. `docker build` and `docker run` this repo's Dockerfile on it
5. Point Cloudflare Tunnel (or Caddy) at it for HTTPS

Cost: **₹0**. Setup: 30–45 minutes. ARM is fine — onnxruntime and faiss-cpu both ship
arm64 wheels.

### Google Cloud Run *(least setup)*

Free tier covers 180,000 vCPU-seconds and 360,000 GiB-seconds per month, and it scales to
zero, so a demo that gets a few hundred requests costs nothing.

```bash
gcloud run deploy ragoa-voice-rag \
  --source . --region asia-south1 \
  --memory 2Gi --cpu 2 --port 7860 \
  --allow-unauthenticated \
  --set-env-vars SARVAM_API_KEY=...
```

The one wrinkle: scale-to-zero means a cold request loads the 536MB model first, so the
first hit after idle takes 20–40 seconds. `--min-instances 1` fixes that but leaves the
free tier.

---

## Paid, if it ever becomes an option

| host | cost | note |
|---|---|---|
| Hugging Face PRO | $9/mo | Docker Spaces now need PRO; free CPU is static-only |
| Render Standard | $25/mo | `starter` is 512MB — same as free — so it does **not** work |
