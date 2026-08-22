# Deploying this thing

Read the first section before picking a host. It'll save you an evening.

---

## Vercel can't run the backend, and it's worth knowing why up front

Vercel is great, but its Python runtime is serverless functions, and this pipeline doesn't
fit in one. Actual measured sizes:

| what | size |
|---|---|
| onnxruntime | 45 MB |
| numpy | 34 MB |
| faiss-cpu | 15 MB |
| fastembed | 2 MB |
| the embedding model itself | ~240 MB |
| the built index | 38 MB |

A Vercel serverless function caps at **250 MB unzipped**. Strip out pandas, pyarrow and
datasets (all build-time only, since the index ships prebuilt) and you're still at roughly
375 MB. There's no trimming your way under that — the model alone is most of it.

And even if it squeezed in, it'd be miserable. Every cold start would load a 240 MB ONNX
session and 15,449 vectors from scratch. That's seconds of latency on a project whose whole
pitch is a 99 ms P50.

So: **Vercel hosts the page, something with a real container runs the pipeline.** Two ways
to play it.

---

## Option A — Render only, one URL *(recommended)*

FastAPI serves the page and the API together, so one service gets you both: one link for
the form, one thing that can break.

**There's a Blueprint at the repo root now**, with `rootDir: V9.0` baked in — so you don't
have to set a root directory by hand, which was the step most likely to trip you up.

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**.
2. Connect `0neHackers/ragoa-voice-rag`. Render finds `render.yaml` at the root by itself.
3. It'll prompt for **`SARVAM_API_KEY`** — the only secret, marked `sync: false` so it
   never touches git. Paste it there.
4. **Apply.** First build takes **15–25 minutes**: it installs deps, downloads the dataset
   parquet, and embeds all 15,449 chunks *into the image* so the container is ready the
   instant it boots. The alternative — embedding at startup — means serving 503s for a
   quarter of an hour while Render's health check kills and restarts it forever.
5. Check `https://<your-service>.onrender.com/health` returns `"ok": true` with
   `"index_size": 15449`.

**Stay on `starter` or higher.** The free tier's 512 MB will OOM — the index plus the ONNX
session needs about a gig. Free instances also sleep after inactivity, and a judge opening a
cold link and waiting 50 seconds is not the first impression you want.

The blueprint pins `region: singapore`, the closest Render region to India and to Sarvam's
endpoints. Every generation, translation and TTS call is a round trip to Sarvam, so region
choice shows up directly in the latency a judge sees.

### If you'd rather I did it

I can't — Render's API returns 401 without a token and I have no credentials for your
account. If you want me to run the deploy and verify it end to end, generate a key at
**Render Dashboard → Account Settings → API Keys → Create API Key** and send it. Otherwise
the five steps above are the whole job.

## Option B — Vercel frontend + Render backend

Only worth it if you want the page on a custom domain or served from the edge. It's two
things to keep alive instead of one.

**1.** Deploy the backend on Render exactly as in Option A. Note its URL.

**2.** Point the frontend at it. Edit `demo/config.js`:

```js
window.__API_BASE__ = 'https://voice-rag-0nehackers.onrender.com';
```

**3.** Deploy just the `demo/` folder to Vercel:

```bash
cd master_repo/V9.0/demo
npx vercel --prod
```

When it asks for a framework, say **Other**. There's no build step — it's a static page.

The backend already sends permissive CORS (`allow_origins=["*"]`, no credentials), so
cross-origin calls work without a proxy or any extra config.

**One gotcha:** microphone access needs a secure context. Both Vercel and Render are HTTPS,
so you're fine — but if you test by opening `index.html` off your filesystem, the mic button
will silently do nothing. Serve it over http://localhost or https://, always.

---

## Option C — Hugging Face Spaces

Honestly a decent fit, since the model and dataset are already on HF and the free tier gives
you 16 GB. Create a Space with the **Docker** SDK, push the contents of `V9.0/`, and add
`SARVAM_API_KEY` as a Space secret. Same Dockerfile, no changes.

---

## GitHub — done

**https://github.com/0neHackers/ragoa-voice-rag** — public, `main`, 15 commits, 434 files.

The per-phase history (V0.0 → V9.0) is intact, which matters: Video 1 is about process, and
a repo with one giant commit dated the day of the deadline tells the opposite story. Each
version folder is a frozen snapshot with its own dated `CHANGELOG.md` entry.

Checked before pushing, and worth re-checking if you ever force-push:

```bash
git ls-files | grep -i env          # only .env.example files, never .env
```

The real `.env` is git-ignored, and no key material is in any tracked file. The built index
(38 MB) is excluded too — it's reproducible from `retrieval.build_index`, so committing it
into every version folder would bloat history for nothing.

> **Rotate the Sarvam key once judging closes.** It went through a chat transcript, so treat
> it as exposed no matter how carefully the repo handles it.

---

## Before you paste the link into the form

- [ ] `/health` returns `"ok": true` with the right `index_size`
- [ ] Root URL loads in an **incognito** window (no cached session lying to you)
- [ ] Ask a real question, get a real answer
- [ ] Ask something the corpus can't answer, get a refusal with a reason
- [ ] Microphone works on the deployed URL, not just localhost
- [ ] Tick **Translate from English** and ask an English question — it should answer
- [ ] Press **Hear it** on both the question and the answer — Hindi audio should play
- [ ] Try it on a phone — the layout goes down to 320px, so it should be fine, but look
- [ ] **Leave it running past the deadline.** You don't know the judging window.
