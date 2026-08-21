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

FastAPI already serves the HTML, so a single service gets you both. One link for the form,
one thing that can break, done.

**1. Push to GitHub** (see below).

**2.** On Render: **New → Blueprint**, point it at your repo. `render.yaml` is already in
`V5.0/` and configured.

**3. Set the root directory to `master_repo/V7.0`.** Render looks for the Dockerfile
relative to the repo root, and yours lives one level down. Either set it in the dashboard
or move `V5.0/`'s contents to the repo root before pushing.

**4.** Add your keys in the dashboard — `SARVAM_API_KEY` and `ANTHROPIC_API_KEY`. They're
marked `sync: false` in the blueprint specifically so they never end up in git.

**5. Stay on the `starter` plan or higher.** The free tier's 512 MB will OOM — the index
plus the ONNX session needs about a gig. Free instances also sleep after inactivity, and a
judge hitting a cold link and waiting 50 seconds is not the first impression you want.

**6.** First build takes **15–20 minutes**, because the Dockerfile embeds all 15,449 chunks
during the image build rather than at container start. That's deliberate: the alternative is
a container that boots and then serves 503s for a quarter of an hour while Render's health
check kills it and retries forever.

**7.** Check `https://your-app.onrender.com/health` returns `"ok": true`, then open the root
URL in an incognito window and actually ask it something.

---

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
cd master_repo/V7.0/demo
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
you 16 GB. Create a Space with the **Docker** SDK, push the contents of `V5.0/`, and add the
two keys as Space secrets. Same Dockerfile, no changes.

---

## GitHub

The repo is already committed locally at `master_repo/`, with real per-phase history
(V0.0 → V5.0). Keep that history — Video 1 is about process, and a repo with one giant
commit dated the day of the deadline tells the opposite story.

```bash
cd master_repo
git remote add origin https://github.com/<you>/<repo>.git
git branch -M main
git push -u origin main
```

Then confirm nothing secret went up:

```bash
git ls-files | grep -i env
```

You should see `.env.example` files and nothing else. If a real `.env` shows up, stop and
remove it from history before pushing anywhere public.

Make the repo **public** — the form asks for a link judges can open.

---

## Before you paste the link into the form

- [ ] `/health` returns `"ok": true` with the right `index_size`
- [ ] Root URL loads in an **incognito** window (no cached session lying to you)
- [ ] Ask a real question, get a real answer
- [ ] Ask something the corpus can't answer, get a refusal with a reason
- [ ] Microphone works on the deployed URL, not just localhost
- [ ] Try it on a phone — the layout goes down to 320px, so it should be fine, but look
- [ ] **Leave it running past the deadline.** You don't know the judging window.
