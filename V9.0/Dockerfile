# Voice-Enabled RAG — Team 0neHackers
#
# The index is built at IMAGE BUILD time, not at container start. Embedding 15k chunks
# takes ~13 minutes; doing it on boot means a cold deploy serves 503s for a quarter of an
# hour and any platform health check kills the container long before it finishes. Baking
# it in trades a slower image build for a container that is ready the moment it starts.
#
# Runs as a non-root user. That's good practice anywhere, and it's a hard requirement on
# Hugging Face Spaces, which runs every container as UID 1000 — a root-owned /app there
# means the app can't write its caches and dies on boot.
FROM python:3.13-slim AS base

# HF_HOME and FASTEMBED_CACHE_DIR live under /home/app rather than /app so they stay
# writable when the working tree is owned by someone else, and so a platform that mounts
# over /app doesn't take the model cache with it.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app \
    HF_HOME=/home/app/hf_cache \
    FASTEMBED_CACHE_DIR=/home/app/model_cache

# libgomp1 is required by onnxruntime; libsndfile1 by soundfile for audio decoding.
#
# curl and ca-certificates are here for operability, not for the app. This image doubles
# as the Codespaces devcontainer, and python:3.13-slim ships without curl — which meant
# every "is the server actually up?" check inside a Codespace failed with
# `sh: curl: not found` before it could tell you anything. 2MB to make the container
# debuggable is a good trade.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 app

# cloudflared, for a public URL that doesn't fight you.
#
# Codespaces' own port forwarding turned out to be unusable as a submission link: the
# port visibility silently reverts to private, non-browser clients get a flat 401, and
# anonymous visitors hit an "about to access a development port" interstitial that GitHub
# provides no way to disable. Running a tunnel from inside the container keeps GitHub's
# uptime while producing a plain HTTPS URL that just serves the app.
ARG TARGETARCH=amd64
RUN curl -fsSL -o /usr/local/bin/cloudflared \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${TARGETARCH}" \
    && chmod +x /usr/local/bin/cloudflared \
    && cloudflared --version

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Everything the app touches at runtime has to belong to it — the working tree, the model
# cache, and the corpus cache the index build writes.
RUN chown -R app:app /app /home/app
USER app

# Use the index that ships in the repo; build it only if it's missing.
#
# It used to always rebuild here, and that quietly didn't work on small machines.
# `build_parallelism` is `min(8, cpu_count - 2)`, so a 2-core Codespace gets **one**
# worker at ~294ms/chunk — 15,449 chunks is roughly **75 minutes**, past the build
# timeout. The build looks hung at "Embedding 15449 chunks..." because that step prints
# nothing until it finishes.
#
# The 38MB index is committed for exactly this reason, so the normal path is a file copy
# that costs seconds. The fallback still exists so the image builds from a clean checkout
# without one, and it says which path it took rather than leaving you to guess.
ARG CORPUS_LIMIT=1500
RUN if [ -f index_store/vectors.npy ]; then \
        echo "using the index committed in the repo (no rebuild)"; \
    else \
        echo "no index found — building one (slow on few cores)"; \
        python -m retrieval.build_index --limit ${CORPUS_LIMIT} --out index_store; \
    fi \
    && python -c "from retrieval.vector_index import VectorIndex; \
                  i=VectorIndex.load('index_store'); \
                  print(f'index ready: {i.size} chunks, {i.backend}')"

# 7860 is the Hugging Face Spaces convention and is what `app_port` in the Space README
# points at. Render overrides PORT from its own environment, so one default serves both.
ENV INDEX_DIR=/app/index_store \
    PORT=7860 \
    EMBED_THREADS=1 \
    STT_TRANSPORT=auto

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
        p=os.getenv('PORT','7860'); \
        sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=4).status==200 else 1)"

# One worker on purpose: each would load its own copy of the index and the ONNX session,
# and the pipeline is CPU-bound on embedding, so a second worker competes for the same
# cores rather than adding throughput.
CMD ["sh", "-c", "uvicorn demo.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
