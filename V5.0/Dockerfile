# Voice-Enabled RAG — Team 0neHackers
#
# The index is built at IMAGE BUILD time, not at container start. Embedding 15k chunks
# takes ~13 minutes; doing it on boot means a cold deploy serves 503s for a quarter of an
# hour and any platform health check kills the container long before it finishes. Baking
# it in trades a slower image build for a container that is ready the moment it starts.
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    FASTEMBED_CACHE_DIR=/app/model_cache

WORKDIR /app

# libgomp1 is required by onnxruntime; libsndfile1 by soundfile for audio decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the index into the image. CORPUS_LIMIT is a build arg so a smaller/larger corpus
# can be baked without editing the Dockerfile.
ARG CORPUS_LIMIT=1500
RUN python -m retrieval.build_index --limit ${CORPUS_LIMIT} --out index_store \
    && python -c "from retrieval.vector_index import VectorIndex; \
                  i=VectorIndex.load('index_store'); \
                  print(f'index baked: {i.size} chunks, {i.backend}')"

ENV INDEX_DIR=/app/index_store \
    PORT=8000 \
    EMBED_THREADS=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# One worker on purpose: each would load its own copy of the index and the ONNX session,
# and the pipeline is CPU-bound on embedding, so a second worker competes for the same
# cores rather than adding throughput.
CMD ["sh", "-c", "uvicorn demo.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
