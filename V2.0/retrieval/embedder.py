"""Embedding wrapper around fastembed's ONNX runtime.

One process-wide instance per model. The model is a ~220MB ONNX session and the cost of
constructing it is ~2.5s — paying that on a query path would blow the entire latency
budget on its own, so instances are cached by model name and the first construction is
expected to happen at index build time, not at request time.

Everything downstream assumes L2-normalised vectors: it makes FAISS inner-product search
identical to cosine similarity, which in turn lets the confidence guardrail state its
threshold in cosine terms that mean the same thing across corpora.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_CACHE: dict[str, "Embedder"] = {}
_LOCK = threading.Lock()


class Embedder:
    def __init__(self, model_name: str | None = None, threads: int | None = None) -> None:
        self.model_name = model_name or os.getenv("EMBED_MODEL", DEFAULT_MODEL)
        repo_root = Path(__file__).resolve().parent.parent.parent
        os.environ.setdefault("HF_HOME", str(repo_root / "hf_cache"))

        # fastembed defaults its ONNX cache to the OS temp directory, which Windows and
        # most PaaS containers are free to clear. Pinning it under the repo means a
        # deployed instance cannot lose its model between restarts and re-download
        # 220MB on a user's first query.
        cache_dir = os.getenv("FASTEMBED_CACHE_DIR") or str(repo_root / "model_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        from fastembed import TextEmbedding

        self._model = TextEmbedding(self.model_name, cache_dir=cache_dir, threads=threads)
        self._dim: int | None = None
        self._warm = False

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed_texts(["dimension probe"]).shape[1])
        return self._dim

    def embed_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Embed a list of documents. Returns float32 `(n, dim)`, L2-normalised."""
        if not texts:
            return np.zeros((0, self.dim if self._dim else 384), dtype="float32")
        vecs = np.asarray(list(self._model.embed(texts, batch_size=batch_size)), dtype="float32")
        return _l2_normalise(vecs)

    def warmup(self) -> float:
        """Run one throwaway inference so the first real query is not an outlier.

        The first ONNX `run()` pays graph initialisation and weight paging — on this
        model that is several hundred milliseconds. Benchmarks and the demo both call
        this at startup so the reported P100 measures steady-state retrieval rather
        than a one-off cold start.
        """
        import time

        t0 = time.perf_counter()
        self.embed_texts(["warmup"])
        self._warm = True
        return (time.perf_counter() - t0) * 1000.0

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns float32 `(dim,)`, L2-normalised.

        Separate from `embed_texts` because this is the one embedding call on the hot
        path, and because asymmetric models need a different prefix here. The current
        model (MiniLM, mean-pooled) is symmetric and needs none — see DECISIONS.md D3.
        """
        return self.embed_texts([text])[0]


def _l2_normalise(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)  # a zero vector would otherwise produce NaN
    return vecs / norms


def get_embedder(model_name: str | None = None) -> Embedder:
    """Process-wide cached embedder. Thread-safe, double-checked."""
    key = model_name or os.getenv("EMBED_MODEL", DEFAULT_MODEL)
    if key not in _CACHE:
        with _LOCK:
            if key not in _CACHE:
                _CACHE[key] = Embedder(key)
    return _CACHE[key]


__all__ = ["Embedder", "get_embedder", "DEFAULT_MODEL"]
