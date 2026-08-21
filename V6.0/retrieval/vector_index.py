"""In-process dense vector index — FAISS, with an exact-NumPy fallback.

Per DECISIONS.md D4 there is no network hop here: at a <200ms budget, a hosted vector DB
would spend 20–100ms on round-trip before computing anything.

`IndexFlatIP` (exact inner product) rather than an ANN index like HNSW or IVF, for two
reasons. At ~20k vectors an exact search is a single 20k×384 matmul in the low
single-digit milliseconds, so ANN buys nothing measurable. And ANN buys it at the cost
of recall — which would silently move the score distribution the confidence guardrail
thresholds against, making the guardrail's calibration a function of the index's
approximation error rather than of the corpus.

Vectors are L2-normalised upstream, so inner product **is** cosine similarity and every
score in this module is in [-1, 1].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.types import Chunk

try:  # pragma: no cover - environment dependent
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False


@dataclass(slots=True)
class DenseHit:
    idx: int
    score: float


class _NumpyFlatIP:
    """Exact inner-product search in NumPy.

    Implements the same math as `faiss.IndexFlatIP` so behaviour is identical whichever
    backend is active — the fallback must not quietly change retrieval quality.
    """

    def __init__(self, dim: int) -> None:
        self.d = dim
        self._vectors = np.zeros((0, dim), dtype="float32")

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = (
            vectors.astype("float32", copy=False) if self.ntotal == 0
            else np.vstack([self._vectors, vectors.astype("float32", copy=False)])
        )

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            empty = np.full((queries.shape[0], k), -1, dtype="int64")
            return np.zeros((queries.shape[0], k), dtype="float32"), empty

        sims = queries.astype("float32", copy=False) @ self._vectors.T
        k = min(k, self.ntotal)
        # argpartition is O(n) vs argsort's O(n log n); only the top-k needs ordering.
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        scores = np.take_along_axis(part_scores, order, axis=1)
        return scores.astype("float32"), idx.astype("int64")


class VectorIndex:
    """Dense index over chunk embeddings, with the chunks kept alongside."""

    def __init__(self, dim: int, backend: str | None = None) -> None:
        self.dim = dim
        use_faiss = FAISS_AVAILABLE if backend is None else (backend == "faiss")
        if use_faiss and not FAISS_AVAILABLE:
            raise RuntimeError("faiss backend requested but faiss is not installed")
        self.backend = "faiss" if use_faiss else "numpy"
        self._index = faiss.IndexFlatIP(dim) if use_faiss else _NumpyFlatIP(dim)
        self.chunks: list[Chunk] = []

    # -- build --------------------------------------------------------------- #

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} chunks vs {vectors.shape[0]} vectors"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim}-dim vectors, got {vectors.shape[1]}")
        self._index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.chunks.extend(chunks)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    # -- query --------------------------------------------------------------- #

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[DenseHit]:
        if self.size == 0:
            return []
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32")
        scores, idx = self._index.search(q, min(k, self.size))
        return [
            DenseHit(idx=int(i), score=float(s))
            for s, i in zip(scores[0], idx[0]) if i >= 0
        ]

    def centroid(self) -> np.ndarray:
        """Mean of all indexed vectors, L2-normalised.

        Used by the off-topic guardrail as a cheap "is this query anywhere near what the
        corpus is about" reference, so it needs no separate model or training pass.
        """
        vectors = self._all_vectors()
        if vectors.shape[0] == 0:
            return np.zeros(self.dim, dtype="float32")
        mean = vectors.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return (mean / norm).astype("float32") if norm > 1e-12 else mean.astype("float32")

    def _all_vectors(self) -> np.ndarray:
        if self.backend == "numpy":
            return self._index._vectors  # noqa: SLF001 - our own fallback class
        return self._index.reconstruct_n(0, self.size)

    # -- persistence --------------------------------------------------------- #

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        vectors = self._all_vectors()
        np.save(directory / "vectors.npy", vectors)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        (directory / "index_meta.json").write_text(
            json.dumps({"dim": self.dim, "size": self.size, "backend": self.backend}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path, backend: str | None = None) -> "VectorIndex":
        directory = Path(directory)
        meta = json.loads((directory / "index_meta.json").read_text(encoding="utf-8"))

        index = cls(dim=meta["dim"], backend=backend)
        vectors = np.load(directory / "vectors.npy")
        chunks = [
            Chunk.model_validate_json(line)
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        index.add(chunks, vectors)
        return index


__all__ = ["VectorIndex", "DenseHit", "FAISS_AVAILABLE"]
