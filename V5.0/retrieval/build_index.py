"""Build and persist the retrieval index — Phase 3 deliverable.

    python -m retrieval.build_index --limit 2000
    python -m retrieval.build_index --limit 500 --strategies fixed_size --out index_fixed

Chunking happens here, at build time, not per query. That is the whole reason the
retrieval pipeline can fit in a 200ms budget: the query path only embeds one string and
searches two prebuilt indexes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunking.registry import DEFAULT_ENSEMBLE, get_strategy  # noqa: E402
from data.loader import load_examples  # noqa: E402
from harness.types import Chunk  # noqa: E402
from retrieval.embedder import get_embedder  # noqa: E402
from retrieval.lexical import LexicalIndex  # noqa: E402
from retrieval.vector_index import FAISS_AVAILABLE, VectorIndex  # noqa: E402

DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "index_store"


def dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks with identical text.

    Running several strategies over the same corpus produces heavy overlap — most
    MSMARCO-XI passages are short enough that metadata-aware and recursive emit the
    identical string. Indexing both wastes memory and, worse, lets one passage occupy
    several of the top-k slots, crowding out genuinely different context.
    """
    seen: set[str] = set()
    out: list[Chunk] = []
    for chunk in chunks:
        key = chunk.text.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def build(
    lang: str = "hi",
    split: str = "validation",
    limit: int = 2000,
    strategies: tuple[str, ...] = DEFAULT_ENSEMBLE,
    out_dir: Path = DEFAULT_INDEX_DIR,
    verbose: bool = True,
) -> dict:
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    stats: dict[str, object] = {"lang": lang, "split": split, "limit": limit,
                                "strategies": list(strategies)}

    t0 = time.perf_counter()
    examples = load_examples(lang, split, limit)
    stats["load_seconds"] = round(time.perf_counter() - t0, 2)
    log(f"Loaded {len(examples)} examples in {stats['load_seconds']}s")

    t0 = time.perf_counter()
    raw: list[Chunk] = []
    per_strategy: dict[str, int] = {}
    for name in strategies:
        produced = get_strategy(name).chunk_examples(examples)
        per_strategy[name] = len(produced)
        raw.extend(produced)
        log(f"  {name}: {len(produced)} chunks")

    chunks = dedupe_chunks(raw)
    stats["chunk_seconds"] = round(time.perf_counter() - t0, 2)
    stats["chunks_raw"] = len(raw)
    stats["chunks_indexed"] = len(chunks)
    stats["chunks_per_strategy"] = per_strategy
    log(f"Chunked: {len(raw)} raw -> {len(chunks)} unique in {stats['chunk_seconds']}s")

    if not chunks:
        raise RuntimeError("No chunks produced — refusing to build an empty index.")

    embedder = get_embedder()
    log(f"Embedding {len(chunks)} chunks with {embedder.model_name}...")
    t0 = time.perf_counter()
    vectors = embedder.embed_texts([c.text for c in chunks])
    stats["embed_seconds"] = round(time.perf_counter() - t0, 2)
    log(f"  {stats['embed_seconds']}s ({stats['embed_seconds'] / len(chunks) * 1000:.1f} ms/chunk)")

    t0 = time.perf_counter()
    index = VectorIndex(dim=vectors.shape[1])
    index.add(chunks, vectors)
    lexical = LexicalIndex(chunks)
    stats["index_seconds"] = round(time.perf_counter() - t0, 2)
    stats["backend"] = index.backend
    stats["dim"] = index.dim
    log(f"Indexed {index.size} vectors ({index.backend}) + BM25 over {lexical.size} docs "
        f"in {stats['index_seconds']}s")

    t0 = time.perf_counter()
    index.save(out_dir)
    (Path(out_dir) / "build_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"Saved to {out_dir} in {time.perf_counter() - t0:.1f}s")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the dense + lexical retrieval index.")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--strategies", default=",".join(DEFAULT_ENSEMBLE))
    ap.add_argument("--out", default=str(DEFAULT_INDEX_DIR))
    args = ap.parse_args(argv)

    if not FAISS_AVAILABLE:
        print("NOTE: faiss unavailable — using the exact NumPy index (identical math).",
              file=sys.stderr)

    build(
        lang=args.lang, split=args.split, limit=args.limit,
        strategies=tuple(s.strip() for s in args.strategies.split(",") if s.strip()),
        out_dir=Path(args.out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
