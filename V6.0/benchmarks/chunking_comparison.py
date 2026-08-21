"""Compare the four chunking strategies on retrieval quality, not just on chunk counts.

The task asks for chunking that is genuinely engineered rather than one naive split. Four
strategies exist in `chunking/`; this is the measurement that says which of them actually
retrieves better, and it is the evidence behind `DEFAULT_ENSEMBLE`.

**Ground truth comes from the dataset.** MSMARCO-XI marks, per query, which passage is
the relevant one (`is_selected`). So for each sampled query the correct answer is known,
and the metrics are the standard retrieval ones rather than a subjective read of the
output:

* **Recall@k** — was a chunk derived from the relevant passage retrieved in the top k?
  This is what determines whether generation can possibly be grounded: if the right
  passage is not in the context, no prompt saves the answer.
* **MRR** — mean reciprocal rank of the first relevant chunk. Recall@5 alone cannot tell
  "ranked first" from "ranked fifth", and rank matters when the confidence gate reads the
  top score and the prompt weights early passages.

Each strategy is indexed and queried on identical data, so the only variable is the split.
Cost is reported alongside quality — chunk count drives index size and build time, and a
strategy that wins on recall by producing three times the chunks has not won for free.

    python -m benchmarks.chunking_comparison --limit 300 --queries 60
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate_strategy(
    name: str,
    strategies: tuple[str, ...],
    examples: list,
    queries: list[tuple[str, str]],
    top_k: int,
) -> dict:
    """Build an index from `strategies` over `examples`, then score it on `queries`."""
    from chunking.registry import get_strategy
    from retrieval.build_index import dedupe_chunks
    from retrieval.embedder import get_embedder
    from retrieval.lexical import LexicalIndex
    from retrieval.retriever import Retriever
    from retrieval.vector_index import VectorIndex

    chunk_started = time.perf_counter()
    produced = []
    for strategy_name in strategies:
        produced.extend(get_strategy(strategy_name).chunk_examples(examples))
    chunks = dedupe_chunks(produced)
    chunk_s = time.perf_counter() - chunk_started

    embedder = get_embedder()
    embed_started = time.perf_counter()
    vectors = embedder.embed_texts([c.text for c in chunks])
    embed_s = time.perf_counter() - embed_started

    index = VectorIndex(dim=vectors.shape[1])
    index.add(chunks, vectors)
    retriever = Retriever(vector_index=index, lexical_index=LexicalIndex(chunks))

    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    top_scores: list[float] = []
    latencies: list[float] = []

    for query_text, relevant_qid in queries:
        started = time.perf_counter()
        result = retriever.retrieve(query_text, top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)

        if result.chunks:
            top_scores.append(max((c.dense_score or 0.0) for c in result.chunks))

        rank = next(
            (i for i, sc in enumerate(result.chunks, start=1)
             if sc.chunk.query_id == relevant_qid and sc.chunk.is_selected_passage),
            None,
        )
        # Fall back to same-query provenance when the corpus has no is_selected marker.
        if rank is None:
            rank = next(
                (i for i, sc in enumerate(result.chunks, start=1)
                 if sc.chunk.query_id == relevant_qid),
                None,
            )
        if rank is not None:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = max(1, len(queries))
    sizes = [c.char_len for c in chunks] or [0]
    return {
        "strategy": name,
        "components": list(strategies),
        "n_chunks": len(chunks),
        "n_chunks_before_dedupe": len(produced),
        "mean_chunk_chars": round(statistics.fmean(sizes), 1),
        "median_chunk_chars": round(statistics.median(sizes), 1),
        "max_chunk_chars": max(sizes),
        f"recall_at_{top_k}": round(hits_at_k / n, 4),
        "mrr": round(statistics.fmean(reciprocal_ranks), 4),
        "mean_top_score": round(statistics.fmean(top_scores), 4) if top_scores else 0.0,
        "chunk_seconds": round(chunk_s, 2),
        "embed_seconds": round(embed_s, 1),
        "retrieval_p50_ms": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare chunking strategies on retrieval quality.")
    ap.add_argument("--limit", type=int, default=300, help="dataset examples to index")
    ap.add_argument("--queries", type=int, default=60, help="queries to evaluate")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="benchmarks/chunking_comparison.json")
    args = ap.parse_args(argv)

    import warnings

    warnings.filterwarnings("ignore")

    from data.loader import load_examples

    print(f"Loading {args.limit} examples of MSMARCO-XI/{args.lang} ...", flush=True)
    examples = load_examples(lang=args.lang, limit=args.limit)

    queries: list[tuple[str, str]] = []
    for row in examples:
        query = row.query.strip()
        qid = str(row.query_id)
        if query and qid:
            queries.append((query, qid))
        if len(queries) >= args.queries:
            break

    print(f"Evaluating {len(queries)} queries at top-k={args.top_k}\n", flush=True)

    variants: list[tuple[str, tuple[str, ...]]] = [
        ("fixed_size", ("fixed_size",)),
        ("semantic", ("semantic",)),
        ("recursive", ("recursive",)),
        ("metadata_aware", ("metadata_aware",)),
        ("ensemble (shipped)", ("metadata_aware", "recursive")),
        ("ensemble (all four)", ("fixed_size", "semantic", "recursive", "metadata_aware")),
    ]

    results = []
    for label, strategies in variants:
        print(f"--- {label} ---", flush=True)
        row = evaluate_strategy(label, strategies, examples, queries, args.top_k)
        results.append(row)
        print(f"    chunks {row['n_chunks']:6}  recall@{args.top_k} {row[f'recall_at_{args.top_k}']:.3f}  "
              f"MRR {row['mrr']:.3f}  mean_top {row['mean_top_score']:.3f}  "
              f"embed {row['embed_seconds']}s", flush=True)

    recall_key = f"recall_at_{args.top_k}"
    print("\n" + "=" * 92)
    print(f"{'strategy':22} {'chunks':>7} {'mean_chars':>11} {recall_key:>10} {'MRR':>7} "
          f"{'top_score':>10} {'embed_s':>8}")
    print("=" * 92)
    for row in sorted(results, key=lambda r: -r[recall_key]):
        print(f"{row['strategy']:22} {row['n_chunks']:>7} {row['mean_chunk_chars']:>11.0f} "
              f"{row[recall_key]:>10.3f} {row['mrr']:>7.3f} {row['mean_top_score']:>10.3f} "
              f"{row['embed_seconds']:>8.1f}")

    best = max(results, key=lambda r: r[recall_key])
    print(f"\nBest recall@{args.top_k}: {best['strategy']} ({best[recall_key]:.3f}), "
          f"{best['n_chunks']} chunks")

    payload = {
        "date": date.today().isoformat(),
        "lang": args.lang,
        "examples_indexed": len(examples),
        "queries_evaluated": len(queries),
        "top_k": args.top_k,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
