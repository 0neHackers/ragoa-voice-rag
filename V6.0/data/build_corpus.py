"""Corpus + chunking inspection CLI — Phase 2 deliverable.

Downloads/caches the dataset slice and reports what each chunking strategy does to it,
so the "vast chunking" claim is backed by measured numbers rather than four class names.

    python -m data.build_corpus --limit 2000
    python -m data.build_corpus --limit 300 --strategies fixed_size,recursive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunking.registry import STRATEGIES, get_strategy  # noqa: E402
from data.loader import corpus_stats, load_examples  # noqa: E402


def strategy_report(name: str, examples: list) -> dict:
    strategy = get_strategy(name)
    t0 = time.perf_counter()
    chunks = strategy.chunk_examples(examples)
    build_s = time.perf_counter() - t0

    lengths = sorted(c.char_len for c in chunks)
    n = len(lengths) or 1
    return {
        "strategy": name,
        "chunks": len(chunks),
        "build_seconds": round(build_s, 2),
        "chunks_per_passage": round(len(chunks) / max(sum(len(e.passages) for e in examples), 1), 2),
        "chars_p50": lengths[n // 2] if lengths else 0,
        "chars_p90": lengths[int(n * 0.9) - 1] if lengths else 0,
        "chars_max": lengths[-1] if lengths else 0,
        "chars_min": lengths[0] if lengths else 0,
        "config": strategy.describe(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build and profile the chunked corpus.")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--out", default=None, help="write the report to this JSON path")
    args = ap.parse_args(argv)

    print(f"Loading {args.lang}/{args.split} (limit={args.limit})...", flush=True)
    t0 = time.perf_counter()
    examples = load_examples(args.lang, args.split, args.limit)
    print(f"  {len(examples)} examples in {time.perf_counter() - t0:.1f}s\n")

    stats = corpus_stats(examples)
    print("Corpus:")
    for k, v in stats.items():
        print(f"  {k:24s} {v}")

    reports = []
    print("\nChunking strategies:")
    print(f"  {'strategy':<18}{'chunks':>9}{'per-psg':>9}{'p50':>7}{'p90':>7}{'max':>8}{'build s':>9}")
    for name in [s.strip() for s in args.strategies.split(",") if s.strip()]:
        r = strategy_report(name, examples)
        reports.append(r)
        print(f"  {r['strategy']:<18}{r['chunks']:>9}{r['chunks_per_passage']:>9}"
              f"{r['chars_p50']:>7}{r['chars_p90']:>7}{r['chars_max']:>8}{r['build_seconds']:>9}")

    payload = {"corpus": stats, "lang": args.lang, "split": args.split,
               "limit": args.limit, "strategies": reports}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
