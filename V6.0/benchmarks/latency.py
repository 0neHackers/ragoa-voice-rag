"""Latency benchmark — P50 / P70 / P100 over real queries.

Reports the two numbers DECISIONS.md D5 commits to, separately and labelled:

* **retrieval pipeline** — guardrails, query embed, dense + BM25 search, RRF fusion. This
  is the number held to the task's <200ms bar.
* **full pipeline** — the above plus LLM generation and the groundedness check. Reported
  whenever generation is enabled, and never folded into the headline number.

Three things here are deliberate, because each is a way this measurement could flatter
itself:

**Queries come from the dataset, not from a hand-picked list.** They are real MSMARCO-XI
queries sampled from the corpus that was indexed, so the benchmark measures the retrieval
the system actually does. A curated set of questions known to work well is how a lucky
run gets reported as a typical one.

**Warmup runs are excluded and stated.** The first ONNX inference pays graph
initialisation — several hundred ms on this model. Including it makes P100 a measurement
of cold start; excluding it silently would be hiding a real cost. So warmup is run
explicitly, its cost is printed, and the reported percentiles are steady-state.

**Declines are counted, not discarded.** A guardrail decline is a fast path — dropping
those runs from the sample would lower every percentile by removing exactly the cheapest
requests. They are included in the latency numbers and reported separately as a rate, so
a suspiciously fast P50 is visible as a high decline rate rather than looking like speed.

    python -m benchmarks.latency --n 50 --index index_store
    python -m benchmarks.latency --n 50 --no-generation      # retrieval only
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile. P100 is the true maximum.

    Nearest-rank rather than interpolated because the task asks for P100, and an
    interpolating percentile (numpy's default) reports something that is not any observed
    measurement. For a latency budget the honest P100 is the worst request actually served.
    """
    if not sorted_values:
        return 0.0
    if p >= 100:
        return sorted_values[-1]
    rank = max(1, int(round(p / 100.0 * len(sorted_values))))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def summarise(label: str, values: list[float], budget_ms: float | None = None) -> dict:
    if not values:
        return {"label": label, "n": 0}
    ordered = sorted(values)
    row = {
        "label": label,
        "n": len(ordered),
        "p50_ms": round(percentile(ordered, 50), 2),
        "p70_ms": round(percentile(ordered, 70), 2),
        "p90_ms": round(percentile(ordered, 90), 2),
        "p95_ms": round(percentile(ordered, 95), 2),
        "p100_ms": round(percentile(ordered, 100), 2),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "min_ms": round(ordered[0], 2),
        "stdev_ms": round(statistics.pstdev(ordered), 2) if len(ordered) > 1 else 0.0,
    }
    if budget_ms is not None:
        row["budget_ms"] = budget_ms
        row["within_budget"] = row["p100_ms"] <= budget_ms
        row["pct_within_budget"] = round(
            100.0 * sum(1 for v in ordered if v <= budget_ms) / len(ordered), 1
        )
    return row


def load_queries(lang: str, n: int, index_query_ids: set[str]) -> list[str]:
    """Sample real dataset queries whose passages are in the index."""
    from data.loader import load_examples

    rows = load_examples(lang=lang, limit=n * 12)
    queries: list[str] = []
    seen: set[str] = set()

    for row in rows:
        qid = str(row.query_id)
        query = row.query.strip()
        if not query or query in seen:
            continue
        if index_query_ids and qid not in index_query_ids:
            continue
        seen.add(query)
        queries.append(query)
        if len(queries) >= n:
            break
    return queries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark pipeline latency (P50/P70/P100).")
    ap.add_argument("--index", default="index_store")
    ap.add_argument("--n", type=int, default=50, help="queries to run (task asks for 30-50+)")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--warmup", type=int, default=3, help="excluded warmup runs")
    ap.add_argument("--no-generation", action="store_true",
                    help="retrieval pipeline only — no LLM call")
    ap.add_argument("--out", default=None, help="defaults to benchmarks/results_<date>.json")
    args = ap.parse_args(argv)

    import warnings

    warnings.filterwarnings("ignore")

    from harness.factory import build_pipeline
    from harness.types import ResponseStatus

    print(f"Building pipeline from {args.index} ...", flush=True)
    build_started = time.perf_counter()
    pipeline = build_pipeline(
        args.index, with_stt=False, with_generation=not args.no_generation, warmup=False,
    )
    build_ms = (time.perf_counter() - build_started) * 1000

    warm_started = time.perf_counter()
    pipeline.retriever.embedder.warmup()
    warmup_ms = (time.perf_counter() - warm_started) * 1000

    health = pipeline.health()
    # An LLM-less run still reports a "full pipeline" number, and it would be a
    # misleading one — extractive generation is a string slice, so full_ms collapses onto
    # retrieval_ms. Detect it and label every downstream report accordingly.
    generation_mode = getattr(pipeline.generator, "mode", None) if pipeline.generator else None
    extractive_only = generation_mode == "extractive"

    print(f"Index: {health['index_size']} chunks ({health['index_backend']}), "
          f"model {health['embed_model']}")
    print(f"Load {build_ms:.0f}ms, embedder warmup {warmup_ms:.0f}ms (excluded from results)")
    if not health["generation_enabled"]:
        print("Generation: disabled (retrieval-only run)")
    elif extractive_only:
        print("Generation: EXTRACTIVE FALLBACK — no ANTHROPIC_API_KEY is set.")
        print("  The full-pipeline number below therefore does NOT include an LLM call")
        print("  and is not a measurement of end-to-end answer latency.")
    else:
        print(f"Generation: ENABLED ({generation_mode})")

    index_query_ids = {c.query_id for c in pipeline.retriever.chunks if c.query_id}
    queries = load_queries(args.lang, args.n, index_query_ids)
    if len(queries) < args.n:
        print(f"WARNING: only {len(queries)} distinct queries available (asked for {args.n}).")
    if len(queries) < 30:
        print("WARNING: fewer than 30 queries — below the task's stated minimum.")

    # -- warmup, excluded --------------------------------------------------
    for query in queries[: args.warmup]:
        pipeline.run_text(query)

    # -- measured ----------------------------------------------------------
    retrieval_ms: list[float] = []
    full_ms: list[float] = []
    per_stage: dict[str, list[float]] = {}
    statuses: dict[str, int] = {}
    decline_reasons: dict[str, int] = {}
    rows: list[dict] = []

    print(f"\nRunning {len(queries)} queries ...", flush=True)
    started = time.perf_counter()

    for i, query in enumerate(queries, start=1):
        response = pipeline.run_text(query)

        retrieval_ms.append(response.latency.retrieval_ms)
        statuses[response.status.value] = statuses.get(response.status.value, 0) + 1
        if response.status is ResponseStatus.DECLINED:
            decline_reasons[response.reason or "?"] = decline_reasons.get(response.reason or "?", 0) + 1
        if health["generation_enabled"] and response.latency.generation_ms > 0:
            full_ms.append(response.latency.full_ms)

        for stage, ms in response.latency.stages.items():
            per_stage.setdefault(stage, []).append(ms)

        rows.append({
            "query": query[:120],
            "status": response.status.value,
            "reason": response.reason,
            "retrieval_ms": round(response.latency.retrieval_ms, 2),
            "full_ms": round(response.latency.full_ms, 2),
            "top_score": round(
                max((c.dense_score or 0.0) for c in response.retrieved), 4
            ) if response.retrieved else None,
        })

        if i % 10 == 0:
            print(f"  {i}/{len(queries)} ...", flush=True)

    wall_s = time.perf_counter() - started

    # -- report ------------------------------------------------------------
    budget = pipeline.config.retrieval_budget_ms
    retrieval_summary = summarise("retrieval_pipeline", retrieval_ms, budget_ms=budget)
    full_label = ("full_pipeline_extractive_no_llm" if extractive_only
                  else "full_pipeline_with_generation")
    full_summary = summarise(full_label, full_ms) if full_ms else None

    stage_summaries = [
        summarise(f"stage:{stage}", values)
        for stage, values in sorted(per_stage.items(), key=lambda kv: -statistics.fmean(kv[1]))
    ]

    print("\n" + "=" * 78)
    print("LATENCY RESULTS")
    print("=" * 78)
    print(f"\nRETRIEVAL PIPELINE  <- this is the number held to the {budget:.0f}ms bar")
    print("  (guardrails + query embed + dense search + BM25 + RRF fusion)")
    _print_row(retrieval_summary)
    print(f"  within {budget:.0f}ms: {retrieval_summary['pct_within_budget']}% of queries"
          f"   P100 {'PASS' if retrieval_summary['within_budget'] else 'OVER BUDGET'}")

    if full_summary:
        if extractive_only:
            print("\nFULL PIPELINE (EXTRACTIVE FALLBACK — NO LLM CALL)")
            print("  Not a valid end-to-end answer latency. With no ANTHROPIC_API_KEY the")
            print("  generation stage is a string slice, so this collapses onto retrieval.")
        else:
            print("\nFULL PIPELINE (including LLM generation + groundedness)")
            print("  Not held to the 200ms bar — generation latency is the provider's TTFT.")
        _print_row(full_summary)

    print("\nPER-STAGE (mean-ordered)")
    for row in stage_summaries[:10]:
        print(f"  {row['label']:34} p50 {row['p50_ms']:8.2f}ms   p100 {row['p100_ms']:8.2f}ms")

    print(f"\nOUTCOMES over {len(queries)} queries")
    for status, count in sorted(statuses.items()):
        print(f"  {status:10} {count:4}  ({count / len(queries):.0%})")
    if decline_reasons:
        print("  declines by reason:")
        for reason, count in sorted(decline_reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {reason:22} {count}")
    print(f"\nWall clock {wall_s:.1f}s for {len(queries)} queries "
          f"({wall_s / len(queries) * 1000:.0f}ms/query end to end)")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "index": str(args.index),
        "index_size": health["index_size"],
        "index_backend": health["index_backend"],
        "embed_model": health["embed_model"],
        "generation_enabled": health["generation_enabled"],
        "generation_mode": generation_mode,
        "full_pipeline_includes_llm": bool(health["generation_enabled"] and not extractive_only),
        "n_queries": len(queries),
        "warmup_runs_excluded": args.warmup,
        "warmup_ms": round(warmup_ms, 2),
        "index_load_ms": round(build_ms, 2),
        "platform": f"{platform.system()} {platform.release()} / Python {platform.python_version()}",
        "retrieval_pipeline": retrieval_summary,
        "full_pipeline": full_summary,
        "per_stage": stage_summaries,
        "outcomes": statuses,
        "decline_reasons": decline_reasons,
        "wall_clock_s": round(wall_s, 2),
        "queries": rows,
    }

    out = Path(args.out) if args.out else Path(
        f"benchmarks/results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")

    latest = out.parent / "results_latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {latest}")
    return 0


def _print_row(row: dict) -> None:
    print(f"  n={row['n']}   P50 {row['p50_ms']:.2f}ms   P70 {row['p70_ms']:.2f}ms   "
          f"P100 {row['p100_ms']:.2f}ms")
    print(f"  mean {row['mean_ms']:.2f}ms   min {row['min_ms']:.2f}ms   "
          f"stdev {row['stdev_ms']:.2f}ms   P95 {row['p95_ms']:.2f}ms")


if __name__ == "__main__":
    raise SystemExit(main())
