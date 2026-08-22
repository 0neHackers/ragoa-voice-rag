"""Calibrate the confidence-gate threshold against real data.

The gate's job is to separate "the corpus can answer this" from "it cannot". That
boundary is a property of the embedding model and the corpus, not something to guess, so
it is measured against two populations built from the dataset itself:

*Answerable* — real MSMARCO-XI queries whose own passages are in the index. These must be
answered; a decline here is a false refusal, the expensive error for a demo.

*Unanswerable* — queries the corpus provably cannot serve. Three kinds, because they fail
differently: held-out queries whose passages were excluded from the index (the realistic
case — a plausible question about absent content), synthetic gibberish, and questions
about subject matter the corpus has no notion of.

The output is the score distribution of both populations and the threshold that best
separates them. A threshold is only worth shipping if the populations actually separate;
if they overlap heavily this prints that fact rather than picking a number that looks
decisive, because a confidently-chosen threshold over overlapping distributions is a
guardrail that fires at random.

    python -m benchmarks.calibrate_thresholds --index index_store --n 60
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Questions about subject matter a Hindi MS MARCO web corpus has no notion of.
OUT_OF_DOMAIN_HI: tuple[str, ...] = (
    "मेरी पालतू बिल्ली का नाम क्या है?",
    "कल मेरे घर पर रात के खाने में क्या बना था?",
    "मेरे बैंक खाते में कितने रुपये बचे हैं?",
    "मेरा पासवर्ड क्या है?",
    "मेरी अगली छुट्टी कब है?",
    "मेरे पड़ोसी का फ़ोन नंबर क्या है?",
)

GIBBERISH: tuple[str, ...] = (
    "अस्दफ ग्ह्ज्क ल्म्न्प",
    "क्ख्ग् घ्ङ्च् छ्ज्झ्",
    "ब्ल्ब्ल् ब्ल्ब्ल् ब्ल्ब्ल्",
    "१२३४५ ६७८९० ००००",
    "ॐॐॐ ठठठ फफफ",
    "यययय ररर ललल वववव",
)


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values), p)) if values else 0.0


def summarise(name: str, scores: list[float]) -> dict[str, float]:
    arr = np.asarray(scores) if scores else np.zeros(1)
    return {
        "population": name, "n": len(scores),
        "min": round(float(arr.min()), 4), "p05": round(percentile(scores, 5), 4),
        "p25": round(percentile(scores, 25), 4), "median": round(percentile(scores, 50), 4),
        "p75": round(percentile(scores, 75), 4), "p95": round(percentile(scores, 95), 4),
        "max": round(float(arr.max()), 4), "mean": round(float(arr.mean()), 4),
    }


def best_threshold(answerable: list[float], unanswerable: list[float]) -> dict[str, float]:
    """Sweep candidate thresholds and score each by balanced accuracy.

    Balanced accuracy rather than raw accuracy because the two populations are not the
    same size, and because the two errors are not equally bad: a false refusal on a
    good question is worse for a demo than letting one weak query through to a
    groundedness check that may still catch it. The sweep reports both rates so the
    tradeoff is visible rather than buried in a single score.
    """
    candidates = np.arange(0.20, 0.80, 0.005)
    best = {"threshold": 0.42, "balanced_accuracy": 0.0}

    for threshold in candidates:
        true_pass = sum(1 for s in answerable if s >= threshold) / max(1, len(answerable))
        true_block = sum(1 for s in unanswerable if s < threshold) / max(1, len(unanswerable))
        balanced = (true_pass + true_block) / 2
        if balanced > best["balanced_accuracy"]:
            best = {
                "threshold": round(float(threshold), 3),
                "balanced_accuracy": round(balanced, 4),
                "answerable_pass_rate": round(true_pass, 4),
                "unanswerable_block_rate": round(true_block, 4),
            }
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate the retrieval-confidence threshold.")
    ap.add_argument("--index", default="index_store")
    ap.add_argument("--n", type=int, default=60, help="answerable queries to sample")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--out", default="benchmarks/threshold_calibration.json")
    args = ap.parse_args(argv)

    import warnings

    warnings.filterwarnings("ignore")

    from data.loader import load_examples
    from retrieval.lexical import LexicalIndex
    from retrieval.retriever import Retriever
    from retrieval.vector_index import VectorIndex

    print(f"Loading index from {args.index} ...", flush=True)
    vector_index = VectorIndex.load(args.index)
    retriever = Retriever(vector_index=vector_index,
                          lexical_index=LexicalIndex(vector_index.chunks))
    retriever.embedder.warmup()

    indexed_query_ids = {c.query_id for c in vector_index.chunks if c.query_id}
    print(f"Index holds {vector_index.size} chunks from {len(indexed_query_ids)} source queries.")

    # -- answerable: queries whose own passages are indexed ------------------
    print("Loading dataset for query text ...", flush=True)
    # Held-out queries only exist beyond the indexed range, so load past it —
    # sampling within it yields zero held-out queries and a meaningless population.
    rows = load_examples(lang=args.lang, limit=max(args.n * 12, len(indexed_query_ids) + 500))

    answerable_queries: list[str] = []
    heldout_queries: list[str] = []
    for row in rows:
        qid = str(row.query_id)
        query = row.query.strip()
        if not query:
            continue
        if qid in indexed_query_ids and len(answerable_queries) < args.n:
            answerable_queries.append(query)
        elif qid not in indexed_query_ids and len(heldout_queries) < args.n // 2:
            heldout_queries.append(query)

    print(f"Sampled {len(answerable_queries)} answerable, {len(heldout_queries)} held-out queries.")

    def top_scores(query: str) -> tuple[float, float]:
        """Best dense cosine and best BM25 for one query. Both matter — see below."""
        result = retriever.retrieve(query, top_k=5)
        dense = [c.dense_score for c in result.chunks if c.dense_score is not None]
        lexical = [c.lexical_score for c in result.chunks if c.lexical_score is not None]
        return (max(dense) if dense else 0.0, max(lexical) if lexical else 0.0)

    populations: dict[str, list[float]] = {}
    lexical_populations: dict[str, list[float]] = {}
    for name, queries in (
        ("answerable", answerable_queries),
        ("heldout", heldout_queries),
        ("out_of_domain", list(OUT_OF_DOMAIN_HI)),
        ("gibberish", list(GIBBERISH)),
    ):
        print(f"Scoring {name} ({len(queries)}) ...", flush=True)
        scored = [top_scores(q) for q in queries]
        populations[name] = [d for d, _ in scored]
        lexical_populations[name] = [lx for _, lx in scored]

    unanswerable = (populations["out_of_domain"] + populations["gibberish"]
                    + populations["heldout"])
    unanswerable_lex = (lexical_populations["out_of_domain"] + lexical_populations["gibberish"]
                        + lexical_populations["heldout"])

    summaries = [summarise(name, scores) for name, scores in populations.items()]
    summaries.append(summarise("unanswerable_combined", unanswerable))
    recommendation = best_threshold(populations["answerable"], unanswerable)

    # Overlap is the number that decides whether any threshold is meaningful.
    a_p05 = percentile(populations["answerable"], 5)
    u_p95 = percentile(unanswerable, 95)
    separated = a_p05 > u_p95

    print()
    print(f"{'population':24} {'n':>4} {'min':>7} {'p05':>7} {'median':>7} {'p95':>7} {'max':>7}")
    for row in summaries:
        print(f"{row['population']:24} {row['n']:>4} {row['min']:>7.3f} {row['p05']:>7.3f} "
              f"{row['median']:>7.3f} {row['p95']:>7.3f} {row['max']:>7.3f}")

    # -- the lexical signal, reported alongside ------------------------------
    print()
    print(f"{'population':24} {'n':>4} {'bm25 min':>9} {'bm25 med':>9} {'bm25 max':>9}  zero-scoring")
    for name, scores in lexical_populations.items():
        if not scores:
            continue
        zeros = sum(1 for s_ in scores if s_ <= 0.0)
        print(f"{name:24} {len(scores):>4} {min(scores):>9.2f} {percentile(scores, 50):>9.2f} "
              f"{max(scores):>9.2f}  {zeros}/{len(scores)}")

    a_lex_zero = sum(1 for s_ in lexical_populations["answerable"] if s_ <= 0.0)
    u_lex_zero = sum(1 for s_ in unanswerable_lex if s_ <= 0.0)
    print()
    print(f"Lexical-support gate (BM25 > 0) would refuse "
          f"{u_lex_zero}/{len(unanswerable_lex)} unanswerable and "
          f"{a_lex_zero}/{len(lexical_populations['answerable'])} answerable queries.")

    print()
    print(f"answerable p05 = {a_p05:.3f}   unanswerable p95 = {u_p95:.3f}")
    if separated:
        print("Populations separate cleanly — a threshold between them is meaningful.")
    else:
        print("WARNING: populations OVERLAP. Any single threshold will both refuse good "
              "questions and admit unanswerable ones; the numbers below are the least-bad "
              "compromise, not a clean boundary.")

    print(f"\nRecommended min_top_score = {recommendation['threshold']}")
    print(f"  answerable pass rate     = {recommendation.get('answerable_pass_rate', 0):.1%}")
    print(f"  unanswerable block rate  = {recommendation.get('unanswerable_block_rate', 0):.1%}")
    print(f"  balanced accuracy        = {recommendation['balanced_accuracy']:.1%}")

    payload = {
        "date": date.today().isoformat(),
        "index": str(args.index),
        "index_size": vector_index.size,
        "embed_model": retriever.embedder.model_name,
        "populations": summaries,
        "lexical": {
            name: {
                "n": len(scores),
                "min": round(min(scores), 3) if scores else None,
                "median": round(percentile(scores, 50), 3) if scores else None,
                "max": round(max(scores), 3) if scores else None,
                "zero_scoring": sum(1 for s_ in scores if s_ <= 0.0),
            }
            for name, scores in lexical_populations.items() if scores
        },
        "answerable_p05": round(a_p05, 4),
        "unanswerable_p95": round(u_p95, 4),
        "populations_separate": bool(separated),
        "recommendation": recommendation,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
