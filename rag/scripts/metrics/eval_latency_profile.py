"""
End-to-End Latency Profiling Script.

Runs all queries from the eval dataset through the full HybridQueryEngine
pipeline, capturing per-query latency bucketed by type (RAG, SQL, hybrid).
Reports p50, p95, mean, and min/max per category.

WHY THIS MATTERS FOR A CV:
  "p50 latency of Xs for RAG queries and Ys for SQL queries" shows that
  the system is production-viable.  Latency targeting is table-stakes for
  any system targeting real users.

Usage:
    python -m scripts.metrics.eval_latency_profile --config config/proyectos_docentes.yaml
    python -m scripts.metrics.eval_latency_profile --config config/proyectos_docentes.yaml --dry-run
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_dataset(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def _run_live(config_path: str, queries: List[dict]):
    """Run all queries through the live HybridQueryEngine."""
    import logging
    logging.basicConfig(level=logging.WARNING)

    from rag_framework import RAGFramework

    rag = RAGFramework.from_yaml(config_path)
    rag.load_index()
    rag._query_ops._ensure_query_engine()
    rag._query_ops._ensure_hybrid_engine()

    hybrid_engine = rag._hybrid_engine

    results = []
    for item in queries:
        qid = item["id"]
        query = item["query"]
        qtype = item["type"]

        t0 = time.time()
        try:
            response = hybrid_engine.query(query)
            success = True
            total_time_ms = response.total_time_ms
        except Exception as e:
            success = False
            total_time_ms = (time.time() - t0) * 1000

        elapsed_s = total_time_ms / 1000

        results.append({
            "id": qid,
            "type": qtype,
            "query": query[:50],
            "success": success,
            "elapsed_s": elapsed_s,
        })

    return results


def _run_dry(queries: List[dict]):
    """Generate plausible placeholder latency results."""
    import random
    random.seed(42)

    # Realistic latency profiles per type
    profiles = {
        "rag": (1.5, 4.5),       # slower: retrieval + reranking + CRAG + synthesis
        "sql": (0.4, 2.0),       # faster: NL2SQL + execution + synthesis
        "hybrid": (2.0, 6.0),    # slowest: parallel SQL+RAG + hybrid synthesis
        "negative": (1.0, 3.5),  # similar to RAG
    }

    results = []
    for item in queries:
        qtype = item["type"]
        lo, hi = profiles.get(qtype, (1.0, 4.0))
        elapsed = random.uniform(lo, hi)

        results.append({
            "id": item["id"],
            "type": qtype,
            "query": item["query"][:50],
            "success": True,
            "elapsed_s": elapsed,
        })

    return results


def _print_results(results: List[dict], dry_run: bool):
    if dry_run:
        print("\n⚠️  DRY-RUN MODE — values are simulated placeholders\n")

    n = len(results)
    if n == 0:
        print("No queries to profile.")
        return

    # ── Per-query table ──
    print(f"{'ID':<14} {'Type':<10} {'Time(s)':>8} {'OK':>3}  Query")
    print("-" * 80)
    for r in results:
        ok = "✓" if r["success"] else "✗"
        print(
            f"{r['id']:<14} {r['type']:<10} {r['elapsed_s']:>8.2f} {ok:>3}  {r['query']}"
        )

    # ── Bucket by type ──
    by_type: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        if r["success"]:
            by_type[r["type"]].append(r["elapsed_s"])

    all_times = [r["elapsed_s"] for r in results if r["success"]]

    print("\n" + "=" * 70)
    print("LATENCY PROFILE SUMMARY")
    print("=" * 70)
    print(
        f"{'Type':<12} {'Count':>6} {'p50(s)':>8} {'p95(s)':>8} "
        f"{'Mean(s)':>8} {'Min(s)':>8} {'Max(s)':>8}"
    )
    print("-" * 70)

    rows = []
    for qtype in sorted(by_type.keys()):
        times = by_type[qtype]
        count = len(times)
        p50 = _percentile(times, 50)
        p95 = _percentile(times, 95)
        mean = sum(times) / count
        mn = min(times)
        mx = max(times)
        rows.append((qtype, count, p50, p95, mean, mn, mx))
        print(
            f"{qtype:<12} {count:>6} {p50:>8.2f} {p95:>8.2f} "
            f"{mean:>8.2f} {mn:>8.2f} {mx:>8.2f}"
        )

    # Overall
    if all_times:
        print("-" * 70)
        p50 = _percentile(all_times, 50)
        p95 = _percentile(all_times, 95)
        mean = sum(all_times) / len(all_times)
        print(
            f"{'OVERALL':<12} {len(all_times):>6} {p50:>8.2f} {p95:>8.2f} "
            f"{mean:>8.2f} {min(all_times):>8.2f} {max(all_times):>8.2f}"
        )

    # ── Markdown ──
    print("\n\n### Markdown (copy-paste for report)")
    print("```")
    print(f"| Query Type | Count | p50 (s) | p95 (s) | Mean (s) |")
    print(f"|------------|-------|---------|---------|----------|")
    for qtype, count, p50, p95, mean, mn, mx in rows:
        print(f"| {qtype} | {count} | {p50:.2f} | {p95:.2f} | {mean:.2f} |")
    if all_times:
        p50 = _percentile(all_times, 50)
        p95 = _percentile(all_times, 95)
        mean_all = sum(all_times) / len(all_times)
        print(f"| **Overall** | {len(all_times)} | {p50:.2f} | {p95:.2f} | {mean_all:.2f} |")
    print("```")


def main():
    parser = argparse.ArgumentParser(
        description="Profile end-to-end query latency by type."
    )
    parser.add_argument(
        "--config",
        default="config/proyectos_docentes.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--dataset",
        default="tests/evaluation/eval_dataset.json",
        help="Path to eval dataset JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use simulated results (no LLM/DB required)",
    )
    args = parser.parse_args()

    queries = _load_dataset(args.dataset)
    print(f"Loaded {len(queries)} queries from {args.dataset}")

    if args.dry_run:
        results = _run_dry(queries)
    else:
        results = _run_live(args.config, queries)

    _print_results(results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
