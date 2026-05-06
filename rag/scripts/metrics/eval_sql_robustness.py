"""
SQL Robustness Evaluation Script.

Runs the SQL and hybrid queries from the eval dataset through the SQLAgent,
recording first-attempt success rate, retry/relaxation counts, validator
rejection rate, and per-query timing.

WHY THIS MATTERS FOR A CV:
  "X% of NL2SQL queries succeed on first attempt with Y% requiring
  automatic query relaxation" proves robustness in a notoriously fragile
  pipeline (LLM-generated SQL).  Combined with the SQL injection
  detection stats, this shows defense-in-depth.

Usage:
    python -m scripts.metrics.eval_sql_robustness --config config/proyectos_docentes.yaml
    python -m scripts.metrics.eval_sql_robustness --config config/proyectos_docentes.yaml --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_sql_queries(dataset_path: str) -> List[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return [q for q in dataset if q.get("type") in ("sql", "hybrid")]


def _run_live(config_path: str, queries: List[dict]):
    """Run SQL queries through the live SQLAgent."""
    from rag_framework import RAGFramework

    rag = RAGFramework.from_yaml(config_path)
    rag.load_index()
    rag._query_ops._ensure_query_engine()
    rag._query_ops._ensure_hybrid_engine()

    sql_agent = rag._hybrid_engine.sql_agent

    results = []
    for item in queries:
        qid = item["id"]
        query = item["query"]
        query_type = item["type"]

        sql_result = sql_agent.query(query)

        results.append({
            "id": qid,
            "query": query[:60],
            "type": query_type,
            "success": sql_result.success,
            "attempts": sql_result.generation_attempts,
            "relaxed": sql_result.query_relaxed,
            "time_ms": sql_result.total_time_ms,
            "error": sql_result.error,
            "sql": sql_result.query[:80] if sql_result.query else "",
        })

    return results


def _run_dry(queries: List[dict]):
    """Generate plausible placeholder results."""
    import random
    random.seed(42)

    results = []
    for item in queries:
        qid = item["id"]
        success = random.random() < 0.85
        attempts = 1 if success and random.random() < 0.75 else random.randint(2, 3)
        relaxed = success and random.random() < 0.20
        time_ms = random.uniform(300, 2500)

        results.append({
            "id": qid,
            "query": item["query"][:60],
            "type": item["type"],
            "success": success,
            "attempts": attempts,
            "relaxed": relaxed,
            "time_ms": time_ms,
            "error": None if success else "Simulated error",
            "sql": "SELECT ... (placeholder)" if success else "",
        })

    return results


def _print_results(results: List[dict], dry_run: bool):
    if dry_run:
        print("\n⚠️  DRY-RUN MODE — values are simulated placeholders\n")

    n = len(results)
    if n == 0:
        print("No SQL/hybrid queries found in dataset.")
        return

    # ── Per-query table ──
    print(f"{'ID':<12} {'Type':<8} {'OK':>3} {'Att':>4} {'Relax':>6} "
          f"{'Time(ms)':>10}  Query")
    print("-" * 80)
    for r in results:
        ok = "✓" if r["success"] else "✗"
        relax = "yes" if r["relaxed"] else "-"
        print(
            f"{r['id']:<12} {r['type']:<8} {ok:>3} {r['attempts']:>4} "
            f"{relax:>6} {r['time_ms']:>10.0f}  {r['query']}"
        )

    # ── Aggregate stats ──
    successes = sum(1 for r in results if r["success"])
    first_attempt = sum(1 for r in results if r["success"] and r["attempts"] == 1)
    relaxed = sum(1 for r in results if r["relaxed"])
    failed = n - successes
    times = [r["time_ms"] for r in results if r["success"]]

    print("\n" + "=" * 60)
    print("SQL ROBUSTNESS SUMMARY")
    print("=" * 60)
    print(f"  Total queries         : {n}")
    print(f"  Success rate          : {successes}/{n} ({successes/n:.0%})")
    print(f"  First-attempt success : {first_attempt}/{n} ({first_attempt/n:.0%})")
    print(f"  Required relaxation   : {relaxed}/{n} ({relaxed/n:.0%})")
    print(f"  Failed                : {failed}/{n} ({failed/n:.0%})")
    if times:
        times_sorted = sorted(times)
        p50 = times_sorted[len(times_sorted) // 2]
        p95_idx = min(int(len(times_sorted) * 0.95), len(times_sorted) - 1)
        p95 = times_sorted[p95_idx]
        print(f"  Latency p50           : {p50:.0f}ms")
        print(f"  Latency p95           : {p95:.0f}ms")

    # ── Markdown ──
    print("\n\n### Markdown (copy-paste for report)")
    print("```")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Success rate | {successes/n:.0%} ({successes}/{n}) |")
    print(f"| First-attempt success | {first_attempt/n:.0%} ({first_attempt}/{n}) |")
    print(f"| Required relaxation | {relaxed/n:.0%} ({relaxed}/{n}) |")
    if times:
        print(f"| Latency p50 | {p50:.0f}ms |")
        print(f"| Latency p95 | {p95:.0f}ms |")
    print("```")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NL2SQL robustness and reliability."
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

    queries = _load_sql_queries(args.dataset)
    print(f"Loaded {len(queries)} SQL/hybrid queries from {args.dataset}")

    if args.dry_run:
        results = _run_dry(queries)
    else:
        results = _run_live(args.config, queries)

    _print_results(results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
