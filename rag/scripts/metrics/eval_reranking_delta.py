"""
Reranking Impact Measurement Script.

Runs the retrieval evaluation harness twice — once in retrieval-only mode
and once with the full pipeline (preprocessor + reranker) — then computes
the delta in Hit Rate@k, MRR, and Precision@k.

WHY THIS MATTERS FOR A CV:
  A measurable MRR lift (e.g. +15%) from adding a cross-encoder reranker
  directly proves that the reranking stage adds real retrieval quality,
  not just complexity.  This is the single most defensible IR metric.

Usage:
    python -m scripts.metrics.eval_reranking_delta --config config/proyectos_docentes.yaml
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tests.evaluation.eval_retrieval import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser(
        description="Measure reranking impact on retrieval quality."
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
        "--top-k",
        type=int,
        default=None,
        help="Override retrieval top_k",
    )
    args = parser.parse_args()

    # ── Run 1: retrieval-only (no preprocessor, no reranker) ──
    print("=" * 70)
    print("RUN 1: Retrieval-Only (no reranker, no preprocessor)")
    print("=" * 70)
    report_base = evaluate_retrieval(
        config_path=args.config,
        dataset_path=args.dataset,
        top_k_override=args.top_k,
        full_pipeline=False,
    )

    # ── Run 2: full pipeline (preprocessor + reranker) ──
    print("\n" + "=" * 70)
    print("RUN 2: Full Pipeline (preprocessor + reranker)")
    print("=" * 70)
    report_full = evaluate_retrieval(
        config_path=args.config,
        dataset_path=args.dataset,
        top_k_override=args.top_k,
        full_pipeline=True,
    )

    # ── Compute deltas ──
    def _delta(a, b):
        return b - a

    def _pct(a, b):
        if a == 0:
            return float("inf") if b > 0 else 0.0
        return ((b - a) / a) * 100

    metrics = [
        ("Hit Rate@k", report_base.hit_rate, report_full.hit_rate),
        ("MRR", report_base.mrr, report_full.mrr),
        ("Mean Precision@k", report_base.mean_precision, report_full.mean_precision),
    ]

    # ── Print summary table ──
    print("\n")
    print("=" * 70)
    print("RERANKING IMPACT SUMMARY")
    print("=" * 70)
    print(
        f"{'Metric':<20} {'Retrieval-Only':>16} {'Full Pipeline':>16} "
        f"{'Δ':>8} {'Δ%':>8}"
    )
    print("-" * 70)
    for name, base_val, full_val in metrics:
        d = _delta(base_val, full_val)
        p = _pct(base_val, full_val)
        sign = "+" if d >= 0 else ""
        print(
            f"{name:<20} {base_val:>16.3f} {full_val:>16.3f} "
            f"{sign}{d:>7.3f} {sign}{p:>6.1f}%"
        )

    # ── Per-query comparison (misses recovered) ──
    base_misses = {r.query_id for r in report_base.results if not r.hit}
    full_misses = {r.query_id for r in report_full.results if not r.hit}
    recovered = base_misses - full_misses
    lost = full_misses - base_misses

    if recovered:
        print(f"\n  Misses RECOVERED by reranking ({len(recovered)}):")
        for qid in sorted(recovered):
            r = next(r for r in report_base.results if r.query_id == qid)
            print(f"    {qid}: {r.query[:60]}")

    if lost:
        print(f"\n  Hits LOST after reranking ({len(lost)}):")
        for qid in sorted(lost):
            r = next(r for r in report_full.results if r.query_id == qid)
            print(f"    {qid}: {r.query[:60]}")

    # ── Markdown output for report ──
    print("\n\n### Markdown (copy-paste for report)")
    print("```")
    print(f"| Metric | Retrieval-Only | + Reranker | Δ | Δ% |")
    print(f"|--------|---------------|-----------|---|-----|")
    for name, base_val, full_val in metrics:
        d = _delta(base_val, full_val)
        p = _pct(base_val, full_val)
        sign = "+" if d >= 0 else ""
        print(
            f"| {name} | {base_val:.3f} | {full_val:.3f} | {sign}{d:.3f} | {sign}{p:.1f}% |"
        )
    print("```")

    # ── Timing comparison ──
    base_avg = report_base.elapsed_total_s / max(len(report_base.results), 1)
    full_avg = report_full.elapsed_total_s / max(len(report_full.results), 1)
    print(f"\nAvg time/query: retrieval-only={base_avg:.2f}s, full={full_avg:.2f}s")


if __name__ == "__main__":
    main()
