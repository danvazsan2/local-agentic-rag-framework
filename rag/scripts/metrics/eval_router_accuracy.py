"""
Router Accuracy Evaluation Script.

Routes all queries in the evaluation dataset through the QueryRouter and
compares the predicted source type against the ground-truth `type` labels.
Computes accuracy, per-class precision/recall, and a confusion matrix.

WHY THIS MATTERS FOR A CV:
  A high routing accuracy (>90%) on a labeled dataset proves the 3-layer
  routing architecture correctly directs queries to the right data source,
  avoiding wasted LLM/retrieval calls and improving user experience.

Usage:
    python -m scripts.metrics.eval_router_accuracy --config config/proyectos_docentes.yaml
    python -m scripts.metrics.eval_router_accuracy --config config/proyectos_docentes.yaml --dry-run
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Type mapping ──────────────────────────────────────────────────
# eval_dataset.json uses: rag, sql, hybrid, negative
# The router returns: unstructured, structured, hybrid

_LABEL_TO_SOURCE = {
    "rag": "unstructured",
    "sql": "structured",
    "hybrid": "hybrid",
    "negative": "unstructured",  # negative queries have no DB answer → RAG
}

_ALL_SOURCES = ["unstructured", "structured", "hybrid"]


def _load_dataset(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_live(config_path: str, dataset: List[dict]) -> List[Tuple[str, str, str, float]]:
    """Route each query through the real router and return (qid, truth, pred, conf)."""
    from rag_framework import RAGFramework

    rag = RAGFramework.from_yaml(config_path)
    rag.load_index()

    # Build the hybrid engine to get the router
    rag._query_ops._ensure_query_engine()
    rag._query_ops._ensure_hybrid_engine()
    hybrid_engine = rag._hybrid_engine

    results = []
    for item in dataset:
        qid = item["id"]
        query = item["query"]
        truth = _LABEL_TO_SOURCE.get(item["type"], "unstructured")

        routing = hybrid_engine.router.route(query)
        pred = routing.source.value
        conf = routing.confidence

        results.append((qid, truth, pred, conf))

    return results


def _run_dry(dataset: List[dict]) -> List[Tuple[str, str, str, float]]:
    """Generate plausible placeholder results for dry-run mode."""
    import random
    random.seed(42)

    results = []
    for item in dataset:
        qid = item["id"]
        truth = _LABEL_TO_SOURCE.get(item["type"], "unstructured")
        # Simulate ~88% accuracy
        if random.random() < 0.88:
            pred = truth
        else:
            pred = random.choice([s for s in _ALL_SOURCES if s != truth])
        conf = round(random.uniform(0.70, 0.99), 2)
        results.append((qid, truth, pred, conf))
    return results


def _print_results(results: List[Tuple[str, str, str, float]], dry_run: bool, errors_only: bool = False):
    """Print accuracy, confusion matrix, per-class stats, and confidence distribution."""

    if dry_run:
        print("\n⚠️  DRY-RUN MODE — values are simulated placeholders\n")

    n = len(results)
    correct = sum(1 for _, t, p, _ in results if t == p)
    accuracy = correct / n if n else 0

    # ── Per-query table ──
    if errors_only:
        errors = [(qid, t, p, c) for qid, t, p, c in results if t != p]
        print(f"Showing {len(errors)} MISCLASSIFIED queries (out of {n}):\n")
        display = errors
    else:
        display = results

    print(f"{'ID':<14} {'Truth':<14} {'Predicted':<14} {'Conf':>6} {'✓':>3}")
    print("-" * 55)
    for qid, truth, pred, conf in display:
        ok = "✓" if truth == pred else "✗"
        print(f"{qid:<14} {truth:<14} {pred:<14} {conf:>6.2f} {ok:>3}")

    # ── Confusion matrix ──
    matrix: Dict[str, Counter] = defaultdict(Counter)
    for _, truth, pred, _ in results:
        matrix[truth][pred] += 1

    labels = sorted({t for _, t, _, _ in results} | {p for _, _, p, _ in results})

    print("\n" + "=" * 55)
    print("CONFUSION MATRIX")
    print("=" * 55)
    header = f"{'True \\ Pred':<14}" + "".join(f"{l:>14}" for l in labels)
    print(header)
    print("-" * len(header))
    for true_label in labels:
        row = f"{true_label:<14}" + "".join(
            f"{matrix[true_label][pred_label]:>14}" for pred_label in labels
        )
        print(row)

    # ── Per-class precision / recall ──
    print("\n" + "=" * 55)
    print("PER-CLASS METRICS")
    print("=" * 55)
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)

    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        print(
            f"{label:<14} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10}"
        )

    print(f"\nOverall accuracy: {accuracy:.2%} ({correct}/{n})")

    # ── Confidence distribution ──
    confs = [c for _, _, _, c in results]
    print(f"\nConfidence distribution: min={min(confs):.2f}, "
          f"max={max(confs):.2f}, mean={sum(confs)/len(confs):.2f}")

    # ── Markdown ──
    print("\n\n### Markdown (copy-paste for report)")
    print("```")
    print(f"| Class | Precision | Recall | F1 | Support |")
    print(f"|-------|-----------|--------|-----|---------|")
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        print(f"| {label} | {precision:.2f} | {recall:.2f} | {f1:.2f} | {support} |")
    print(f"\nOverall accuracy: **{accuracy:.0%}** ({correct}/{n})")
    print("```")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate query router accuracy against labeled dataset."
    )
    parser.add_argument(
        "--config",
        default="config/proyectos_docentes.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--dataset",
        default="tests/evaluation/eval_router_dataset.json",
        help="Path to router eval dataset JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use simulated results instead of live routing",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only print misclassified queries in the detail table",
    )
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} queries from {args.dataset}\n")

    if args.dry_run:
        results = _run_dry(dataset)
    else:
        results = _run_live(args.config, dataset)

    _print_results(results, dry_run=args.dry_run, errors_only=args.errors_only)


if __name__ == "__main__":
    main()

