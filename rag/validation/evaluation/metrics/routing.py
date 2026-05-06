"""
Routing metrics: accuracy, per-class P/R/F1, confusion matrix,
and confidence distribution split by correct/incorrect predictions.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional


# Ground-truth mapping: dataset query type → router source label
_LABEL_TO_SOURCE = {
    "rag": "unstructured",
    "sql": "structured",
    "hybrid": "hybrid",
    "negative": "unstructured",
    "out_of_domain": "unstructured",
}


def compute_routing_metrics(
    per_query: List[dict],
) -> dict:
    """Compute routing metrics from per-query routing results.

    Args:
        per_query: list of dicts, each with keys:
            id, type, partition, truth, predicted, confidence, correct

    Returns:
        Dict with accuracy, per_class metrics, confusion_matrix,
        and confidence distributions.
    """
    n = len(per_query)
    if n == 0:
        return {"accuracy": 0.0, "n_queries": 0}

    correct = sum(1 for r in per_query if r["correct"])

    # ── Confusion matrix ──
    labels = sorted(set(_LABEL_TO_SOURCE.values()))
    matrix: Dict[str, Counter] = defaultdict(Counter)
    for r in per_query:
        matrix[r["truth"]][r["predicted"]] += 1

    # ── Per-class P/R/F1 ──
    per_class = {}
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
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # ── Confidence distribution: correct vs incorrect ──
    correct_confs = [r["confidence"] for r in per_query if r["correct"]]
    incorrect_confs = [r["confidence"] for r in per_query if not r["correct"]]

    def _stats(vals):
        if not vals:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0, "values": []}
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "count": len(vals),
            "values": vals,
        }

    # ── By partition ──
    by_partition = {}
    partitions = sorted(set(r.get("partition", "") for r in per_query))
    for part in partitions:
        if not part:
            continue
        subset = [r for r in per_query if r.get("partition") == part]
        n_part = len(subset)
        c_part = sum(1 for r in subset if r["correct"])
        by_partition[part] = {
            "accuracy": c_part / n_part if n_part > 0 else 0.0,
            "correct": c_part,
            "n_queries": n_part,
        }

    return {
        "accuracy": correct / n,
        "correct": correct,
        "n_queries": n,
        "per_class": per_class,
        "confusion_matrix": {k: dict(v) for k, v in matrix.items()},
        "confidence": {
            "correct": _stats(correct_confs),
            "incorrect": _stats(incorrect_confs),
            "overall": _stats(correct_confs + incorrect_confs),
        },
        "by_partition": by_partition,
    }


def get_routing_errors(per_query: List[dict]) -> List[dict]:
    """Return all misrouted queries with details."""
    return [
        {
            "id": r["id"],
            "query": r.get("query", ""),
            "type": r["type"],
            "partition": r.get("partition", ""),
            "truth": r["truth"],
            "predicted": r["predicted"],
            "confidence": r["confidence"],
        }
        for r in per_query
        if not r["correct"]
    ]


def get_ground_truth_source(query_type: str) -> str:
    """Map dataset query type to expected router source label."""
    return _LABEL_TO_SOURCE.get(query_type, "unstructured")
