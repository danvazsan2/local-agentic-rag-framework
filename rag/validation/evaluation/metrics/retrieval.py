"""
Retrieval quality metrics: HR@k, MRR@k, P@k, NDCG@k.

Computes per-query and aggregate metrics with partition-aware breakdowns.
Operates on raw retrieval results (list of filenames + expected patterns).
"""

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────
# Per-query metric functions
# ──────────────────────────────────────────────────────────────

def hit_at_k(files: List[str], pattern: str, k: int) -> bool:
    """True if any retrieved file at rank ≤ k matches the expected pattern."""
    r = re.compile(pattern, re.IGNORECASE)
    return any(r.search(f) for f in files[:k])


def reciprocal_rank(files: List[str], pattern: str, k: int) -> float:
    """Reciprocal rank of the first relevant file within top-k (0 if none)."""
    r = re.compile(pattern, re.IGNORECASE)
    for i, f in enumerate(files[:k], 1):
        if r.search(f):
            return 1.0 / i
    return 0.0


def precision_at_k(files: List[str], pattern: str, k: int) -> float:
    """Fraction of top-k retrieved files that match the expected pattern."""
    subset = files[:k]
    if not subset:
        return 0.0
    r = re.compile(pattern, re.IGNORECASE)
    return sum(1 for f in subset if r.search(f)) / len(subset)


def ndcg_at_k(files: List[str], pattern: str, k: int) -> float:
    """NDCG@k with binary relevance (1 if file matches pattern, 0 otherwise)."""
    r = re.compile(pattern, re.IGNORECASE)
    subset = files[:k]
    dcg = sum(
        (1.0 if r.search(f) else 0.0) / math.log2(i + 1)
        for i, f in enumerate(subset, 1)
    )
    if dcg == 0:
        return 0.0
    n_rel = sum(1 for f in subset if r.search(f))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# Per-query result container
# ──────────────────────────────────────────────────────────────

def compute_per_query(
    query_id: str,
    query: str,
    pattern: str,
    files: List[str],
    k_values: List[int],
    partition: str = "",
    difficulty: str = "",
    query_type: str = "",
) -> dict:
    """Compute all retrieval metrics for one query."""
    return {
        "id": query_id,
        "query": query,
        "partition": partition,
        "difficulty": difficulty,
        "type": query_type,
        "pattern": pattern,
        "files": files,
        "hit_at_k": {k: hit_at_k(files, pattern, k) for k in k_values},
        "rr_at_k": {k: reciprocal_rank(files, pattern, k) for k in k_values},
        "precision_at_k": {k: precision_at_k(files, pattern, k) for k in k_values},
        "ndcg_at_k": {k: ndcg_at_k(files, pattern, k) for k in k_values},
    }


# ──────────────────────────────────────────────────────────────
# Aggregation with partition support
# ──────────────────────────────────────────────────────────────

def aggregate_metrics(
    per_query: List[dict],
    k_values: List[int],
    partition_filter: Optional[str] = None,
) -> dict:
    """Aggregate retrieval metrics over a list of per-query results.

    Args:
        per_query: list of dicts from compute_per_query()
        k_values: k values to report
        partition_filter: if set, only include queries from this partition

    Returns:
        Dict with n_queries and metric averages per k.
    """
    if partition_filter:
        subset = [q for q in per_query if q["partition"] == partition_filter]
    else:
        subset = per_query

    n = len(subset)
    if n == 0:
        return {"n_queries": 0}

    agg = {"n_queries": n}
    for k in k_values:
        agg[f"hr@{k}"] = sum(q["hit_at_k"][k] for q in subset) / n
        agg[f"mrr@{k}"] = sum(q["rr_at_k"][k] for q in subset) / n
        agg[f"p@{k}"] = sum(q["precision_at_k"][k] for q in subset) / n
        agg[f"ndcg@{k}"] = sum(q["ndcg_at_k"][k] for q in subset) / n

    return agg


def aggregate_by_partition(
    per_query: List[dict],
    k_values: List[int],
) -> Dict[str, dict]:
    """Aggregate metrics split by partition (overall + per partition).

    Returns:
        {"overall": {...}, "well_formed": {...}, "adversarial": {...}}
    """
    result = {
        "overall": aggregate_metrics(per_query, k_values),
        "well_formed": aggregate_metrics(per_query, k_values, "well_formed"),
        "adversarial": aggregate_metrics(per_query, k_values, "adversarial"),
    }
    return result


def get_failures(
    per_query: List[dict],
    k: int = 10,
    partition_filter: Optional[str] = None,
) -> List[dict]:
    """Return queries where HR@k == 0 (retrieval failures).

    Each entry includes id, query, expected pattern, and top-3 retrieved docs.
    """
    subset = per_query
    if partition_filter:
        subset = [q for q in subset if q["partition"] == partition_filter]

    failures = []
    for q in subset:
        if not q["hit_at_k"].get(k, False):
            failures.append({
                "id": q["id"],
                "query": q["query"],
                "partition": q["partition"],
                "difficulty": q["difficulty"],
                "expected_pattern": q["pattern"],
                "top_3_retrieved": q["files"][:3],
                "cause": "",  # to be annotated manually
            })
    return failures


def format_table(aggregates: dict, k_values: List[int], label: str = "") -> str:
    """Format aggregated metrics as a console table."""
    lines = []
    if label:
        lines.append(f"  {label} (n={aggregates['n_queries']})")
    lines.append(f"  {'k':>3}  {'HR@k':>7}  {'MRR@k':>7}  {'P@k':>7}  {'NDCG@k':>7}")
    lines.append(f"  {'-' * 35}")
    for k in k_values:
        lines.append(
            f"  {k:>3}  {aggregates.get(f'hr@{k}', 0):>7.3f}  "
            f"{aggregates.get(f'mrr@{k}', 0):>7.3f}  "
            f"{aggregates.get(f'p@{k}', 0):>7.3f}  "
            f"{aggregates.get(f'ndcg@{k}', 0):>7.3f}"
        )
    return "\n".join(lines)
