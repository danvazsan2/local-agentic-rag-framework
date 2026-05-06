"""
NL2SQL metrics: success rate, first-attempt rate, retry distribution,
relaxation rate, and per-attempt latencies.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional


def compute_sql_metrics(events: List[dict]) -> dict:
    """Compute SQL robustness metrics from events with SQL trace data.

    Args:
        events: list of event dicts that have 'sql' field populated

    Returns:
        Dict with success_rate, first_attempt_rate, relaxation_rate,
        retry_distribution, and per-query details.
    """
    sql_events = [ev for ev in events if ev.get("sql")]
    n = len(sql_events)
    if n == 0:
        return {"n_queries": 0}

    per_query = []
    retry_counts = []

    for ev in sql_events:
        sql = ev["sql"]
        success = sql.get("success", False)
        attempts = sql.get("total_attempts", 0)
        relaxed = sql.get("query_relaxed", False)
        total_ms = sql.get("total_ms", 0)

        per_query.append({
            "id": ev.get("query_id", "?"),
            "partition": ev.get("_partition", ev.get("partition", "")),
            "success": success,
            "attempts": attempts,
            "relaxed": relaxed,
            "total_ms": total_ms,
        })
        retry_counts.append(attempts)

    successes = sum(1 for q in per_query if q["success"])
    first_attempt = sum(1 for q in per_query if q["success"] and q["attempts"] == 1)
    relaxed = sum(1 for q in per_query if q["relaxed"])
    times = [q["total_ms"] for q in per_query if q["success"] and q["total_ms"] > 0]

    retry_dist = dict(Counter(retry_counts).most_common())

    return {
        "n_queries": n,
        "success_rate": round(successes / n, 4) if n else 0,
        "first_attempt_rate": round(first_attempt / n, 4) if n else 0,
        "relaxation_rate": round(relaxed / n, 4) if n else 0,
        "retry_distribution": retry_dist,
        "latency_ms": {
            "mean": round(sum(times) / len(times), 2) if times else 0,
            "p50": _percentile(times, 50),
            "p95": _percentile(times, 95),
        },
        "per_query": per_query,
    }


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]
