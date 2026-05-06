"""
Variance analysis for latency over N repeated runs.

Aggregates events by (config, query_id) across run_iteration
and computes mean ± stddev per phase.
"""

import math
from collections import defaultdict
from typing import Dict, List


def compute_variance(events: List[dict]) -> dict:
    """Compute per-phase mean ± stddev over N iterations per query.

    Groups events by query_id, then computes variance across iterations.

    Returns:
        {
            "per_query": {query_id: {phase: {mean, stddev, values}}},
            "aggregate": {phase: {mean, stddev, n_queries}},
        }
    """
    # Group by query_id
    by_query: Dict[str, List[dict]] = defaultdict(list)
    for ev in events:
        qid = ev.get("query_id", "?")
        by_query[qid].append(ev)

    per_query = {}
    phase_aggregates: Dict[str, List[float]] = defaultdict(list)

    for qid, q_events in sorted(by_query.items()):
        if len(q_events) < 2:
            continue

        # Collect phase values across iterations
        phase_vals: Dict[str, List[float]] = defaultdict(list)
        for ev in q_events:
            for phase, ms in ev.get("phases", {}).items():
                if isinstance(ms, (int, float)):
                    phase_vals[phase].append(ms)

        query_stats = {}
        for phase, vals in phase_vals.items():
            mean = sum(vals) / len(vals)
            stddev = (
                math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))
                if len(vals) > 1 else 0.0
            )
            query_stats[phase] = {
                "mean": round(mean, 2),
                "stddev": round(stddev, 2),
                "values": [round(v, 2) for v in vals],
                "n": len(vals),
            }
            phase_aggregates[phase].append(mean)

        per_query[qid] = query_stats

    # Aggregate across queries
    aggregate = {}
    for phase, means in phase_aggregates.items():
        overall_mean = sum(means) / len(means) if means else 0
        overall_std = (
            math.sqrt(sum((x - overall_mean) ** 2 for x in means) / (len(means) - 1))
            if len(means) > 1 else 0.0
        )
        aggregate[phase] = {
            "mean": round(overall_mean, 2),
            "stddev": round(overall_std, 2),
            "n_queries": len(means),
        }

    return {
        "per_query": per_query,
        "aggregate": aggregate,
    }
