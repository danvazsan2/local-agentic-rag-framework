"""
Latency metrics: per-phase statistics with mean, median, p95, stddev.
"""

import math
from collections import defaultdict
from typing import Dict, List

PHASE_ORDER = [
    "preprocessor_ms", "vector_ms", "bm25_ms", "rrf_ms",
    "retrieval_ms", "reranker_ms", "router_keyword_ms",
    "router_llm_ms", "crag_ms", "synthesis_ms", "total_ms",
]


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


def _stddev(data: List[float]) -> float:
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    return math.sqrt(sum((x - mean) ** 2 for x in data) / (len(data) - 1))


def _stats(data: List[float]) -> dict:
    if not data:
        return {"mean": 0, "median": 0, "p95": 0, "stddev": 0, "count": 0}
    return {
        "mean": round(sum(data) / len(data), 2),
        "median": round(_percentile(data, 50), 2),
        "p95": round(_percentile(data, 95), 2),
        "stddev": round(_stddev(data), 2),
        "count": len(data),
        "min": round(min(data), 2),
        "max": round(max(data), 2),
    }


def compute_latency_by_phase(events: List[dict]) -> Dict[str, dict]:
    """Per-phase latency statistics across events."""
    vals: Dict[str, List[float]] = defaultdict(list)
    for ev in events:
        for phase, ms in ev.get("phases", {}).items():
            if isinstance(ms, (int, float)):
                vals[phase].append(ms)
    result = {}
    for p in PHASE_ORDER:
        if p in vals:
            result[p] = _stats(vals[p])
    for p in sorted(vals):
        if p not in result:
            result[p] = _stats(vals[p])
    return result


def compute_latency_by_type(events, dataset_index) -> Dict[str, dict]:
    """Total latency statistics grouped by query type."""
    by_type: Dict[str, List[float]] = defaultdict(list)
    for ev in events:
        t = ev.get("phases", {}).get("total_ms")
        if t is None:
            continue
        qtype = dataset_index.get(ev.get("query_id", ""), {}).get("type", "?")
        by_type[qtype].append(t)
    return {k: _stats(v) for k, v in sorted(by_type.items())}


def compute_phase_percentages(events: List[dict]) -> Dict[str, float]:
    """Average % of total time spent in each phase."""
    sums: Dict[str, float] = defaultdict(float)
    total = 0.0
    for ev in events:
        phases = ev.get("phases", {})
        t = phases.get("total_ms")
        if not t or t <= 0:
            continue
        total += t
        for p, ms in phases.items():
            if p != "total_ms" and isinstance(ms, (int, float)):
                sums[p] += ms
    if total <= 0:
        return {}
    return {p: round(ms / total * 100, 2) for p, ms in sorted(sums.items(), key=lambda x: -x[1])}
