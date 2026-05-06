"""
Instrumentation module for per-phase query tracing.

Usage:
    trace = QueryTrace(query_id="r01", run_id="eval_v1", query="...")
    with phase_timer(trace, "retrieval_ms"):
        nodes = retriever.retrieve(query)
    write_event(run_dir, trace)

All trace parameters are Optional — if trace is None every helper is a no-op,
so downstream code can safely call these functions without guarding.
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QueryTrace:
    """Full instrumentation record for one query execution."""

    query_id: str
    run_id: str
    query: str
    configuration: str = "full_pipeline"
    run_iteration: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Phase timings (name → ms).  Populated via phase_timer().
    phases: Dict[str, float] = field(default_factory=dict)

    # Routing decision dict (source, confidence, method, reasoning).
    route: Optional[Dict[str, Any]] = None

    # CRAG metadata dict (relevance_ratio, rewrite_triggered, grading_per_chunk_ms, …).
    crag: Optional[Dict[str, Any]] = None

    # SQL metadata dict (attempts list, total_attempts, total_ms, …).
    sql: Optional[Dict[str, Any]] = None

    # Filenames / ids of retrieved documents (populated by query_engine).
    retrieved_docs: List[Any] = field(default_factory=list)

    # Final synthesized response string.
    response: str = ""

    # Error message if the query failed.
    error: Optional[str] = None


@contextmanager
def phase_timer(trace: Optional["QueryTrace"], name: str):
    """Context manager that records elapsed ms into trace.phases[name].

    If trace is None this is a pure no-op — no timing overhead at all.
    """
    if trace is None:
        yield
        return

    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        trace.phases[name] = round(elapsed_ms, 2)


def write_event(run_dir: Path, trace: "QueryTrace") -> None:
    """Append one JSON line to <run_dir>/events.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "query_id": trace.query_id,
        "timestamp": trace.timestamp,
        "run_id": trace.run_id,
        "run_iteration": trace.run_iteration,
        "configuration": trace.configuration,
        "query": trace.query,
        "route": trace.route,
        "phases": trace.phases,
        "crag": trace.crag,
        "sql": trace.sql,
        "retrieved_docs": trace.retrieved_docs,
        "response": trace.response,
        "error": trace.error,
    }
    events_path = run_dir / "events.jsonl"
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
