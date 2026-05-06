"""
Inspection tool for evaluation run events.

Usage (run from the rag/ directory):

    # Show all queries in a run
    python validation/inspect_run.py eval_v1

    # Filter by query type
    python validation/inspect_run.py eval_v1 --type sql

    # Filter by partition
    python validation/inspect_run.py eval_v1 --partition adversarial

    # Inspect a specific query
    python validation/inspect_run.py eval_v1 --query-id s01

    # Latency percentiles
    python validation/inspect_run.py eval_v1 --percentile 95

    # Show full SQL attempt traces for sql queries
    python validation/inspect_run.py eval_v1 --type sql --show-sql
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
_RUNS_DIR = Path(__file__).parent / "runs"

# Also load dataset to enrich events with partition/type info
_DATASET_PATH = Path(__file__).parent / "dataset.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_events(run_id: str) -> List[dict]:
    events_path = _RUNS_DIR / run_id / "events.jsonl"
    if not events_path.exists():
        print(f"ERROR: {events_path} not found.", file=sys.stderr)
        sys.exit(1)
    events = []
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_dataset_index() -> Dict[str, dict]:
    """Return {query_id: dataset_entry} for fast lookup."""
    if not _DATASET_PATH.exists():
        return {}
    with open(_DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"]: q for q in data}


def enrich(events: List[dict], dataset: Dict[str, dict]) -> List[dict]:
    """Add partition/type/difficulty from dataset into each event."""
    for ev in events:
        ds = dataset.get(ev.get("query_id", ""), {})
        ev.setdefault("_type", ds.get("type", "?"))
        ev.setdefault("_partition", ds.get("partition", "?"))
        ev.setdefault("_difficulty", ds.get("difficulty", "?"))
    return events


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_events(
    events: List[dict],
    query_id: Optional[str] = None,
    qtype: Optional[str] = None,
    partition: Optional[str] = None,
) -> List[dict]:
    if query_id:
        events = [e for e in events if e.get("query_id") == query_id]
    if qtype:
        events = [e for e in events if e.get("_type") == qtype]
    if partition:
        events = [e for e in events if e.get("_partition") == partition]
    return events


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(len(s) * pct / 100), len(s) - 1)
    return s[idx]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_HR = "=" * 70
_HR2 = "-" * 70

_PHASE_ORDER = [
    "preprocessor_ms",
    "vector_ms",
    "bm25_ms",
    "rrf_ms",
    "retrieval_ms",
    "reranker_ms",
    "router_keyword_ms",
    "router_llm_ms",
    "crag_ms",
    "synthesis_ms",
    "total_ms",
]


def _phase_table(phases: dict) -> str:
    if not phases:
        return "  (no phase data)"
    lines = []
    known = {k: phases[k] for k in _PHASE_ORDER if k in phases}
    extra = {k: v for k, v in phases.items() if k not in known}
    for k, v in {**known, **extra}.items():
        lines.append(f"  {k:<22} {v:>9.1f} ms")
    return "\n".join(lines)


def _route_line(route: Optional[dict]) -> str:
    if not route:
        return "(none)"
    return (
        f"{route.get('source','?')}  "
        f"conf={route.get('confidence', 0):.2f}  "
        f"method={route.get('method','?')}"
    )


def _crag_summary(crag: Optional[dict]) -> str:
    if not crag:
        return "(not run)"
    ratio = crag.get("relevance_ratio", 0)
    rewrite = crag.get("rewrite_triggered", False)
    chunks = crag.get("grading_per_chunk_ms", [])
    total_grade_ms = sum(chunks)
    return (
        f"relevance_ratio={ratio:.2f}  rewrite={'YES' if rewrite else 'no'}  "
        f"chunks={len(chunks)}  grading_total={total_grade_ms:.0f}ms"
    )


def _sql_summary(sql: Optional[dict]) -> str:
    if not sql:
        return "(not run)"
    attempts = sql.get("total_attempts", 0)
    total_ms = sql.get("total_ms", 0)
    success = sql.get("success", False)
    relaxed = sql.get("query_relaxed", False)
    return (
        f"attempts={attempts}  total={total_ms:.0f}ms  "
        f"success={'YES' if success else 'FAIL'}  relaxed={'YES' if relaxed else 'no'}"
    )


def print_event(ev: dict, verbose: bool = False, show_sql: bool = False) -> None:
    qid = ev.get("query_id", "?")
    qtype = ev.get("_type", "?")
    partition = ev.get("_partition", "?")
    diff = ev.get("_difficulty", "?")
    query = ev.get("query", "")[:80]
    error = ev.get("error")

    print(_HR)
    print(
        f"  {qid}  [{qtype}·{partition}·{diff}]"
        + (f"  ERROR: {error}" if error else "")
    )
    print(f"  Query: {query}")
    print()
    print(f"  Route  : {_route_line(ev.get('route'))}")
    print(f"  CRAG   : {_crag_summary(ev.get('crag'))}")
    print(f"  SQL    : {_sql_summary(ev.get('sql'))}")

    if ev.get("phases"):
        print(f"\n  Phases:")
        print(_phase_table(ev["phases"]))

    if ev.get("retrieved_docs"):
        docs = ev["retrieved_docs"]
        print(f"\n  Retrieved docs ({len(docs)}):")
        for d in docs[:5]:
            print(f"    {d}")
        if len(docs) > 5:
            print(f"    … ({len(docs) - 5} more)")

    if verbose and ev.get("response"):
        print(f"\n  Response:\n    {ev['response'][:300]}")

    if show_sql and ev.get("sql") and ev["sql"].get("attempts"):
        print(f"\n  SQL attempts:")
        for att in ev["sql"]["attempts"]:
            n = att.get("attempt", "?")
            gen = att.get("sql_generation_ms") or 0
            val = att.get("validation_ms") or 0
            exe = att.get("execution_ms") or 0
            err = att.get("error_type") or "ok"
            print(f"    #{n}  gen={gen:.0f}ms  val={val:.0f}ms  exec={exe:.0f}ms  → {err}")


# ---------------------------------------------------------------------------
# Percentile report
# ---------------------------------------------------------------------------


def print_percentile_report(events: List[dict], pct: float) -> None:
    print(_HR)
    print(f"LATENCY PERCENTILES (p{pct:.0f})")
    print(_HR)

    # Overall total_ms
    totals = [e["phases"]["total_ms"] for e in events if e.get("phases", {}).get("total_ms")]
    if totals:
        print(f"\n  Total latency (n={len(totals)}):")
        for p in [50, 75, 90, 95, 99]:
            print(f"    p{p:<3} = {percentile(totals, p):>8.1f} ms")

    # Per phase
    all_phase_names = set()
    for ev in events:
        all_phase_names.update(ev.get("phases", {}).keys())

    known = [k for k in _PHASE_ORDER if k in all_phase_names]
    extra = sorted(all_phase_names - set(known))

    print(f"\n  Per-phase p{pct:.0f}:")
    header = f"  {'Phase':<22}  {'p{:.0f}'.format(pct):>10}  {'mean':>10}  {'n':>6}"
    print(header)
    print(f"  {_HR2}")
    for phase in known + extra:
        vals = [e["phases"][phase] for e in events if phase in e.get("phases", {})]
        if vals:
            mean = sum(vals) / len(vals)
            print(
                f"  {phase:<22}  {percentile(vals, pct):>10.1f}  {mean:>10.1f}  {len(vals):>6}"
            )

    # By type
    types = sorted({e.get("_type", "?") for e in events})
    if len(types) > 1:
        print(f"\n  Total ms by query type (p{pct:.0f} | mean | n):")
        for t in types:
            vals = [
                e["phases"]["total_ms"]
                for e in events
                if e.get("_type") == t and e.get("phases", {}).get("total_ms")
            ]
            if vals:
                mean = sum(vals) / len(vals)
                print(
                    f"    {t:<14}  p{pct:.0f}={percentile(vals, pct):>8.1f}ms"
                    f"  mean={mean:>8.1f}ms  n={len(vals)}"
                )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(events: List[dict]) -> None:
    print(_HR)
    print(f"RUN SUMMARY — {len(events)} events")
    print(_HR)
    print(
        f"  {'ID':<8}  {'Type':<8}  {'Part':<12}  {'Route':<14}  {'Total ms':>10}  {'Status':<8}"
    )
    print(f"  {_HR2}")
    for ev in events:
        qid = ev.get("query_id", "?")
        qtype = ev.get("_type", "?")
        part = ev.get("_partition", "?")
        route_src = ev.get("route", {}).get("source", "?") if ev.get("route") else "?"
        total_ms = ev.get("phases", {}).get("total_ms", 0)
        status = "ERROR" if ev.get("error") else "ok"
        print(
            f"  {qid:<8}  {qtype:<8}  {part:<12}  {route_src:<14}  {total_ms:>10.1f}  {status:<8}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect events from a validation run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_id", help="Run identifier (folder under validation/runs/)")
    parser.add_argument("--query-id", metavar="ID", help="Show a single query by ID")
    parser.add_argument(
        "--type",
        choices=["rag", "sql", "hybrid", "negative", "out_of_domain"],
        help="Filter by query type",
    )
    parser.add_argument(
        "--partition",
        choices=["well_formed", "adversarial"],
        help="Filter by partition",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        metavar="PCT",
        help="Print latency percentile report (e.g. 95)",
    )
    parser.add_argument(
        "--show-sql",
        action="store_true",
        help="Show per-attempt SQL traces",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show response preview",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print summary table (default: on)",
    )
    args = parser.parse_args()

    events = load_events(args.run_id)
    dataset = load_dataset_index()
    events = enrich(events, dataset)

    events = filter_events(
        events,
        query_id=args.query_id,
        qtype=args.type,
        partition=args.partition,
    )

    if not events:
        print("No events matched the filters.")
        return

    if args.percentile is not None:
        print_percentile_report(events, args.percentile)
    elif args.query_id:
        print_event(events[0], verbose=True, show_sql=args.show_sql)
    else:
        if args.summary:
            print_summary_table(events)
        if args.show_sql or args.verbose:
            for ev in events:
                print_event(ev, verbose=args.verbose, show_sql=args.show_sql)


if __name__ == "__main__":
    main()
