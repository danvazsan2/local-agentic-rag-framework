"""
Case study generator: detailed trace analysis for selected queries.

Generates markdown files with step-by-step analysis of interesting
cases (s01 mandatory, plus 2-4 additional).
"""

import json
from pathlib import Path
from typing import Dict, List


def generate_case_study(
    query_id: str,
    events: List[dict],
    dataset_entry: dict,
    output_dir: Path,
) -> Path:
    """Generate a case study markdown file for a single query.

    Args:
        query_id: the query ID (e.g., "s01")
        events: all events for this query (may be multiple runs)
        dataset_entry: the dataset entry for this query
        output_dir: directory for case_studies/ output

    Returns:
        Path to the generated .md file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{query_id}.md"

    lines = [
        f"# Case Study: {query_id}",
        "",
        f"**Query**: {dataset_entry.get('query', '?')}",
        f"**Type**: {dataset_entry.get('type', '?')}",
        f"**Partition**: {dataset_entry.get('partition', '?')}",
        f"**Difficulty**: {dataset_entry.get('difficulty', '?')}",
        f"**Expected abstention**: {dataset_entry.get('expected_abstention', False)}",
        "",
        f"**Rationale**: {dataset_entry.get('rationale', '')}",
        "",
        "---",
        "",
    ]

    for i, ev in enumerate(events):
        lines.append(f"## Run {i + 1} (iteration {ev.get('run_iteration', '?')})")
        lines.append("")

        # Routing
        route = ev.get("route", {})
        if route:
            lines.append(f"**Route**: {route.get('source', '?')} "
                         f"(confidence={route.get('confidence', 0):.2f}, "
                         f"method={route.get('method', '?')})")
        else:
            lines.append("**Route**: (not recorded)")
        lines.append("")

        # Phases
        phases = ev.get("phases", {})
        if phases:
            lines.append("### Phase Timings")
            lines.append("")
            lines.append("| Phase | Time (ms) |")
            lines.append("|-------|-----------|")
            for phase, ms in sorted(phases.items()):
                lines.append(f"| {phase} | {ms:.1f} |")
            lines.append("")

        # SQL attempts
        sql = ev.get("sql", {})
        if sql and sql.get("attempts"):
            lines.append("### SQL Attempts")
            lines.append("")
            for att in sql["attempts"]:
                n = att.get("attempt", "?")
                lines.append(f"#### Attempt {n}")
                lines.append(f"- Generation time: {att.get('sql_generation_ms', 0):.0f} ms")
                lines.append(f"- Validation time: {att.get('validation_ms', 0):.0f} ms")
                lines.append(f"- Execution time: {att.get('execution_ms', 0):.0f} ms")
                if att.get("generated_sql"):
                    lines.append(f"- SQL: `{att['generated_sql']}`")
                if att.get("error_type") and att["error_type"] != "ok":
                    lines.append(f"- Error: {att.get('error_type', '')} — {att.get('error_message', '')}")
                lines.append("")
            lines.append(f"**Total SQL time**: {sql.get('total_ms', 0):.0f} ms")
            lines.append(f"**Success**: {sql.get('success', False)}")
            lines.append(f"**Relaxed**: {sql.get('query_relaxed', False)}")
            lines.append("")

        # CRAG
        crag = ev.get("crag", {})
        if crag:
            lines.append("### CRAG")
            lines.append(f"- Relevance ratio: {crag.get('relevance_ratio', 0):.2f}")
            lines.append(f"- Rewrite triggered: {crag.get('rewrite_triggered', False)}")
            if crag.get("rewritten_query"):
                lines.append(f"- Rewritten query: {crag['rewritten_query']}")
            lines.append("")

        # Retrieved docs
        docs = ev.get("retrieved_docs", [])
        if docs:
            lines.append("### Retrieved Documents")
            for j, d in enumerate(docs[:5]):
                lines.append(f"{j+1}. {d}")
            if len(docs) > 5:
                lines.append(f"... ({len(docs) - 5} more)")
            lines.append("")

        # Response preview
        response = ev.get("response", "")
        if response:
            lines.append("### Response")
            lines.append(f"```\n{response[:500]}\n```")
            lines.append("")

        # Error
        if ev.get("error"):
            lines.append(f"### ⚠ Error\n{ev['error']}")
            lines.append("")

        lines.append("---")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path
