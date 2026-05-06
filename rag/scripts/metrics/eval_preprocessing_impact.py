"""
Preprocessing Rank Shift Measurement Script.

Compares retrieval results with and without the metadata query preprocessor
to measure how much the preprocessor changes document ranking.  Reports
per-query rank shift delta and aggregate statistics.

WHY THIS MATTERS FOR A CV:
  "The metadata preprocessor shifted target documents by an average of
  X positions higher" proves that the fuzzy-matching + numeral expansion
  layer is not decorative — it materially improves retrieval precision
  for domain-specific queries.

Usage:
    python -m scripts.metrics.eval_preprocessing_impact --config config/proyectos_docentes.yaml
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _find_rank(files: List[str], pattern: str) -> Optional[int]:
    """Find the rank (1-indexed) of the first file matching the pattern."""
    regex = re.compile(pattern, re.IGNORECASE)
    for i, f in enumerate(files, start=1):
        if regex.search(f):
            return i
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Measure the impact of the query preprocessor on document ranking."
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
    args = parser.parse_args()

    from rag_framework import RAGFramework

    # Load dataset
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    rag_queries = [q for q in dataset if q.get("expected_source_pattern")]
    print(f"Loaded {len(rag_queries)} RAG queries from {args.dataset}")

    # Initialise framework
    print(f"Loading framework from {args.config} ...")
    rag = RAGFramework.from_yaml(args.config)
    rag.load_index()

    rag._query_ops._ensure_query_engine()
    query_engine = rag._query_engine

    retriever = query_engine.retriever
    postprocessors = query_engine.node_postprocessors
    preprocessor = query_engine.query_preprocessor

    if preprocessor is None:
        print("WARNING: Query preprocessor is not enabled in config.")
        print("Enable metadata.filtering.enabled in your YAML to measure impact.")
        print("Falling back to retrieval-only comparison (no preprocessing).")

    print(f"\n{'ID':<12} {'Rank w/o':>8} {'Rank w/':>8} {'Shift':>6} {'Match?':>7}  Query")
    print("-" * 90)

    shifts = []
    recoveries = 0
    losses = 0

    for item in rag_queries:
        qid = item["id"]
        query_str = item["query"]
        pattern = item["expected_source_pattern"]

        # ── Run 1: WITHOUT preprocessor ──
        nodes_raw = retriever.retrieve(query_str)
        for pp in postprocessors:
            nodes_raw = pp.postprocess_nodes(nodes_raw, query_str=query_str)
        files_raw = [n.node.metadata.get("file_name", "") for n in nodes_raw]
        rank_raw = _find_rank(files_raw, pattern)

        # ── Run 2: WITH preprocessor ──
        metadata_filters = None
        prefilter_result = None
        if preprocessor is not None:
            prefilter_result = preprocessor.analyse(query_str)
            metadata_filters = prefilter_result.metadata_filters

        if metadata_filters is not None:
            nodes_pp = retriever.retrieve(query_str, filters=metadata_filters)
        else:
            nodes_pp = retriever.retrieve(query_str)

        for pp in postprocessors:
            nodes_pp = pp.postprocess_nodes(nodes_pp, query_str=query_str)

        # Apply boost
        if (
            prefilter_result is not None
            and prefilter_result.boost_field_value
            and preprocessor is not None
        ):
            filter_cfg = preprocessor.config
            nodes_pp = preprocessor.apply_boost(
                nodes_pp,
                boost_field=filter_cfg.boost_field,
                boost_value=prefilter_result.boost_field_value,
                factor=filter_cfg.boost_factor,
            )

        files_pp = [n.node.metadata.get("file_name", "") for n in nodes_pp]
        rank_pp = _find_rank(files_pp, pattern)

        # Compute shift
        rank_raw_str = str(rank_raw) if rank_raw else "miss"
        rank_pp_str = str(rank_pp) if rank_pp else "miss"

        if rank_raw is not None and rank_pp is not None:
            shift = rank_raw - rank_pp  # positive = improved
            shifts.append(shift)
        elif rank_raw is None and rank_pp is not None:
            shift = "recovered"
            recoveries += 1
        elif rank_raw is not None and rank_pp is None:
            shift = "lost"
            losses += 1
        else:
            shift = "both miss"

        matched = ""
        if prefilter_result and prefilter_result.matched_value:
            matched = prefilter_result.matched_value[:20]

        short_query = query_str[:40] + ("..." if len(query_str) > 40 else "")
        print(
            f"{qid:<12} {rank_raw_str:>8} {rank_pp_str:>8} "
            f"{str(shift):>6} {matched:>7}  {short_query}"
        )

    # ── Aggregate ──
    n = len(rag_queries)
    print("\n" + "=" * 60)
    print("PREPROCESSING IMPACT SUMMARY")
    print("=" * 60)

    if shifts:
        avg_shift = sum(shifts) / len(shifts)
        positive_shifts = sum(1 for s in shifts if s > 0)
        negative_shifts = sum(1 for s in shifts if s < 0)
        zero_shifts = sum(1 for s in shifts if s == 0)
        max_improvement = max(shifts) if shifts else 0

        print(f"  Queries with both ranks  : {len(shifts)}")
        print(f"  Avg rank shift (↑ better): {avg_shift:+.2f} positions")
        print(f"  Max improvement          : {max_improvement:+d} positions")
        print(f"  Improved / Same / Worse  : {positive_shifts} / {zero_shifts} / {negative_shifts}")

    print(f"  Misses recovered by PP   : {recoveries}")
    print(f"  Hits lost by PP          : {losses}")

    # ── Markdown ──
    print("\n\n### Markdown (copy-paste for report)")
    print("```")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    if shifts:
        avg_shift = sum(shifts) / len(shifts)
        print(f"| Avg rank shift | {avg_shift:+.2f} positions |")
        print(f"| Queries improved | {sum(1 for s in shifts if s > 0)}/{len(shifts)} |")
    print(f"| Misses recovered | {recoveries} |")
    print(f"| Hits lost | {losses} |")
    print("```")


if __name__ == "__main__":
    main()
