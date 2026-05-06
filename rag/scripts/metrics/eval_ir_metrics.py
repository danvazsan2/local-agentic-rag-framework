"""
IR Metrics Evaluation Script.

Wraps the existing eval harness to produce a clean markdown report with
Hit Rate, MRR, Precision@k, and NDCG@k at multiple k values (3, 5, 10).
NDCG@k is new — the existing harness only computes Hit Rate, MRR, and P@k.

WHY THIS MATTERS FOR A CV:
  Hit Rate@5 and MRR are the canonical IR metrics that every retrieval
  paper reports.  Reporting them at multiple k values shows rigorous
  evaluation methodology.  NDCG@k additionally captures graded relevance.

Usage:
    python -m scripts.metrics.eval_ir_metrics --config config/proyectos_docentes.yaml
    python -m scripts.metrics.eval_ir_metrics --config config/proyectos_docentes.yaml --full-pipeline
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _is_hit_at_k(retrieved_files: List[str], pattern: str, k: int) -> bool:
    regex = re.compile(pattern, re.IGNORECASE)
    return any(regex.search(f) for f in retrieved_files[:k])


def _reciprocal_rank_at_k(retrieved_files: List[str], pattern: str, k: int) -> float:
    regex = re.compile(pattern, re.IGNORECASE)
    for i, f in enumerate(retrieved_files[:k], start=1):
        if regex.search(f):
            return 1.0 / i
    return 0.0


def _precision_at_k(retrieved_files: List[str], pattern: str, k: int) -> float:
    subset = retrieved_files[:k]
    if not subset:
        return 0.0
    regex = re.compile(pattern, re.IGNORECASE)
    hits = sum(1 for f in subset if regex.search(f))
    return hits / len(subset)


def _ndcg_at_k(retrieved_files: List[str], pattern: str, k: int) -> float:
    """Compute NDCG@k with binary relevance (1 if matches pattern, else 0)."""
    regex = re.compile(pattern, re.IGNORECASE)
    subset = retrieved_files[:k]

    # DCG: sum of rel_i / log2(i+1)
    dcg = 0.0
    for i, f in enumerate(subset, start=1):
        rel = 1.0 if regex.search(f) else 0.0
        dcg += rel / math.log2(i + 1)

    if dcg == 0:
        return 0.0

    # IDCG: best possible ordering — all relevant docs first
    n_relevant = sum(1 for f in subset if regex.search(f))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))

    return dcg / idcg if idcg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IR metrics at multiple k values."
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
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline (preprocessor + reranker)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        help="k values to evaluate (default: 3 5 10)",
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
    postprocessors = query_engine.node_postprocessors if args.full_pipeline else []
    preprocessor = query_engine.query_preprocessor if args.full_pipeline else None

    max_k = max(args.k_values)

    # Override top_k to max_k to have enough candidates
    if hasattr(retriever, "config"):
        original_top_k = retriever.config.top_k
        retriever.config.top_k = max(max_k, original_top_k)

    mode = "full pipeline" if args.full_pipeline else "retrieval-only"
    print(f"\nRunning evaluation ({mode}) for k ∈ {args.k_values}...\n")

    # Collect per-query results
    all_files = []  # list of (files, pattern, elapsed)
    t_total = time.time()

    for item in rag_queries:
        pattern = item["expected_source_pattern"]
        query_str = item["query"]

        t0 = time.time()

        # Preprocessing
        metadata_filters = None
        prefilter_result = None
        if preprocessor is not None:
            prefilter_result = preprocessor.analyse(query_str)
            metadata_filters = prefilter_result.metadata_filters

        # Retrieval
        if metadata_filters is not None:
            nodes = retriever.retrieve(query_str, filters=metadata_filters)
        else:
            nodes = retriever.retrieve(query_str)

        # Postprocessing
        for pp in postprocessors:
            nodes = pp.postprocess_nodes(nodes, query_str=query_str)

        # Boost
        if (
            prefilter_result is not None
            and prefilter_result.boost_field_value
            and preprocessor is not None
        ):
            filter_cfg = preprocessor.config
            nodes = preprocessor.apply_boost(
                nodes,
                boost_field=filter_cfg.boost_field,
                boost_value=prefilter_result.boost_field_value,
                factor=filter_cfg.boost_factor,
            )

        elapsed = time.time() - t0
        files = [n.node.metadata.get("file_name", "") for n in nodes]
        all_files.append((files, pattern, elapsed))

    total_time = time.time() - t_total
    n = len(all_files)

    # ── Compute metrics at each k ──
    print(f"{'k':>3} {'HR@k':>8} {'MRR@k':>8} {'P@k':>8} {'NDCG@k':>8}")
    print("-" * 40)

    rows = []
    for k in args.k_values:
        hr = sum(1 for files, pat, _ in all_files if _is_hit_at_k(files, pat, k)) / n
        mrr = sum(_reciprocal_rank_at_k(files, pat, k) for files, pat, _ in all_files) / n
        prec = sum(_precision_at_k(files, pat, k) for files, pat, _ in all_files) / n
        ndcg = sum(_ndcg_at_k(files, pat, k) for files, pat, _ in all_files) / n
        rows.append((k, hr, mrr, prec, ndcg))
        print(f"{k:>3} {hr:>8.3f} {mrr:>8.3f} {prec:>8.3f} {ndcg:>8.3f}")

    avg_time = total_time / max(n, 1)
    print(f"\nTotal time: {total_time:.1f}s | Avg time/query: {avg_time:.2f}s")

    # ── Markdown output ──
    print("\n\n### Markdown (copy-paste for report)")
    print("```")
    print(f"| k | HR@k | MRR@k | P@k | NDCG@k |")
    print(f"|---|------|-------|-----|--------|")
    for k, hr, mrr, prec, ndcg in rows:
        print(f"| {k} | {hr:.3f} | {mrr:.3f} | {prec:.3f} | {ndcg:.3f} |")
    print("```")


if __name__ == "__main__":
    main()
