"""
Retrieval evaluation script for the RAG framework.

Measures retrieval quality (Hit Rate, MRR, Precision) using an evaluation
dataset, WITHOUT invoking the LLM for synthesis. This allows fast iteration
on retrieval parameters.

Usage:
    python -m tests.evaluation.eval_retrieval --config config/proyectos_docentes.yaml
    python -m tests.evaluation.eval_retrieval --config config/proyectos_docentes.yaml --top-k 10
    python -m tests.evaluation.eval_retrieval --config config/proyectos_docentes.yaml --full-pipeline
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Add project root to path so imports work when run as a script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rag_framework import RAGFramework


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    query_id: str
    query: str
    query_type: str
    expected_source_pattern: Optional[str]
    retrieved_files: List[str]
    retrieved_scores: List[float]
    hit: bool = False
    reciprocal_rank: float = 0.0
    precision_at_k: float = 0.0
    elapsed_s: float = 0.0


@dataclass
class EvalReport:
    results: List[QueryResult] = field(default_factory=list)
    hit_rate: float = 0.0
    mrr: float = 0.0
    mean_precision: float = 0.0
    elapsed_total_s: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_hit(retrieved_files: List[str], pattern: str) -> bool:
    """Check if any retrieved file matches the expected source regex."""
    regex = re.compile(pattern, re.IGNORECASE)
    return any(regex.search(f) for f in retrieved_files)


def _reciprocal_rank(retrieved_files: List[str], pattern: str) -> float:
    regex = re.compile(pattern, re.IGNORECASE)
    for i, f in enumerate(retrieved_files, start=1):
        if regex.search(f):
            return 1.0 / i
    return 0.0


def _precision_at_k(retrieved_files: List[str], pattern: str) -> float:
    if not retrieved_files:
        return 0.0
    regex = re.compile(pattern, re.IGNORECASE)
    hits = sum(1 for f in retrieved_files if regex.search(f))
    return hits / len(retrieved_files)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    config_path: str,
    dataset_path: str,
    top_k_override: Optional[int] = None,
    full_pipeline: bool = False,
) -> EvalReport:
    """Run retrieval evaluation.

    Args:
        config_path: Path to YAML config.
        dataset_path: Path to JSON evaluation dataset.
        top_k_override: Override the top_k parameter for retrieval.
        full_pipeline: If True, run through query preprocessor + reranker
                       (but still skip LLM synthesis). If False, raw retrieval only.
    """
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Filter to only RAG-type queries (those with expected_source_pattern)
    rag_queries = [q for q in dataset if q.get("expected_source_pattern")]
    if not rag_queries:
        print("No RAG queries with expected_source_pattern found in dataset.")
        return EvalReport()

    print(f"Loaded {len(rag_queries)} RAG queries from {dataset_path}")

    # Initialise framework (load existing index, no ingestion)
    print(f"Loading framework from {config_path} ...")
    rag = RAGFramework.from_yaml(config_path)
    rag.load_index()

    # Get retriever and postprocessors via the query engine
    rag._query_ops._ensure_query_engine()
    query_engine = rag._query_engine

    retriever = query_engine.retriever
    postprocessors = query_engine.node_postprocessors if full_pipeline else []
    preprocessor = query_engine.query_preprocessor if full_pipeline else None

    if top_k_override is not None:
        # Override retriever top_k if provided
        if hasattr(retriever, "_top_k"):
            retriever._top_k = top_k_override
        print(f"Overriding top_k to {top_k_override}")

    report = EvalReport()
    t_start = time.time()

    print(
        f"\nRunning evaluation ({'full pipeline' if full_pipeline else 'retrieval only'})...\n"
    )
    print(f"{'ID':<12} {'Hit':>3} {'RR':>6} {'P@k':>6} {'#Docs':>5} {'Time':>6}  Query")
    print("-" * 90)

    for item in rag_queries:
        qid = item["id"]
        query_str = item["query"]
        pattern = item["expected_source_pattern"]

        t0 = time.time()

        # --- Query preprocessing (metadata pre-filter) ---
        metadata_filters = None
        prefilter_result = None
        if preprocessor is not None:
            prefilter_result = preprocessor.analyse(query_str)
            metadata_filters = prefilter_result.metadata_filters

        # --- Retrieval ---
        if metadata_filters is not None:
            nodes = retriever.retrieve(query_str, filters=metadata_filters)
        else:
            nodes = retriever.retrieve(query_str)

        # --- Postprocessing (reranking) ---
        for pp in postprocessors:
            nodes = pp.postprocess_nodes(nodes, query_str=query_str)

        # --- Metadata boost ---
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

        # Collect retrieved file names
        files = []
        scores = []
        for n in nodes:
            fname = n.node.metadata.get("file_name", "")
            files.append(fname)
            scores.append(n.score if n.score is not None else 0.0)

        # Compute metrics
        hit = _is_hit(files, pattern)
        rr = _reciprocal_rank(files, pattern)
        prec = _precision_at_k(files, pattern)

        result = QueryResult(
            query_id=qid,
            query=query_str,
            query_type=item.get("type", "rag"),
            expected_source_pattern=pattern,
            retrieved_files=files,
            retrieved_scores=scores,
            hit=hit,
            reciprocal_rank=rr,
            precision_at_k=prec,
            elapsed_s=elapsed,
        )
        report.results.append(result)

        hit_sym = "Y" if hit else "N"
        short_query = query_str[:50] + ("..." if len(query_str) > 50 else "")
        print(
            f"{qid:<12} {hit_sym:>3} {rr:>6.3f} {prec:>6.3f} {len(files):>5} {elapsed:>5.2f}s  {short_query}"
        )

    report.elapsed_total_s = time.time() - t_start

    # Aggregate metrics
    n = len(report.results)
    report.hit_rate = sum(r.hit for r in report.results) / n if n else 0
    report.mrr = sum(r.reciprocal_rank for r in report.results) / n if n else 0
    report.mean_precision = (
        sum(r.precision_at_k for r in report.results) / n if n else 0
    )

    # Print summary
    print("\n" + "=" * 90)
    print("AGGREGATE METRICS")
    print("=" * 90)
    print(f"  Queries evaluated : {n}")
    print(
        f"  Hit Rate@k        : {report.hit_rate:.3f}  ({sum(r.hit for r in report.results)}/{n})"
    )
    print(f"  MRR               : {report.mrr:.3f}")
    print(f"  Mean Precision@k  : {report.mean_precision:.3f}")
    print(f"  Total time        : {report.elapsed_total_s:.1f}s")
    print(f"  Avg time/query    : {report.elapsed_total_s / max(n, 1):.2f}s")

    # Show misses for debugging
    misses = [r for r in report.results if not r.hit]
    if misses:
        print(f"\n  MISSES ({len(misses)}):")
        for r in misses:
            top_files = ", ".join(r.retrieved_files[:3]) or "(none)"
            print(f"    {r.query_id}: {r.query[:60]}")
            print(f"      Expected: {r.expected_source_pattern}")
            print(f"      Got top 3: {top_files}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality without LLM synthesis."
    )
    parser.add_argument(
        "--config",
        default="config/proyectos_docentes.yaml",
        help="Path to YAML config (default: config/proyectos_docentes.yaml)",
    )
    parser.add_argument(
        "--dataset",
        default="tests/evaluation/eval_dataset.json",
        help="Path to eval dataset JSON (default: tests/evaluation/eval_dataset.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override retrieval top_k",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline (preprocessor + reranker), not just raw retrieval",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    evaluate_retrieval(
        config_path=args.config,
        dataset_path=args.dataset,
        top_k_override=args.top_k,
        full_pipeline=args.full_pipeline,
    )


if __name__ == "__main__":
    main()
