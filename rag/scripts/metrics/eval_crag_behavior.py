"""
Corrective RAG Behavior Profiling Script.

Runs RAG queries through the full CRAG pipeline, recording per-query:
relevance grade distribution, whether rewrite was triggered, and
relevance ratio before/after correction.

WHY THIS MATTERS FOR A CV:
  "Corrective RAG filtered out X% of irrelevant documents and triggered
  automatic query rewriting in Y% of cases" proves the CRAG loop is
  not decorative — it measurably improves context quality for synthesis.

Usage:
    python -m scripts.metrics.eval_crag_behavior --config config/proyectos_docentes.yaml
    python -m scripts.metrics.eval_crag_behavior --config config/proyectos_docentes.yaml --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_rag_queries(dataset_path: str) -> List[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return [q for q in dataset if q.get("expected_source_pattern")]


def _run_live(config_path: str, queries: List[dict]):
    """Run RAG queries through the full CRAG pipeline."""
    import logging
    import time

    logging.basicConfig(level=logging.WARNING)

    from rag_framework import RAGFramework

    rag = RAGFramework.from_yaml(config_path)
    rag.load_index()
    rag._query_ops._ensure_query_engine()
    query_engine = rag._query_engine

    # CRAG must be enabled
    if query_engine.corrective_rag_engine is None:
        print("ERROR: Corrective RAG is not enabled in the config.")
        print("Enable it with: corrective_rag.enabled: true")
        sys.exit(1)

    crag_engine = query_engine.corrective_rag_engine
    retriever = query_engine.retriever
    postprocessors = query_engine.node_postprocessors
    preprocessor = query_engine.query_preprocessor

    results = []
    for item in queries:
        qid = item["id"]
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

        # Reranking
        for pp in postprocessors:
            nodes = pp.postprocess_nodes(nodes, query_str=query_str)

        # CRAG processing
        crag_result = crag_engine.process(query_str, nodes)
        elapsed = time.time() - t0

        results.append({
            "id": qid,
            "query": query_str[:55],
            "total_retrieved": crag_result.total_retrieved,
            "relevant": crag_result.relevant_count,
            "irrelevant": crag_result.irrelevant_count,
            "ambiguous": crag_result.ambiguous_count,
            "rewritten": crag_result.query_rewritten,
            "rewritten_query": crag_result.rewritten_query,
            "relevance_ratio": crag_result.relevance_ratio,
            "filtered_count": len(crag_result.filtered_nodes),
            "elapsed_s": elapsed,
        })

    return results


def _run_dry(queries: List[dict]):
    """Generate plausible placeholder results."""
    import random
    random.seed(42)

    results = []
    for item in queries:
        total = random.randint(3, 5)
        relevant = random.randint(1, total)
        irrelevant = random.randint(0, total - relevant)
        ambiguous = total - relevant - irrelevant
        rewritten = relevant / total < 0.5 and random.random() < 0.6
        ratio = relevant / total

        results.append({
            "id": item["id"],
            "query": item["query"][:55],
            "total_retrieved": total,
            "relevant": relevant,
            "irrelevant": irrelevant,
            "ambiguous": ambiguous,
            "rewritten": rewritten,
            "rewritten_query": "reformulated query" if rewritten else None,
            "relevance_ratio": ratio,
            "filtered_count": relevant + ambiguous,
            "elapsed_s": random.uniform(1.5, 8.0),
        })

    return results


def _print_results(results: List[dict], dry_run: bool):
    if dry_run:
        print("\n⚠️  DRY-RUN MODE — values are simulated placeholders\n")

    n = len(results)
    if n == 0:
        print("No RAG queries found.")
        return

    # ── Per-query table ──
    print(
        f"{'ID':<10} {'Rel':>4} {'Irr':>4} {'Amb':>4} {'Ratio':>6} "
        f"{'Rewrite':>8} {'Time':>6}  Query"
    )
    print("-" * 90)
    for r in results:
        rw = "✓" if r["rewritten"] else "-"
        print(
            f"{r['id']:<10} {r['relevant']:>4} {r['irrelevant']:>4} "
            f"{r['ambiguous']:>4} {r['relevance_ratio']:>6.2f} "
            f"{rw:>8} {r['elapsed_s']:>5.1f}s  {r['query']}"
        )

    # ── Aggregate stats ──
    total_docs = sum(r["total_retrieved"] for r in results)
    total_relevant = sum(r["relevant"] for r in results)
    total_irrelevant = sum(r["irrelevant"] for r in results)
    total_ambiguous = sum(r["ambiguous"] for r in results)
    rewrites = sum(1 for r in results if r["rewritten"])
    mean_ratio = sum(r["relevance_ratio"] for r in results) / n
    docs_filtered = total_irrelevant  # docs removed by CRAG

    print("\n" + "=" * 60)
    print("CRAG BEHAVIOR SUMMARY")
    print("=" * 60)
    print(f"  Queries evaluated       : {n}")
    print(f"  Total docs graded       : {total_docs}")
    print(
        f"  Grade distribution      : {total_relevant} relevant, "
        f"{total_irrelevant} irrelevant, {total_ambiguous} ambiguous"
    )
    print(f"  Docs filtered (irrelev) : {docs_filtered}/{total_docs} "
          f"({docs_filtered/total_docs:.0%})")
    print(f"  Mean relevance ratio    : {mean_ratio:.2f}")
    print(f"  Queries with rewrite    : {rewrites}/{n} ({rewrites/n:.0%})")
    print(f"  Avg time/query          : {sum(r['elapsed_s'] for r in results)/n:.1f}s")

    # ── Markdown ──
    print("\n\n### Markdown (copy-paste for report)")
    print("```")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Docs graded | {total_docs} |")
    print(f"| Irrelevant docs filtered | {docs_filtered}/{total_docs} "
          f"({docs_filtered/total_docs:.0%}) |")
    print(f"| Mean relevance ratio | {mean_ratio:.2f} |")
    print(f"| Queries triggering rewrite | {rewrites}/{n} ({rewrites/n:.0%}) |")
    print("```")


def main():
    parser = argparse.ArgumentParser(
        description="Profile Corrective RAG behavior across eval queries."
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
        "--dry-run",
        action="store_true",
        help="Use simulated results (no LLM required)",
    )
    args = parser.parse_args()

    queries = _load_rag_queries(args.dataset)
    print(f"Loaded {len(queries)} RAG queries from {args.dataset}")

    if args.dry_run:
        results = _run_dry(queries)
    else:
        results = _run_live(args.config, queries)

    _print_results(results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
