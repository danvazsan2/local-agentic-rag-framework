"""
Configurable retrieval runner for ablation study.

Runs retrieval through any of the C1-C7 configurations by
dynamically toggling pipeline components (alpha, preprocessor, reranker).

Key design: for BM25-only or vector-only configs, we call the
sub-retrievers directly to avoid unnecessary embedding/BM25 calls.

Usage:
    from validation.evaluation.runner import run_config_retrieval

    results = run_config_retrieval(rag, queries, config, k_values)
"""

import logging
import time
from typing import List, Optional, Tuple

from validation.evaluation.configurations import PipelineConfig, apply_config, restore_alpha
from validation.evaluation.metrics.retrieval import compute_per_query, aggregate_by_partition
from validation.evaluation.metrics.routing import get_ground_truth_source

logger = logging.getLogger(__name__)

# Maximum retries for transient Ollama errors (NaN embeddings, timeouts)
_MAX_RETRIES = 2


def _retrieve_with_config(
    retriever, postprocessors, preprocessor, query_str: str,
    pipeline_cfg: PipelineConfig,
) -> List[str]:
    """Run one query through configured retrieval and return filenames.

    Selectively uses only the sub-retrievers needed by the config:
    - BM25-only (C1): calls bm25_retriever directly, skips embeddings
    - Vector-only (C2, C7): calls vector_retriever directly, skips BM25
    - Hybrid (C3-C6): calls full HybridRetriever with RRF fusion
    """
    metadata_filters = None
    prefilter_result = None

    if preprocessor is not None:
        prefilter_result = preprocessor.analyse(query_str)
        metadata_filters = prefilter_result.metadata_filters

    # ── Select retrieval path based on config ──
    nodes = _run_retrieval(
        retriever, query_str, pipeline_cfg, metadata_filters
    )

    # ── Postprocess (reranker) ──
    for pp in postprocessors:
        nodes = pp.postprocess_nodes(nodes, query_str=query_str)

    # ── Metadata boost ──
    if (
        prefilter_result is not None
        and prefilter_result.boost_field_value
        and preprocessor is not None
    ):
        cfg = preprocessor.config
        nodes = preprocessor.apply_boost(
            nodes,
            boost_field=cfg.boost_field,
            boost_value=prefilter_result.boost_field_value,
            factor=cfg.boost_factor,
        )

    return [n.node.metadata.get("file_name", "") for n in nodes]


def _run_retrieval(retriever, query_str, pipeline_cfg, metadata_filters):
    """Dispatch retrieval to the correct sub-retriever(s).

    For BM25-only: skip vector retrieval entirely (no embedding call).
    For vector-only: skip BM25 retrieval entirely.
    For hybrid: use the full HybridRetriever with RRF.
    """
    from rag_framework.core.retrieval import HybridRetriever

    is_hybrid = isinstance(retriever, HybridRetriever)

    if not is_hybrid:
        # Fallback: non-hybrid retriever, just call it
        return retriever.retrieve(query_str)

    # ── BM25-only (C1): skip vector retrieval ──
    if pipeline_cfg.use_bm25 and not pipeline_cfg.use_vector:
        nodes = retriever.bm25_retriever.retrieve(query_str)
        if metadata_filters is not None:
            nodes = retriever._manual_metadata_filter(nodes, metadata_filters)
        return nodes[:retriever.config.top_k]

    # ── Vector-only (C2, C7): skip BM25 retrieval ──
    if pipeline_cfg.use_vector and not pipeline_cfg.use_bm25:
        if metadata_filters is not None:
            from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
            filtered_retriever = VectorIndexRetriever(
                index=retriever.index,
                similarity_top_k=retriever.config.top_k,
                filters=metadata_filters,
            )
            return filtered_retriever.retrieve(query_str)
        else:
            return retriever.vector_retriever.retrieve(query_str)

    # ── Hybrid (C3-C6): full RRF fusion ──
    if metadata_filters is not None:
        return retriever.retrieve(query_str, filters=metadata_filters)
    else:
        return retriever.retrieve(query_str)


def run_config_retrieval(
    rag,
    queries: List[dict],
    pipeline_cfg: PipelineConfig,
    k_values: List[int],
) -> dict:
    """Run retrieval evaluation for one configuration.

    Args:
        rag: loaded RAGFramework instance
        queries: dataset queries with expected_source_pattern
        pipeline_cfg: ablation configuration
        k_values: list of k values (e.g., [1, 3, 5, 10])

    Returns:
        Dict with per_query results and aggregated metrics by partition.
    """
    # Only evaluate queries that have expected_source_pattern
    rag_queries = [q for q in queries if q.get("expected_source_pattern")]

    # Apply configuration overrides
    retriever, postprocessors, preprocessor, original_alpha = apply_config(
        pipeline_cfg, rag
    )

    # Ensure retriever returns enough candidates
    max_k = max(k_values)
    if hasattr(retriever, "config"):
        saved_top_k = retriever.config.top_k
        retriever.config.top_k = max(max_k, retriever.config.top_k)
    else:
        saved_top_k = None

    per_query = []
    errors = []
    t_start = time.time()

    for i, item in enumerate(rag_queries):
        qid = item["id"]
        try:
            files = _retrieve_with_config(
                retriever, postprocessors, preprocessor,
                item["query"], pipeline_cfg,
            )
        except Exception as e:
            # Retry once for transient Ollama errors
            retried = False
            for attempt in range(1, _MAX_RETRIES + 1):
                logger.warning(
                    "[%s] q=%s attempt %d failed: %s — retrying",
                    pipeline_cfg.id, qid, attempt, e,
                )
                time.sleep(1)
                try:
                    files = _retrieve_with_config(
                        retriever, postprocessors, preprocessor,
                        item["query"], pipeline_cfg,
                    )
                    retried = True
                    break
                except Exception:
                    continue

            if not retried:
                logger.error(
                    "[%s] q=%s SKIPPED after %d retries: %s",
                    pipeline_cfg.id, qid, _MAX_RETRIES, e,
                )
                errors.append({"id": qid, "error": str(e)})
                files = []

        result = compute_per_query(
            query_id=qid,
            query=item["query"],
            pattern=item["expected_source_pattern"],
            files=files,
            k_values=k_values,
            partition=item.get("partition", ""),
            difficulty=item.get("difficulty", ""),
            query_type=item.get("type", ""),
        )
        per_query.append(result)

        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == len(rag_queries):
            print(f"    [{pipeline_cfg.id}] {i+1}/{len(rag_queries)} queries processed")

    total_time = time.time() - t_start

    # Restore original settings
    restore_alpha(retriever, original_alpha)
    if saved_top_k is not None and hasattr(retriever, "config"):
        retriever.config.top_k = saved_top_k

    # Aggregate by partition
    aggregated = aggregate_by_partition(per_query, k_values)
    aggregated["config_id"] = pipeline_cfg.id
    aggregated["config_name"] = pipeline_cfg.name
    aggregated["total_time_s"] = round(total_time, 2)
    aggregated["n_queries"] = len(per_query)

    return {
        "config": pipeline_cfg.id,
        "config_name": pipeline_cfg.name,
        "per_query": per_query,
        "aggregated": aggregated,
        "k_values": k_values,
        "errors": errors,
    }


def run_config_routing(rag, queries: List[dict]) -> dict:
    """Run routing evaluation (config-independent).

    Returns per_query routing results with confidence.
    """
    rag._query_ops._ensure_query_engine()
    rag._hybrid_ops.ensure_hybrid_engine()
    router = rag._hybrid_engine.router

    per_query = []
    for item in queries:
        truth = get_ground_truth_source(item["type"])
        routing = router.route(item["query"])
        pred = routing.source.value
        conf = routing.confidence

        per_query.append({
            "id": item["id"],
            "type": item["type"],
            "query": item["query"],
            "partition": item.get("partition", ""),
            "truth": truth,
            "predicted": pred,
            "confidence": conf,
            "correct": truth == pred,
        })

    return per_query
