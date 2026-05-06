"""
Pipeline configurations for ablation study.

Defines 7 configurations (C1-C7) that control which components
are active during retrieval. Each configuration can be applied
at runtime by overriding the loaded YAML config.

Usage:
    from validation.evaluation.configurations import CONFIGS, get_config, apply_config

    cfg = get_config("C6")
    retriever, postprocessors, preprocessor = apply_config(cfg, rag)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PipelineConfig:
    """One ablation configuration for the retrieval pipeline."""

    id: str               # "C1" .. "C7"
    name: str             # human-readable name
    use_bm25: bool        # activate BM25 retriever
    use_vector: bool      # activate vector retriever
    use_rrf: bool         # use Reciprocal Rank Fusion
    use_preprocessor: bool  # metadata-aware query preprocessor
    use_reranker: bool    # cross-encoder reranker
    use_crag: bool        # Corrective RAG
    alpha: float          # RRF weight: 0=BM25-only, 1=vector-only, 0.5=balanced
    description: str = ""


# ──────────────────────────────────────────────────────────────
# The 7 configurations for the ablation study
# ──────────────────────────────────────────────────────────────

CONFIGS: Dict[str, PipelineConfig] = {
    "C1": PipelineConfig(
        id="C1",
        name="bm25_only",
        use_bm25=True,
        use_vector=False,
        use_rrf=False,
        use_preprocessor=False,
        use_reranker=False,
        use_crag=False,
        alpha=0.0,
        description="BM25 lexical search only (baseline sparse).",
    ),
    "C2": PipelineConfig(
        id="C2",
        name="vector_only",
        use_vector=True,
        use_bm25=False,
        use_rrf=False,
        use_preprocessor=False,
        use_reranker=False,
        use_crag=False,
        alpha=1.0,
        description="Dense vector search only (baseline dense).",
    ),
    "C3": PipelineConfig(
        id="C3",
        name="hybrid_rrf",
        use_bm25=True,
        use_vector=True,
        use_rrf=True,
        use_preprocessor=False,
        use_reranker=False,
        use_crag=False,
        alpha=0.5,
        description="Hybrid BM25+vector with RRF fusion, no enhancements.",
    ),
    "C4": PipelineConfig(
        id="C4",
        name="hybrid_preproc",
        use_bm25=True,
        use_vector=True,
        use_rrf=True,
        use_preprocessor=True,
        use_reranker=False,
        use_crag=False,
        alpha=0.5,
        description="Hybrid + query preprocessor (metadata pre-filter).",
    ),
    "C5": PipelineConfig(
        id="C5",
        name="hybrid_reranker",
        use_bm25=True,
        use_vector=True,
        use_rrf=True,
        use_preprocessor=False,
        use_reranker=True,
        use_crag=False,
        alpha=0.5,
        description="Hybrid + cross-encoder reranker, no preprocessor.",
    ),
    "C6": PipelineConfig(
        id="C6",
        name="full_pipeline",
        use_bm25=True,
        use_vector=True,
        use_rrf=True,
        use_preprocessor=True,
        use_reranker=True,
        use_crag=False,
        alpha=0.6,  # production alpha
        description="Full pipeline: hybrid + preprocessor + reranker.",
    ),
    "C7": PipelineConfig(
        id="C7",
        name="naive_rag",
        use_bm25=False,
        use_vector=True,
        use_rrf=False,
        use_preprocessor=False,
        use_reranker=False,
        use_crag=False,
        alpha=1.0,
        description="Naive RAG baseline: vector-only, no metadata, no reranker.",
    ),
}

# Additional CRAG variant for Thesis 3 analysis
CONFIGS["C6_crag"] = PipelineConfig(
    id="C6_crag",
    name="full_pipeline_crag",
    use_bm25=True,
    use_vector=True,
    use_rrf=True,
    use_preprocessor=True,
    use_reranker=True,
    use_crag=True,
    alpha=0.6,
    description="Full pipeline + Corrective RAG (for CRAG impact analysis).",
)

# Ordered list for consistent iteration
CONFIG_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
CONFIG_ORDER_WITH_CRAG = CONFIG_ORDER + ["C6_crag"]


def get_config(config_id: str) -> PipelineConfig:
    """Get a pipeline configuration by ID.

    Args:
        config_id: Configuration identifier (e.g., "C1", "C6", "C6_crag")

    Returns:
        PipelineConfig instance

    Raises:
        KeyError: if config_id is not recognized
    """
    config_id = config_id.upper()
    if config_id not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS.keys()))
        raise KeyError(f"Unknown config '{config_id}'. Valid: {valid}")
    return CONFIGS[config_id]


def get_all_configs(include_crag: bool = False) -> List[PipelineConfig]:
    """Return configurations in canonical order.

    Args:
        include_crag: if True, also includes C6_crag variant
    """
    order = CONFIG_ORDER_WITH_CRAG if include_crag else CONFIG_ORDER
    return [CONFIGS[cid] for cid in order]


def apply_config(pipeline_cfg: PipelineConfig, rag):
    """Apply a PipelineConfig to a loaded RAGFramework instance.

    Ensures the query engine is built, then returns the retriever
    components with selective enable/disable based on the config.

    Returns:
        (retriever, postprocessors, preprocessor, original_alpha) tuple.
    """
    # Ensure query engine is built so we have access to retriever components
    rag._query_ops._ensure_query_engine()
    qe = rag._query_engine

    hybrid_retriever = qe.retriever
    all_postprocessors = qe.node_postprocessors or []
    query_preprocessor = getattr(qe, "query_preprocessor", None)

    # ── Postprocessors (reranker) ──
    postprocessors = all_postprocessors if pipeline_cfg.use_reranker else []

    # ── Preprocessor (metadata filter) ──
    preprocessor = query_preprocessor if pipeline_cfg.use_preprocessor else None

    # ── Alpha override for RRF ──
    # The HybridRetriever uses config.alpha for RRF weighting.
    # We temporarily override it for the evaluation run.
    original_alpha = None
    if hasattr(hybrid_retriever, "config"):
        original_alpha = hybrid_retriever.config.alpha
        hybrid_retriever.config.alpha = pipeline_cfg.alpha

    return hybrid_retriever, postprocessors, preprocessor, original_alpha


def restore_alpha(retriever, original_alpha):
    """Restore the original alpha value after evaluation."""
    if original_alpha is not None and hasattr(retriever, "config"):
        retriever.config.alpha = original_alpha


def describe_configs() -> str:
    """Return a formatted table describing all configurations."""
    lines = [
        f"{'ID':<10} {'Name':<22} {'BM25':>5} {'Vec':>5} {'RRF':>5} "
        f"{'Prep':>5} {'Rank':>5} {'CRAG':>5} {'α':>5}",
        "-" * 75,
    ]
    for cfg in get_all_configs(include_crag=True):
        yn = lambda b: "✓" if b else "✗"
        lines.append(
            f"{cfg.id:<10} {cfg.name:<22} {yn(cfg.use_bm25):>5} "
            f"{yn(cfg.use_vector):>5} {yn(cfg.use_rrf):>5} "
            f"{yn(cfg.use_preprocessor):>5} {yn(cfg.use_reranker):>5} "
            f"{yn(cfg.use_crag):>5} {cfg.alpha:>5.1f}"
        )
    return "\n".join(lines)
