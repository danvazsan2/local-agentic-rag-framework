"""
Retrieval and router configuration dataclasses.
"""

from dataclasses import dataclass, field
from typing import List

from rag_framework.utils.constants import DEFAULT_TOP_K
from rag_framework.config.reranker_config import RerankerConfig


@dataclass
class RouterConfig:
    """Configuration for query routing.

    Supports a 3-layer routing strategy:
      1. **Keyword rules** (fast, no LLM) — fires only when confidence is
         above ``keyword_confidence_threshold``.
      2. **LLM classification** — when rules are inconclusive.
      3. **Post-execution fallback** — retries with an alternative source
         when the primary source returns empty results.
    """

    # Enable routing (if disabled, always use unstructured)
    enabled: bool = False

    # Default source when LLM routing fails or is unavailable
    default_source: str = "unstructured"

    # Use LLM as fallback when routing fails
    use_llm_fallback: bool = True

    # Confidence threshold for LLM routing (below this → consider HYBRID)
    confidence_threshold: float = 0.7

    # --- Layer 1: keyword-based routing ---
    # Enable fast keyword-based routing before calling the LLM
    use_keyword_routing: bool = True

    # Minimum confidence to accept the keyword routing decision without LLM.
    # 0.0–1.0; higher = more conservative (only very clear matches skip LLM).
    keyword_confidence_threshold: float = 0.8

    # Domain-specific keywords that indicate SQL queries
    structured_keywords: List[str] = field(default_factory=list)

    # Domain-specific keywords that indicate document queries
    unstructured_keywords: List[str] = field(default_factory=list)

    # --- Layer 3: post-execution fallback ---
    # When the primary source returns empty results (e.g. SQL 0 rows),
    # automatically retry with an alternative source.
    fallback_on_empty: bool = True

    # Strategy: "try_unstructured", "try_hybrid", "none"
    fallback_strategy: str = "try_unstructured"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval settings."""

    # Search type
    use_hybrid_search: bool = True

    # Retrieval parameters
    top_k: int = DEFAULT_TOP_K

    # Hybrid search parameters
    alpha: float = 0.5  # 0=only BM25, 1=only vector
    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    # Reranker
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
