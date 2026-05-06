"""
Retrieval module with hybrid search and reranking support.
"""

import logging
import time
from typing import List, Optional, Tuple, Any

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import MetadataFilters

from rag_framework.config.models import RAGConfig, RetrievalConfig

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining vector search with BM25.

    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """

    def __init__(
        self, index: VectorStoreIndex, nodes: List[TextNode], config: RetrievalConfig
    ):
        """
        Initialize hybrid retriever.

        Args:
            index: Vector store index
            nodes: Original nodes for BM25
            config: Retrieval configuration
        """
        from llama_index.retrievers.bm25 import BM25Retriever

        self.index = index
        self.nodes = nodes
        self.config = config

        # Vector retriever
        self.vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=config.top_k,
        )

        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=config.top_k,
        )

    def retrieve(
        self,
        query: str,
        filters: Optional[MetadataFilters] = None,
        trace=None,
    ) -> List[NodeWithScore]:
        """
        Perform hybrid retrieval.

        Args:
            query: Search query
            filters: Optional metadata filters for pre-filtering
            trace: Optional QueryTrace for per-phase timing

        Returns:
            List of nodes with combined scores
        """
        # Apply metadata filters to vector retriever if provided
        if filters is not None:
            filtered_vector_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.config.top_k,
                filters=filters,
            )
            t0 = time.perf_counter()
            vector_results = filtered_vector_retriever.retrieve(query)
            if trace is not None:
                trace.phases["vector_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            t0 = time.perf_counter()
            bm25_results = self.bm25_retriever.retrieve(query)
            bm25_results = self._manual_metadata_filter(bm25_results, filters)
            if trace is not None:
                trace.phases["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        else:
            t0 = time.perf_counter()
            vector_results = self.vector_retriever.retrieve(query)
            if trace is not None:
                trace.phases["vector_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            t0 = time.perf_counter()
            bm25_results = self.bm25_retriever.retrieve(query)
            if trace is not None:
                trace.phases["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Combine using RRF
        t0 = time.perf_counter()
        combined = self._reciprocal_rank_fusion(vector_results, bm25_results)
        if trace is not None:
            trace.phases["rrf_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return combined[: self.config.top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[NodeWithScore],
        bm25_results: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF Score = sum(1 / (k + rank))
        """
        k = self.config.rrf_k
        alpha = self.config.alpha

        node_scores = {}
        node_map = {}

        # Process vector results
        for rank, node_with_score in enumerate(vector_results):
            node_id = node_with_score.node.node_id
            rrf_score = alpha * (1 / (k + rank + 1))
            node_scores[node_id] = node_scores.get(node_id, 0) + rrf_score
            node_map[node_id] = node_with_score.node

        # Process BM25 results
        for rank, node_with_score in enumerate(bm25_results):
            node_id = node_with_score.node.node_id
            rrf_score = (1 - alpha) * (1 / (k + rank + 1))
            node_scores[node_id] = node_scores.get(node_id, 0) + rrf_score
            node_map[node_id] = node_with_score.node

        # Sort by combined score
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)

        # Create NodeWithScore objects
        return [
            NodeWithScore(node=node_map[node_id], score=score)
            for node_id, score in sorted_nodes
        ]

    @staticmethod
    def _manual_metadata_filter(
        nodes: List[NodeWithScore], filters: MetadataFilters
    ) -> List[NodeWithScore]:
        """Apply metadata filters manually (for retrievers that lack native support)."""
        filtered = []
        for nws in nodes:
            match = True
            for f in filters.filters:
                node_val = nws.node.metadata.get(f.key)
                if node_val != f.value:
                    match = False
                    break
            if match:
                filtered.append(nws)
        return filtered


class RetrieverManager:
    """
    Manages retriever creation and configuration.
    """

    def __init__(self, config: RAGConfig):
        """
        Initialize retriever manager.

        Args:
            config: RAG configuration
        """
        self.config = config
        self.retrieval_config = config.retrieval

    def create_retriever(
        self,
        index: VectorStoreIndex,
        nodes: Optional[List[TextNode]] = None,
    ) -> Tuple[Any, List[Any]]:
        """
        Create retriever with optional hybrid search and reranking.

        Args:
            index: Vector store index
            nodes: Original nodes (required for hybrid search)

        Returns:
            Tuple of (retriever, postprocessors)
        """
        logger.info("Configuring retriever")

        postprocessors = []

        # Create retriever
        use_hybrid = self.retrieval_config.use_hybrid_search and nodes is not None

        if use_hybrid:
            logger.debug("Retrieval mode: Hybrid Search (Vector + BM25)")
            retriever = HybridRetriever(
                index=index, nodes=nodes, config=self.retrieval_config
            )
        else:
            logger.debug("Retrieval mode: Vector Search")
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=self.retrieval_config.top_k,
            )

        # Add reranker if enabled
        if self.retrieval_config.reranker.enabled:
            logger.debug(f"Reranker: {self.retrieval_config.reranker.model}")
            reranker = self._create_reranker()
            postprocessors.append(reranker)

        logger.debug(
            f"Top-K: {self.retrieval_config.top_k}"
            + (
                f", Top-K after rerank: {self.retrieval_config.reranker.top_n}"
                if self.retrieval_config.reranker.enabled
                else ""
            )
        )

        return retriever, postprocessors

    def _create_reranker(self) -> Any:
        """Create the reranker postprocessor using the reranker provider."""
        from rag_framework.providers import RerankerFactory

        return RerankerFactory.get_reranker(self.retrieval_config.reranker)
