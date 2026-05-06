"""
Reranker Provider abstraction for HuggingFace models.

Supports:
- HuggingFace reranker models (local/remote)
- Local models (already downloaded)
"""

import logging
from typing import Any, Dict, List, Optional, Type
from pathlib import Path

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from rag_framework.config.reranker_config import RerankerConfig
from rag_framework.providers.base import BaseRerankerProvider, ProviderFactory
from rag_framework.providers.common import HuggingFaceValidator

logger = logging.getLogger(__name__)


class HuggingFaceRerankerProvider(BaseRerankerProvider):
    """HuggingFace reranker provider for local/remote models."""

    # Popular HuggingFace reranker models
    RECOMMENDED_MODELS = {
        "bge-reranker-base": "BAAI/bge-reranker-base",
        "bge-reranker-large": "BAAI/bge-reranker-large",
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    }

    def __init__(self, config: RerankerConfig):
        self.config = config

    def get_reranker(self) -> Any:
        """Get HuggingFace reranker instance."""
        from llama_index.postprocessor.flag_embedding_reranker import (
            FlagEmbeddingReranker,
        )

        # Determine if using local or remote model
        if self.config.is_local and self.config.local_model_path:
            model_id = self.config.local_model_path
            logger.info(f"Configuring Local HuggingFace Reranker: {model_id}")
            logger.debug("  Source: Local model (offline)")
        else:
            model_id = self.config.model

            # Check if using a shorthand
            if model_id in self.RECOMMENDED_MODELS:
                model_id = self.RECOMMENDED_MODELS[model_id]

            logger.info(f"Configuring HuggingFace Reranker: {model_id}")
            logger.debug("  Source: HuggingFace Hub")

        logger.debug(f"  device: {self.config.device}")
        logger.debug(f"  top_n: {self.config.top_n}")

        # FlagEmbeddingReranker doesn't accept device parameter directly
        # It will use the model's device automatically
        reranker = FlagEmbeddingReranker(
            model=model_id,
            top_n=self.config.top_n,
        )

        return reranker

    def _get_device(self) -> str:
        """Determine the best available device."""
        return HuggingFaceValidator.get_best_device(self.config.device)

    def validate(self) -> bool:
        """Validate HuggingFace reranker model availability."""
        # If using local model, check if path exists
        if self.config.is_local and self.config.local_model_path:
            model_path = Path(self.config.local_model_path)
            if model_path.exists():
                logger.info(
                    f"Local reranker model found at: {self.config.local_model_path}"
                )
                return True
            else:
                logger.warning(
                    f"Local reranker model not found at: {self.config.local_model_path}"
                )
                return False

        # For remote models, check HuggingFace Hub
        model_id = self.config.model
        # Handle shorthands
        if model_id in self.RECOMMENDED_MODELS:
            model_id = self.RECOMMENDED_MODELS[model_id]

        if HuggingFaceValidator.check_model(model_id):
            return True

        logger.warning(f"Cannot validate HuggingFace reranker model: {model_id}")
        return True  # Allow to proceed, may work with cached model


class OllamaRerankerNodePostprocessor(BaseNodePostprocessor):
    """LlamaIndex postprocessor backed by the Ollama /api/rerank endpoint."""

    model: str
    top_n: int
    base_url: str

    def __init__(
        self, model: str, top_n: int, base_url: str = "http://localhost:11434"
    ):
        super().__init__(model=model, top_n=top_n, base_url=base_url)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes:
            return nodes
        if not query_bundle:
            return nodes[: self.top_n]

        import requests

        documents = [n.node.get_content() for n in nodes]
        try:
            resp = requests.post(
                f"{self.base_url.rstrip('/')}/api/rerank",
                json={
                    "model": self.model,
                    "query": query_bundle.query_str,
                    "documents": documents,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        except Exception as e:
            logger.warning(
                f"OllamaReranker: request failed ({e}), returning top-{self.top_n} in original order"
            )
            return nodes[: self.top_n]

        scored = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        return [
            NodeWithScore(node=nodes[r["index"]].node, score=r["relevance_score"])
            for r in scored[: self.top_n]
        ]


class OllamaRerankerProvider(BaseRerankerProvider):
    """Ollama reranker provider — calls the Ollama /api/rerank REST endpoint."""

    def __init__(self, config: RerankerConfig):
        self.config = config

    def get_reranker(self) -> Any:
        logger.info(
            f"Configuring Ollama Reranker: {self.config.model} (top_n={self.config.top_n})"
        )
        return OllamaRerankerNodePostprocessor(
            model=self.config.model,
            top_n=self.config.top_n,
            base_url=self.config.base_url,
        )

    def validate(self) -> bool:
        import requests

        # 1. Check Ollama is up
        try:
            resp = requests.get(
                f"{self.config.base_url.rstrip('/')}/api/tags", timeout=5.0
            )
            if resp.status_code != 200:
                logger.warning(
                    f"OllamaReranker: Ollama at {self.config.base_url} returned {resp.status_code}"
                )
                return False
        except Exception:
            logger.warning(
                f"OllamaReranker: cannot connect to Ollama at {self.config.base_url}"
            )
            return False

        # 2. Probe /api/rerank with a minimal request to detect 404 early
        try:
            probe = requests.post(
                f"{self.config.base_url.rstrip('/')}/api/rerank",
                json={
                    "model": self.config.model,
                    "query": "test",
                    "documents": ["test"],
                },
                timeout=10.0,
            )
            if probe.status_code == 404:
                logger.error(
                    f"OllamaReranker: /api/rerank returned 404. "
                    f"Your Ollama version does not support the rerank endpoint (requires >= 0.5.1). "
                    f"Switch to provider: huggingface with model: BAAI/bge-reranker-v2-m3 instead."
                )
                return False
        except Exception as e:
            logger.warning(f"OllamaReranker: could not probe /api/rerank ({e})")

        return True


class RerankerFactory(ProviderFactory[RerankerConfig, BaseRerankerProvider]):
    """Factory for creating reranker provider instances."""

    _registry: Dict[str, Type[BaseRerankerProvider]] = {}

    @classmethod
    def get_reranker(cls, config: RerankerConfig) -> Any:
        """
        Convenience method to directly get the reranker instance.

        Args:
            config: Reranker configuration

        Returns:
            Reranker instance compatible with LlamaIndex
        """
        provider = cls.create(config)
        return provider.get_reranker()


# Auto-registration
RerankerFactory.register("huggingface", HuggingFaceRerankerProvider)
RerankerFactory.register("ollama", OllamaRerankerProvider)
