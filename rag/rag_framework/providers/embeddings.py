"""
Embedding Provider abstraction for multiple backends.

Supports:
- Ollama (local embeddings)
- HuggingFace (local/remote embeddings)
- OpenAI (future support)
"""

import logging
from typing import Any, Dict, Optional, Type

from rag_framework.config.embedding_config import EmbeddingConfig
from rag_framework.providers.base import BaseEmbeddingProvider, ProviderFactory
from rag_framework.providers.common import OllamaValidator, HuggingFaceValidator

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local models."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def get_embedding_model(self) -> Any:
        """Get Ollama embedding model instance."""
        from llama_index.embeddings.ollama import OllamaEmbedding

        logger.info(f"Configuring Ollama Embeddings: {self.config.model}")

        embed_model = OllamaEmbedding(
            model_name=self.config.model,
            base_url=self.config.base_url,
        )

        return embed_model

    def validate(self) -> bool:
        """Validate Ollama connection and model availability."""
        if not OllamaValidator.check_connection(self.config.base_url):
            logger.error(f"Cannot connect to Ollama at {self.config.base_url}")
            return False

        if not OllamaValidator.check_model(self.config.base_url, self.config.model):
            logger.warning(
                f"Embedding model {self.config.model} not found. Run: ollama pull {self.config.model}"
            )
            return False

        return True


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embedding provider for local/remote models."""

    # Popular HuggingFace embedding models
    RECOMMENDED_MODELS = {
        "small": "sentence-transformers/all-MiniLM-L6-v2",
        "medium": "sentence-transformers/all-mpnet-base-v2",
        "large": "BAAI/bge-large-en-v1.5",
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    }

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def get_embedding_model(self) -> Any:
        """Get HuggingFace embedding model instance."""
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # Determine if using local or remote model
        if self.config.is_local and self.config.local_model_path:
            model_id = self.config.local_model_path
            logger.info(f"Configuring Local HuggingFace Embeddings: {model_id}")
            logger.debug("  Source: Local model (offline)")
        else:
            model_id = self.config.hf_model_id or self.config.model

            # Check if using a shorthand
            if model_id in self.RECOMMENDED_MODELS:
                model_id = self.RECOMMENDED_MODELS[model_id]

            logger.info(f"Configuring HuggingFace Embeddings: {model_id}")
            logger.debug("  Source: HuggingFace Hub")

        logger.debug(f"  device: {self.config.device}")

        # Determine device
        device = self._get_device()

        embed_kwargs = {}
        if self.config.hf_token and not self.config.is_local:
            embed_kwargs["token"] = self.config.hf_token

        embed_model = HuggingFaceEmbedding(
            model_name=model_id, device=device, **embed_kwargs
        )

        return embed_model

    def _get_device(self) -> str:
        """Determine the best available device."""
        return HuggingFaceValidator.get_best_device(self.config.device)

    def validate(self) -> bool:
        """Validate HuggingFace model availability."""
        # If using local model, check if path exists
        if self.config.is_local and self.config.local_model_path:
            from pathlib import Path

            model_path = Path(self.config.local_model_path)
            if model_path.exists():
                logger.info(f"Local embedding model found at: {self.config.local_model_path}")
                return True
            else:
                logger.error(
                    f"Local embedding model not found at: {self.config.local_model_path}"
                )
                return False

        # For remote models, check HuggingFace Hub
        model_id = self.config.hf_model_id or self.config.model
        # Handle shorthands
        if model_id in self.RECOMMENDED_MODELS:
            model_id = self.RECOMMENDED_MODELS[model_id]

        if HuggingFaceValidator.check_model(model_id, token=self.config.hf_token):
            return True

        logger.warning(f"Cannot validate HuggingFace embedding model: {model_id}")
        return True  # Allow to proceed, may work with cached model


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider (for future support)."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def get_embedding_model(self) -> Any:
        """Get OpenAI embedding model instance."""
        raise NotImplementedError(
            "OpenAI embeddings support is not yet implemented. "
            "Please use 'ollama' or 'huggingface' provider."
        )

    def validate(self) -> bool:
        """Validate OpenAI API key."""
        import os

        return bool(os.getenv("OPENAI_API_KEY"))


class EmbeddingFactory(ProviderFactory[EmbeddingConfig, BaseEmbeddingProvider]):
    """Factory for creating embedding provider instances."""

    _registry: Dict[str, Type[BaseEmbeddingProvider]] = {}

    @classmethod
    def get_embedding_model(cls, config: EmbeddingConfig) -> Any:
        """
        Convenience method to directly get the embedding model instance.

        Args:
            config: Embedding configuration

        Returns:
            Embedding model instance compatible with LlamaIndex
        """
        provider = cls.create(config)
        return provider.get_embedding_model()

    @classmethod
    def list_huggingface_recommendations(cls) -> dict:
        """List recommended HuggingFace embedding models."""
        return HuggingFaceEmbeddingProvider.RECOMMENDED_MODELS.copy()


# Auto-registration
EmbeddingFactory.register("ollama", OllamaEmbeddingProvider)
EmbeddingFactory.register("huggingface", HuggingFaceEmbeddingProvider)
EmbeddingFactory.register("openai", OpenAIEmbeddingProvider)
