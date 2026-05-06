"""
Component validators for RAG framework configuration.

This module provides validation functions that use the generic
ProviderFactory.validate() method, eliminating code duplication.
"""

import logging
from typing import Tuple

from rag_framework.config import RAGConfig
from rag_framework.providers import (
    LLMFactory,
    EmbeddingFactory,
    VectorStoreFactory,
    RerankerFactory,
)

logger = logging.getLogger(__name__)


def validate_llm_config(config: RAGConfig) -> bool:
    """Validate LLM provider configuration."""
    return LLMFactory.validate(config.llm)


def validate_embedding_config(config: RAGConfig) -> bool:
    """Validate embedding provider configuration."""
    return EmbeddingFactory.validate(config.embedding)


def validate_vector_store_config(config: RAGConfig) -> bool:
    """Validate vector store provider configuration."""
    return VectorStoreFactory.validate(config.vector_store)


def validate_reranker_config(config: RAGConfig) -> bool:
    """Validate reranker provider configuration (if enabled)."""
    if not config.retrieval.reranker.enabled:
        return True
    return RerankerFactory.validate(config.retrieval.reranker)


def validate_all_providers(config: RAGConfig) -> bool:
    """Validate all provider configurations.

    Args:
        config: Complete RAG framework configuration

    Returns:
        True if all enabled providers are valid, False otherwise
    """
    validators = [
        ("LLM", validate_llm_config),
        ("Embedding", validate_embedding_config),
        ("VectorStore", validate_vector_store_config),
        ("Reranker", validate_reranker_config),
    ]

    all_valid = True
    for name, validator in validators:
        if not validator(config):
            logger.error(f"{name} validation failed")
            all_valid = False

    return all_valid


class ModelValidator:
    """Backward-compatible wrapper around the new validation functions.

    .. deprecated::
        Use the module-level functions directly instead.
    """

    def __init__(self, config: RAGConfig):
        self.config = config

    def validate_all_models(self) -> bool:
        """Validate all configured models."""
        return validate_all_providers(self.config)

    def validate_llm_provider(self) -> bool:
        """Validate LLM provider."""
        return validate_llm_config(self.config)

    def validate_embedding_provider(self) -> bool:
        """Validate embedding provider."""
        return validate_embedding_config(self.config)

    def validate_reranker_provider(self) -> bool:
        """Validate reranker provider."""
        return validate_reranker_config(self.config)

    def validate_with_details(self) -> Tuple[bool, dict]:
        """Validate models and return detailed results."""
        results = {
            "llm": self.validate_llm_provider(),
            "embedding": self.validate_embedding_provider(),
            "reranker": self.validate_reranker_provider(),
        }
        return all(results.values()), results
