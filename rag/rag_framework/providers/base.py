"""
Abstract base classes for all providers.

Defines the contracts that concrete LLM, embedding, and reranker
providers must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Protocol, Type, TypeVar

from rag_framework.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

TConfig = TypeVar("TConfig")
TProvider = TypeVar("TProvider")


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_llm(self) -> Any:
        """Return the LLM instance compatible with LlamaIndex."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate that the LLM is accessible and working."""
        pass


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def get_embedding_model(self) -> Any:
        """Return the embedding model instance compatible with LlamaIndex."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate that the embedding model is accessible and working."""
        pass


class BaseRerankerProvider(ABC):
    """Abstract base class for reranker providers."""

    @abstractmethod
    def get_reranker(self) -> Any:
        """Return the reranker instance compatible with LlamaIndex."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate that the reranker model is accessible and working."""
        pass


class Validatable(Protocol):
    """Protocol for providers that support validation."""

    def validate(self) -> bool:
        """Validate provider configuration and availability."""
        ...


class ProviderFactory(Generic[TConfig, TProvider]):
    """Generic factory for creating and validating providers.

    This base class eliminates duplication across LLM, Embedding,
    VectorStore, and Reranker factories by providing common registry
    and validation logic.
    """

    _registry: Dict[str, Type[TProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[TProvider]) -> None:
        """Register a provider implementation.

        Args:
            name: Provider identifier (should match config enum value)
            provider_class: Provider class to register
        """
        cls._registry[name] = provider_class

    @classmethod
    def create(cls, config: TConfig) -> TProvider:
        """Create a provider instance from configuration.

        Args:
            config: Provider configuration object

        Returns:
            Instantiated provider

        Raises:
            ConfigurationError: If provider is not registered
        """
        provider_name = getattr(config, "provider", None)
        if not provider_name:
            raise ConfigurationError("Config missing 'provider' attribute")

        provider_class = cls._registry.get(provider_name)
        if not provider_class:
            raise ConfigurationError(
                f"Unknown provider '{provider_name}' for {cls.__name__}"
            )

        return provider_class(config)

    @classmethod
    def validate(cls, config: TConfig) -> bool:
        """Validate a provider configuration.

        Creates the provider and calls its validate() method if available.

        Args:
            config: Provider configuration to validate

        Returns:
            True if validation succeeds, False otherwise
        """
        try:
            provider = cls.create(config)
            if hasattr(provider, "validate") and callable(provider.validate):
                return provider.validate()
            return True  # No validate method = assume valid
        except Exception as e:
            logger.error(f"Validation failed for {cls.__name__}: {e}")
            return False
