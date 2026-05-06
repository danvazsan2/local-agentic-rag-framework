"""
Abstract base class for vector store providers.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStoreProvider(ABC):
    """Abstract base class for vector store providers."""

    @abstractmethod
    def get_vector_store(self) -> Any:
        """Return the vector store instance compatible with LlamaIndex."""
        pass

    @abstractmethod
    def exists(self) -> bool:
        """Check if the vector store already has data."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the vector store."""
        pass
