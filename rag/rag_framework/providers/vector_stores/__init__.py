"""Vector store provider implementations and factory."""

from typing import Any, Dict, List, Type

from rag_framework.config.vector_store_config import VectorStoreConfig
from rag_framework.providers.base import ProviderFactory
from rag_framework.providers.vector_stores.base import BaseVectorStoreProvider
from rag_framework.providers.vector_stores.lancedb import LanceDBVectorStoreProvider
from rag_framework.providers.vector_stores.chroma import ChromaVectorStoreProvider
from rag_framework.providers.vector_stores.faiss import FAISSVectorStoreProvider
from rag_framework.providers.vector_stores.qdrant import QdrantVectorStoreProvider
from rag_framework.providers.vector_stores.pinecone import PineconeVectorStoreProvider


class VectorStoreFactory(ProviderFactory[VectorStoreConfig, BaseVectorStoreProvider]):
    """Factory for creating vector store provider instances."""

    _registry: Dict[str, Type[BaseVectorStoreProvider]] = {}

    @classmethod
    def get_vector_store(cls, config: VectorStoreConfig) -> Any:
        """Convenience method to directly get the vector store instance.

        Args:
            config: Vector store configuration

        Returns:
            Vector store instance compatible with LlamaIndex
        """
        provider = cls.create(config)
        return provider.get_vector_store()

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all available vector store providers."""
        return list(cls._registry.keys())


# Auto-registration
VectorStoreFactory.register("lancedb", LanceDBVectorStoreProvider)
VectorStoreFactory.register("chroma", ChromaVectorStoreProvider)
VectorStoreFactory.register("faiss", FAISSVectorStoreProvider)
VectorStoreFactory.register("qdrant", QdrantVectorStoreProvider)
VectorStoreFactory.register("pinecone", PineconeVectorStoreProvider)


__all__ = [
    "BaseVectorStoreProvider",
    "LanceDBVectorStoreProvider",
    "ChromaVectorStoreProvider",
    "FAISSVectorStoreProvider",
    "QdrantVectorStoreProvider",
    "PineconeVectorStoreProvider",
    "VectorStoreFactory",
]
