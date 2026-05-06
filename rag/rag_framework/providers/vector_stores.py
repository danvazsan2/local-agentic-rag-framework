"""
Backward-compatibility shim for rag_framework.providers.vector_stores.

All classes have been moved to ``rag_framework/providers/vector_stores/`` package.
This file re-exports them so existing imports keep working.

NEW code should import from the package or individual sub-modules.
"""

from rag_framework.providers.vector_stores import (  # noqa: F401
    BaseVectorStoreProvider,
    LanceDBVectorStoreProvider,
    ChromaVectorStoreProvider,
    FAISSVectorStoreProvider,
    QdrantVectorStoreProvider,
    PineconeVectorStoreProvider,
    VectorStoreFactory,
)

__all__ = [
    "BaseVectorStoreProvider",
    "LanceDBVectorStoreProvider",
    "ChromaVectorStoreProvider",
    "FAISSVectorStoreProvider",
    "QdrantVectorStoreProvider",
    "PineconeVectorStoreProvider",
    "VectorStoreFactory",
]
