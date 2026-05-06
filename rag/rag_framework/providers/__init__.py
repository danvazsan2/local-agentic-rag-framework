"""
Providers module for LLM, Embedding, Reranker, and Vector Store abstractions.
"""

from rag_framework.providers.base import (
    BaseLLMProvider,
    BaseEmbeddingProvider,
    BaseRerankerProvider,
    ProviderFactory,
)
from rag_framework.providers.llm import (
    LLMFactory,
    OllamaLLMProvider,
    HuggingFaceLLMProvider,
)
from rag_framework.providers.embeddings import (
    EmbeddingFactory,
    OllamaEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
)
from rag_framework.providers.reranker import (
    RerankerFactory,
    HuggingFaceRerankerProvider,
)
from rag_framework.providers.vector_stores import (
    VectorStoreFactory,
    BaseVectorStoreProvider,
)
from rag_framework.providers.common import (
    OllamaValidator,
    HuggingFaceValidator,
)

__all__ = [
    # ABCs
    "BaseLLMProvider",
    "BaseEmbeddingProvider",
    "BaseRerankerProvider",
    "BaseVectorStoreProvider",
    # Generic factory
    "ProviderFactory",
    # Factories
    "LLMFactory",
    "EmbeddingFactory",
    "RerankerFactory",
    "VectorStoreFactory",
    # Concrete providers
    "OllamaLLMProvider",
    "HuggingFaceLLMProvider",
    "OllamaEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "HuggingFaceRerankerProvider",
    # Validators
    "OllamaValidator",
    "HuggingFaceValidator",
]
