"""
Backward-compatibility shim for rag_framework.config.models.

All classes have been moved to dedicated modules under rag_framework/config/.
This file re-exports them so existing ``from rag_framework.config.models import X``
statements keep working without modification.

NEW code should import from ``rag_framework.config`` or from the specific
sub-module (e.g. ``rag_framework.config.llm_config``).
"""

# --- Enums ----------------------------------------------------------------
from rag_framework.config.enums import (  # noqa: F401
    LLMProvider,
    EmbeddingProvider,
    VectorStoreProvider,
    DatabaseType,
    SourceType,
)

# --- Config dataclasses ---------------------------------------------------
from rag_framework.config.llm_config import LLMConfig  # noqa: F401
from rag_framework.config.embedding_config import EmbeddingConfig  # noqa: F401
from rag_framework.config.vector_store_config import VectorStoreConfig  # noqa: F401
from rag_framework.config.chunking_config import ChunkingConfig  # noqa: F401
from rag_framework.config.reranker_config import RerankerConfig  # noqa: F401
from rag_framework.config.sql_config import (  # noqa: F401
    DatabaseConnectionConfig,
    SchemaConfig,
    SQLSecurityConfig,
    SQLConfig,
)
from rag_framework.config.retrieval_config import (  # noqa: F401
    RetrievalConfig,
    RouterConfig,
)
from rag_framework.config.corrective_rag_config import CorrectiveRAGConfig  # noqa: F401
from rag_framework.config.metadata_config import MetadataExtractionConfig  # noqa: F401
from rag_framework.config.rag_config import RAGConfig, DirectoryConfig  # noqa: F401

__all__ = [
    "LLMProvider",
    "EmbeddingProvider",
    "VectorStoreProvider",
    "DatabaseType",
    "SourceType",
    "LLMConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "ChunkingConfig",
    "RerankerConfig",
    "DatabaseConnectionConfig",
    "SchemaConfig",
    "SQLSecurityConfig",
    "SQLConfig",
    "RetrievalConfig",
    "RouterConfig",
    "CorrectiveRAGConfig",
    "MetadataExtractionConfig",
    "RAGConfig",
    "DirectoryConfig",
]
