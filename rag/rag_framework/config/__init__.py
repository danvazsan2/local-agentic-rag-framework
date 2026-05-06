"""
Configuration module for the RAG Framework.

Provides dataclasses for type-safe configuration and YAML loading/saving.
All config classes are re-exported here for convenient access.
"""

from rag_framework.config.enums import (
    LLMProvider,
    EmbeddingProvider,
    VectorStoreProvider,
    DatabaseType,
    SourceType,
)
from rag_framework.config.llm_config import LLMConfig
from rag_framework.config.embedding_config import EmbeddingConfig
from rag_framework.config.vector_store_config import VectorStoreConfig
from rag_framework.config.chunking_config import ChunkingConfig
from rag_framework.config.reranker_config import RerankerConfig
from rag_framework.config.sql_config import (
    DatabaseConnectionConfig,
    SchemaConfig,
    SQLSecurityConfig,
    SQLConfig,
)
from rag_framework.config.retrieval_config import (
    RetrievalConfig,
    RouterConfig,
)
from rag_framework.config.corrective_rag_config import CorrectiveRAGConfig
from rag_framework.config.metadata_config import (
    MetadataExtractionConfig,
    MetadataFilterConfig,
    DbEnrichmentConfig,
    FilenamePatternConfig,
)
from rag_framework.config.rag_config import RAGConfig, DirectoryConfig
from rag_framework.config.loader import ConfigLoader

__all__ = [
    # Enums
    "LLMProvider",
    "EmbeddingProvider",
    "VectorStoreProvider",
    "DatabaseType",
    "SourceType",
    # Config dataclasses
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
    "MetadataFilterConfig",
    "DbEnrichmentConfig",
    "FilenamePatternConfig",
    "RAGConfig",
    "DirectoryConfig",
    # Loader
    "ConfigLoader",
]
