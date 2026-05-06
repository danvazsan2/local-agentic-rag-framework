"""
Enumerations for the RAG Framework configuration.

Defines supported providers and source types used
across all configuration dataclasses.
"""

from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"  # Future support


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"  # Future support


class VectorStoreProvider(str, Enum):
    """Supported vector store providers."""

    LANCEDB = "lancedb"
    CHROMA = "chroma"
    FAISS = "faiss"
    QDRANT = "qdrant"
    PINECONE = "pinecone"


class DatabaseType(str, Enum):
    """Supported database types."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class SourceType(str, Enum):
    """Types of data sources for routing."""

    UNSTRUCTURED = "unstructured"  # Documents/RAG
    STRUCTURED = "structured"  # SQL/Database
    HYBRID = "hybrid"  # Both sources
