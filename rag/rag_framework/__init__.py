"""
RAG Framework - A flexible and professional RAG system framework.

This framework provides:
- Multiple LLM providers (Ollama, HuggingFace)
- Multiple embedding providers (Ollama, HuggingFace)
- Multiple vector store backends (LanceDB, Chroma, FAISS, Qdrant, Pinecone)
- SQL database querying with natural language
- Intelligent query routing (Documents, SQL, or Hybrid)
- Flexible prompt template system
- YAML-based configuration with validation

Example usage:
    from rag_framework import RAGFramework, RAGConfig

    # Using default config
    rag = RAGFramework()

    # Using custom config file
    rag = RAGFramework.from_yaml("config.yaml")

    # Query with automatic routing
    response = rag.query("How many users are there?")  # Routes to SQL
    response = rag.query("What is the policy?")        # Routes to documents

    # Force specific source
    response = rag.query_documents("Explain the concept")
    response = rag.query_sql("List all products")
    response = rag.query_hybrid("Show sales and explain trends")

Default Configuration:
    - LLM: Ollama with llama3-instruct-8k
    - Embedding: Ollama with nomic-embed-text:v1.5
    - Reranker: BAAI/bge-reranker-base
    - Vector Store: LanceDB
    - Router: Disabled by default (enable in config)
    - SQL: Disabled by default (enable in config)
"""

from rag_framework.config import (
    RAGConfig,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    SQLConfig,
    RouterConfig,
    SourceType,
)
from rag_framework.framework import RAGFramework
from rag_framework.prompts import PromptTemplates, PromptTemplate
from rag_framework.utils.constants import FRAMEWORK_VERSION

__version__ = FRAMEWORK_VERSION
__all__ = [
    "RAGFramework",
    "RAGConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "RetrievalConfig",
    "SQLConfig",
    "RouterConfig",
    "SourceType",
    "PromptTemplates",
    "PromptTemplate",
    "FRAMEWORK_VERSION",
]
