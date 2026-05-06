"""
Shared fixtures for the RAG Framework test suite.

Provides reusable test configurations, temporary directories,
and mock providers used across multiple test modules.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

from rag_framework.config.models import (
    RAGConfig,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    ChunkingConfig,
    RetrievalConfig,
    RerankerConfig,
    DirectoryConfig,
    SQLConfig,
    RouterConfig,
)
from rag_framework.providers.llm import BaseLLMProvider


# ─── Configuration Fixtures ───────────────────────────────────────


@pytest.fixture
def sample_llm_config():
    """Return a minimal LLMConfig with Ollama defaults."""
    return LLMConfig(
        provider="ollama",
        model="llama3-instruct-8k",
        base_url="http://localhost:11434",
        temperature=0.0,
        max_tokens=256,
    )


@pytest.fixture
def sample_rag_config(tmp_path):
    """Return a RAGConfig with test-safe values pointing to a temp directory.

    Uses ``tmp_path`` so filesystem side-effects are isolated.
    """
    docs_dir = str(tmp_path / "documents")
    vs_dir = str(tmp_path / "vector_store")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(vs_dir, exist_ok=True)

    return RAGConfig(
        llm=LLMConfig(
            provider="ollama",
            model="llama3-instruct-8k",
            base_url="http://localhost:11434",
            temperature=0.0,
            max_tokens=256,
        ),
        embedding=EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text:v1.5",
            base_url="http://localhost:11434",
        ),
        vector_store=VectorStoreConfig(
            provider="lancedb",
            persist_directory=vs_dir,
            collection_name="test_documents",
        ),
        chunking=ChunkingConfig(
            chunk_size=512,
            chunk_overlap=50,
        ),
        retrieval=RetrievalConfig(
            use_hybrid_search=True,
            top_k=5,
            reranker=RerankerConfig(enabled=False),
        ),
        directories=DirectoryConfig(
            documents_dir=docs_dir,
            vector_store_dir=vs_dir,
        ),
        sql=SQLConfig(enabled=False),
        router=RouterConfig(enabled=False),
        debug=False,
    )


# ─── Directory Fixtures ───────────────────────────────────────────


@pytest.fixture
def temp_directory():
    """Create a temporary directory that is cleaned up after the test."""
    dirpath = tempfile.mkdtemp(prefix="rag_test_")
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)


# ─── Mock Provider Fixtures ──────────────────────────────────────


@pytest.fixture
def mock_llm_provider():
    """Return a MagicMock that satisfies the BaseLLMProvider interface.

    * ``get_llm()`` returns a MagicMock (stand-in for a LlamaIndex LLM).
    * ``validate()`` returns True.
    """
    provider = MagicMock(spec=BaseLLMProvider)
    provider.get_llm.return_value = MagicMock(name="FakeLLM")
    provider.validate.return_value = True
    return provider
