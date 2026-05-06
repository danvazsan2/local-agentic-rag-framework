"""
Utilities module for the RAG Framework.

Provides shared utilities, constants, and helper functions.
"""

from rag_framework.utils.constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_API_PORT,
    FRAMEWORK_VERSION,
)
from rag_framework.utils.device import get_best_device
from rag_framework.utils.logging import setup_logging, get_logger

__all__ = [
    # Constants
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBEDDING_MODEL", 
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_API_PORT",
    "FRAMEWORK_VERSION",
    # Utilities
    "get_best_device",
    "setup_logging",
    "get_logger",
]
