"""
RAG Framework Operations Package.

This package contains modular components for RAG framework operations,
following the Single Responsibility Principle for better maintainability
and testability.
"""

from rag_framework.operations.lifecycle import LifecycleManager
from rag_framework.operations.index import IndexOperations
from rag_framework.operations.query import QueryOperations
from rag_framework.operations.hybrid import HybridOperations
from rag_framework.operations.config import ConfigOperations

__all__ = [
    "LifecycleManager",
    "IndexOperations",
    "QueryOperations",
    "HybridOperations",
    "ConfigOperations",
]
