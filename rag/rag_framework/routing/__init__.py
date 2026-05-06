"""
Query Routing Module.

Provides LLM-based routing of user queries to appropriate data sources:
- Unstructured: Document-based RAG retrieval
- Structured: SQL database queries
- Hybrid: Combination of both sources

The router uses the configured LLM to classify every query, ensuring
accurate and context-aware routing decisions.
"""

from rag_framework.routing.router import QueryRouter, RoutingResult
from rag_framework.config.models import SourceType

__all__ = [
    "QueryRouter",
    "SourceType",
    "RoutingResult",
]
