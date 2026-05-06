"""
Hybrid Component Operations for RAG Framework.

Handles lazy initialization and management of hybrid query components
including the hybrid engine, SQL agent, and query router.
"""

from typing import TYPE_CHECKING, Optional
import logging

if TYPE_CHECKING:
    from rag_framework.framework import RAGFramework

logger = logging.getLogger(__name__)


class HybridOperations:
    """
    Manages hybrid component initialization and lifecycle.

    Responsibilities:
    - Lazy initialization of hybrid query engine
    - Lazy initialization of SQL agent
    - Lazy initialization of query router
    - Component dependency management

    Uses lazy loading to avoid unnecessary initialization of components
    that may not be needed for basic document-only queries.
    """

    def __init__(self, framework: "RAGFramework") -> None:
        """
        Initialize hybrid operations manager.

        Args:
            framework: Reference to parent RAGFramework instance.
        """
        self.framework = framework

    def ensure_hybrid_engine(self) -> None:
        """
        Ensure hybrid query engine is initialized.

        Initializes the engine if not already created and sets up
        all required dependencies (document RAG components).
        """
        if self.framework._hybrid_engine is None:
            from rag_framework.core.hybrid_engine import HybridQueryEngine

            # Ensure document RAG is ready
            self.framework._index_ops.ensure_index_loaded()
            self.framework._query_ops._ensure_query_engine()

            self.framework._hybrid_engine = HybridQueryEngine(
                config=self.framework.config,
                rag_query_engine=self.framework._query_engine,
            )

            logger.info("Hybrid query engine initialized")

    def ensure_sql_agent(self) -> None:
        """
        Ensure SQL agent is initialized.

        Creates the SQL agent if not already instantiated,
        enabling SQL database queries.
        """
        if self.framework._sql_agent is None:
            from rag_framework.sql import SQLAgent

            self.framework._sql_agent = SQLAgent(self.framework.config)
            logger.info("SQL agent initialized")

    def ensure_router(self) -> None:
        """
        Ensure query router is initialized.

        Creates the router if not already instantiated and updates
        it with SQL schema information if SQL is enabled.
        """
        if self.framework._router is None:
            from rag_framework.routing import QueryRouter

            self.framework._router = QueryRouter(self.framework.config)

            # Update with schema if SQL is enabled
            if self.framework.config.sql.enabled:
                self.ensure_sql_agent()
                schema_info = self.framework._sql_agent.get_schema_for_router()
                self.framework._router.update_schema_info(schema_info)

            logger.info("Query router initialized")

    def reset_hybrid_components(self) -> None:
        """
        Reset all hybrid components to None.

        Forces re-initialization on next use. Useful when configuration
        changes or when components need to be recreated.
        """
        self.framework._hybrid_engine = None
        self.framework._sql_agent = None
        self.framework._router = None
        logger.debug("Hybrid components reset")

    @property
    def has_hybrid_engine(self) -> bool:
        """Check if hybrid engine is initialized."""
        return self.framework._hybrid_engine is not None

    @property
    def has_sql_agent(self) -> bool:
        """Check if SQL agent is initialized."""
        return self.framework._sql_agent is not None

    @property
    def has_router(self) -> bool:
        """Check if router is initialized."""
        return self.framework._router is not None
