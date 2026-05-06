"""
RAG Framework - Main framework class.

Provides a unified interface for building and using RAG systems.
Supports both document-based retrieval and SQL database queries through
intelligent routing.

This module provides the main RAGFramework class which uses composition
to delegate responsibilities to specialized operation managers:
- LifecycleManager: Initialization and factory methods
- IndexOperations: Document ingestion and index management
- QueryOperations: Query processing and chat interface
- HybridOperations: Hybrid component initialization
- ConfigOperations: Configuration management
"""

from typing import Optional, Any, Union
import logging

from rag_framework.config import RAGConfig
from rag_framework.operations import (
    LifecycleManager,
    IndexOperations,
    QueryOperations,
    HybridOperations,
    ConfigOperations,
)

# Configure logger
logger = logging.getLogger(__name__)


class RAGFramework:
    """
    Main RAG Framework class providing a unified interface for RAG operations.

    This class manages the complete RAG pipeline including:
    - Document ingestion and processing
    - Vector index management
    - Query processing with hybrid search and reranking
    - SQL database queries (when enabled)
    - Intelligent routing between sources
    - Interactive chat interface

    The framework uses composition to delegate responsibilities to specialized
    operation managers, following the Single Responsibility Principle.

    Example:
        >>> # Initialize with default configuration
        >>> rag = RAGFramework()
        >>>
        >>> # Ingest documents
        >>> rag.ingest()
        >>>
        >>> # Query the system (auto-routed to appropriate source)
        >>> response = rag.query("What is the main topic?")
        >>>
        >>> # Force query to specific source
        >>> response = rag.query_documents("Explain the concept")
        >>> response = rag.query_sql("How many records are there?")
        >>>
        >>> # Start interactive chat
        >>> rag.chat()
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        """
        Initialize the RAG framework with the given configuration.

        Args:
            config: RAG configuration. If None, loads from default sources.

        Raises:
            RAGFrameworkError: If configuration validation fails.
        """
        # Initialize lifecycle manager first
        self._lifecycle_mgr = LifecycleManager(self)

        # Initialize and validate configuration
        self.config = self._lifecycle_mgr.initialize_from_config(config)

        # Initialize core component managers
        self._ingestion, self._index_manager, self._query_manager = (
            self._lifecycle_mgr.initialize_managers(self.config)
        )

        # Initialize state variables
        self._index, self._nodes, self._query_engine = self._lifecycle_mgr.reset_state()

        # Initialize hybrid components (lazy-loaded)
        self._hybrid_engine = None
        self._sql_agent = None
        self._router = None

        # Initialize operation managers (use composition pattern)
        self._index_ops = IndexOperations(self)
        self._query_ops = QueryOperations(self)
        self._hybrid_ops = HybridOperations(self)
        self._config_ops = ConfigOperations(self)

        # Display configuration summary
        self._lifecycle_mgr.display_initialization_summary(self.config)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_yaml(cls, config_path: str) -> "RAGFramework":
        """
        Create framework instance from YAML configuration file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Initialized RAGFramework instance.

        Raises:
            RAGFrameworkError: If config file cannot be loaded.
        """
        config = LifecycleManager.create_from_yaml(config_path)
        return cls(config)

    # =========================================================================
    # Index Management (delegates to IndexOperations)
    # =========================================================================

    def ingest(
        self, documents_dir: Optional[str] = None, force_reindex: bool = False
    ) -> "RAGFramework":
        """
        Ingest documents and create or update the vector index.

        This method handles the complete ingestion pipeline:
        1. Checks for existing index (unless force_reindex=True)
        2. Processes documents from the specified directory
        3. Creates or updates the vector index

        Args:
            documents_dir: Directory containing documents. Uses config default if None.
            force_reindex: Force re-indexing even if a valid index exists.

        Returns:
            Self for method chaining.

        Raises:
            DocumentIngestionError: If ingestion fails critically.
        """
        self._index, self._nodes = self._index_ops.ingest(documents_dir, force_reindex)
        return self

    def load_index(self) -> "RAGFramework":
        """
        Load an existing index without re-ingesting documents.

        This is useful when you want to query an existing index without
        reprocessing the source documents.

        Returns:
            Self for method chaining.

        Raises:
            IndexNotFoundError: If no index exists and loading fails.
        """
        self._index, self._nodes = self._index_ops.load_index()
        return self

    def clear_index(self) -> "RAGFramework":
        """
        Clear the vector index and reset framework state.

        This removes all indexed data and requires re-ingestion.

        Returns:
            Self for method chaining.
        """
        self._index_ops.clear_index()
        self._index, self._nodes, self._query_engine = self._lifecycle_mgr.reset_state()
        return self

    # =========================================================================
    # Query Methods (delegates to QueryOperations)
    # =========================================================================

    def query(self, question: str) -> Union[str, Any]:
        """
        Query the RAG system with a question.

        When routing is enabled, this method automatically routes the query
        to the appropriate source (documents, SQL, or both) based on the
        question's characteristics.

        When routing is disabled, it uses the traditional document-only RAG.

        Args:
            question: The question to answer.

        Returns:
            Generated response as a string (or HybridQueryResponse if routing enabled).

        Raises:
            IndexNotFoundError: If no index is available for document queries.
            ValueError: If question is empty.
        """
        return self._query_ops.query(question)

    def query_documents(self, question: str) -> str:
        """
        Force query to use only document retrieval (bypasses routing).

        Use this when you specifically need document-based answers
        regardless of the query characteristics.

        Args:
            question: The question to answer.

        Returns:
            Generated response from documents.
        """
        return self._query_ops.query_documents(question)

    def query_sql(self, question: str) -> Any:
        """
        Force query to use only SQL database (bypasses routing).

        Use this when you specifically need database results
        regardless of the query characteristics.

        Args:
            question: The question to answer.

        Returns:
            SQLQueryResult with query results.

        Raises:
            RAGFrameworkError: If SQL is not enabled.
        """
        return self._query_ops.query_sql(question)

    def query_hybrid(self, question: str) -> Any:
        """
        Force query to use both sources (bypasses routing).

        Use this when you know the query needs both database
        and document context.

        Args:
            question: The question to answer.

        Returns:
            Combined response from both sources.
        """
        return self._query_ops.query_hybrid(question)

    def chat(self, message: Optional[str] = None) -> Optional[str]:
        """
        Interact with the RAG system via chat interface.

        Two modes:
        1. Programmatic: Pass a message and get a response
        2. Interactive: Start a chat session with command loop

        Args:
            message: Optional single message for programmatic mode.

        Returns:
            Response string if message provided, None for interactive mode.

        Raises:
            IndexNotFoundError: If no index is available.
        """
        return self._query_ops.chat(message)

    # =========================================================================
    # Configuration Methods (delegates to ConfigOperations)
    # =========================================================================

    def set_prompt_template(self, template_name: str) -> "RAGFramework":
        """
        Change the active prompt template.

        Args:
            template_name: Name of the template to use.

        Returns:
            Self for method chaining.
        """
        self._config_ops.set_prompt_template(template_name)
        return self

    def set_custom_prompt(self, prompt_template: str) -> "RAGFramework":
        """
        Set a custom prompt template.

        The template should include {context_str} and {query_str} placeholders.

        Args:
            prompt_template: Custom prompt string with required placeholders.

        Returns:
            Self for method chaining.
        """
        self._config_ops.set_custom_prompt(prompt_template)
        return self

    def save_config(self, path: str) -> None:
        """
        Save current configuration to a YAML file.

        Args:
            path: Path where the configuration should be saved.

        Raises:
            RAGFrameworkError: If saving fails.
        """
        self._config_ops.save_config(path)

    def validate_models(self) -> bool:
        """
        Validate that all configured models are available and accessible.

        This checks:
        - LLM provider and model availability
        - Embedding provider and model availability

        Returns:
            True if all models are valid and accessible, False otherwise.
        """
        return self._config_ops.validate_models()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def index(self):
        """
        Get the current vector index.

        Returns:
            The loaded vector index or None.
        """
        return self._index

    @property
    def nodes(self):
        """
        Get the current document nodes.

        Returns:
            List of document nodes or None.
        """
        return self._nodes
