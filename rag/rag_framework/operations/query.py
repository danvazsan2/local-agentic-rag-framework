"""
Query Operations for RAG Framework.

Handles all query processing operations including document queries,
SQL queries, hybrid queries, and intelligent routing.
"""

from typing import Optional, Union, Any, TYPE_CHECKING
import logging

from rag_framework.config.models import SourceType
from rag_framework.exceptions import RAGFrameworkError
from rag_framework.display.formatters import DisplayFormatter
from rag_framework.interfaces.chat import ChatInterface

if TYPE_CHECKING:
    from rag_framework.framework import RAGFramework

logger = logging.getLogger(__name__)


class QueryOperations:
    """
    Manages query-related operations.

    Responsibilities:
    - Document-based queries
    - SQL-based queries
    - Hybrid queries combining multiple sources
    - Source routing (UNSTRUCTURED/STRUCTURED/HYBRID)
    - Chat interface management
    - Debug information display
    """

    def __init__(self, framework: "RAGFramework") -> None:
        """
        Initialize query operations manager.

        Args:
            framework: Reference to parent RAGFramework instance.
        """
        self.framework = framework
        self._chat_interface: Optional[ChatInterface] = None

    def query(self, question: str) -> Union[str, Any]:
        """
        Query the RAG system with a question.

        Priority:
        1. Source routing (router.enabled) → SQL/Document/Hybrid routing
        2. Standard document-only RAG

        Args:
            question: The question to answer.

        Returns:
            Generated response as a string (or HybridQueryResponse if routing enabled).

        Raises:
            ValueError: If question is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Use source routing if enabled
        if self.framework.config.router.enabled:
            return self._query_with_routing(question)

        # Traditional document-only query
        return self._query_documents_only(question)

    def _query_with_routing(self, question: str) -> Any:
        """
        Execute query with intelligent routing.

        Args:
            question: The question to answer.

        Returns:
            HybridQueryResponse with routing info and results.
        """
        # Ensure hybrid engine is initialized
        self.framework._hybrid_ops.ensure_hybrid_engine()

        DisplayFormatter.print_question(question)

        try:
            response = self.framework._hybrid_engine.query(question)

            # Log routing decision
            logger.info(
                f"Query routed to {response.routing.source.value} "
                f"(confidence: {response.routing.confidence:.2f})"
            )

            if self.framework.config.debug:
                print(f"\n[Router] Source: {response.routing.source.value}")
                print(f"[Router] Confidence: {response.routing.confidence:.2f}")
                print(f"[Router] Method: {response.routing.method}")
                print(f"[Router] Reasoning: {response.routing.reasoning}")
                print(f"[Sources] {response.get_sources_summary()}")

            return response.response

        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            # Fall back to document-only query
            logger.info("Falling back to document-only query")
            return self._query_documents_only(question)

    def _query_documents_only(self, question: str) -> str:
        """
        Execute traditional document-only RAG query.

        Args:
            question: The question to answer.

        Returns:
            Generated response string.
        """
        self.framework._index_ops.ensure_index_loaded()
        self._ensure_query_engine()

        DisplayFormatter.print_question(question)

        if self.framework.config.debug:
            self._show_retrieved_chunks(question)

        try:
            response = self.framework._query_engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return f"Error processing query: {e}"

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
        return self._query_documents_only(question)

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
        if not self.framework.config.sql.enabled:
            raise RAGFrameworkError(
                "SQL is not enabled. Configure sql.enabled=true in config."
            )

        self.framework._hybrid_ops.ensure_sql_agent()

        DisplayFormatter.print_question(question)

        result = self.framework._sql_agent.query(question)

        if self.framework.config.debug:
            print(f"\n[SQL] Generated query: {result.query}")
            print(f"[SQL] Success: {result.success}")
            if result.result:
                print(f"[SQL] Rows returned: {result.result.row_count}")

        if result.success:
            return result.formatted_result
        else:
            return f"SQL query failed: {result.error}"

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
        self.framework._hybrid_ops.ensure_hybrid_engine()

        DisplayFormatter.print_question(question)

        # Temporarily override routing
        from rag_framework.routing import RoutingResult
        import datetime

        forced_routing = RoutingResult(
            source=SourceType.HYBRID,
            confidence=1.0,
            method="forced",
            reasoning="Explicitly requested hybrid query",
        )

        # Execute hybrid query
        response = self.framework._hybrid_engine._execute_hybrid_query(
            question,
            forced_routing,
            datetime.datetime.now(),
        )

        return response.response

    def _ensure_query_engine(self) -> None:
        """Create or recreate the query engine with current configuration."""
        self.framework._query_engine = (
            self.framework._query_manager.create_query_engine(
                index=self.framework._index,
                nodes=self.framework._nodes,
            )
        )

    def _show_retrieved_chunks(self, question: str) -> None:
        """
        Display the chunks retrieved for answering the question.

        This debug feature shows:
        - Relevance scores
        - Source file information
        - Text preview for each chunk

        Args:
            question: The question being processed.
        """
        try:
            retriever = self._get_retriever()
            nodes = retriever.retrieve(question)

            DisplayFormatter.print_retrieved_chunks_header()

            if not nodes:
                print("   Warning: No se encontraron chunks relevantes")
            else:
                DisplayFormatter.print_chunk_details(nodes)

            print(f"{DisplayFormatter.SEPARATOR_CHUNKS}\n")

        except Exception as e:
            logger.warning(f"Error displaying chunks: {e}")
            DisplayFormatter.print_error(f"Error showing chunks: {e}")

    def _get_retriever(self):
        """
        Extract retriever from the query engine.

        Returns:
            The retriever instance.
        """
        # Try different retriever access patterns
        if hasattr(self.framework._query_engine, "retriever"):
            return self.framework._query_engine.retriever
        elif hasattr(self.framework._query_engine, "_retriever"):
            return self.framework._query_engine._retriever
        else:
            # Fallback to creating a basic retriever
            return self.framework._index.as_retriever(
                similarity_top_k=self.framework.config.retrieval.top_k
            )

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
        """
        self.framework._index_ops.ensure_index_loaded()
        self._ensure_query_engine()

        # Create chat interface if not already created
        if self._chat_interface is None:
            self._chat_interface = ChatInterface(
                query_callback=self.query,
                config_callback=lambda: DisplayFormatter.print_config_summary(
                    self.framework.config
                ),
                templates_callback=lambda: DisplayFormatter.print_available_templates(
                    self.framework.config.prompt_template
                ),
            )

        # Programmatic mode: single message
        if message is not None:
            return self._chat_interface.process_single_message(message)

        # Interactive mode: command loop
        self._chat_interface.start_interactive_session()
        return None
