"""
Index Operations for RAG Framework.

Handles all operations related to document ingestion, index creation,
loading, and management.
"""

from typing import Optional, TYPE_CHECKING
import logging

from rag_framework.exceptions import (
    IndexNotFoundError,
    DocumentIngestionError,
    RAGFrameworkError,
)
from rag_framework.display.formatters import DisplayFormatter

if TYPE_CHECKING:
    from rag_framework.framework import RAGFramework

logger = logging.getLogger(__name__)


class IndexOperations:
    """
    Manages index-related operations.

    Responsibilities:
    - Document ingestion and processing
    - Index creation and persistence
    - Index loading and validation
    - Index clearing and cleanup
    """

    def __init__(self, framework: "RAGFramework") -> None:
        """
        Initialize index operations manager.

        Args:
            framework: Reference to parent RAGFramework instance.
        """
        self.framework = framework

    def ingest(
        self, documents_dir: Optional[str] = None, force_reindex: bool = False
    ) -> tuple:
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
            Tuple of (index, nodes) after ingestion.

        Raises:
            DocumentIngestionError: If ingestion fails critically.
        """
        # Attempt to load existing index if not forcing reindex
        if not force_reindex:
            loaded = self._try_load_existing_index()
            if loaded:
                return loaded

        # Ingest documents
        try:
            nodes = self.framework._ingestion.ingest(documents_dir)
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise DocumentIngestionError(f"Failed to ingest documents: {e}") from e

        if not nodes:
            logger.warning("No documents were ingested")
            print("No documents to index")
            return None, None

        # Create index from ingested nodes
        index = self._create_index_from_nodes(nodes)

        return index, nodes

    def _try_load_existing_index(self) -> Optional[tuple]:
        """
        Attempt to load an existing index from disk.

        Returns:
            Tuple of (index, nodes) if successfully loaded, None otherwise.
        """
        if not self.framework._index_manager.index_exists():
            return None

        print("Found existing index. Loading...")

        try:
            index, nodes = self.framework._index_manager.load_index()

            if index:
                DisplayFormatter.print_index_load_status(
                    has_nodes=bool(nodes),
                    node_count=len(nodes) if nodes else None,
                )
                return (index, nodes)
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            print(f"Could not load existing index: {e}")
            print("   Will re-ingest documents...")

        return None

    def _create_index_from_nodes(self, nodes: list):
        """
        Create vector index from ingested document nodes.

        Args:
            nodes: List of document nodes to index.

        Returns:
            Created index instance.

        Raises:
            RAGFrameworkError: If index creation fails.
        """
        try:
            index = self.framework._index_manager.create_index(nodes)
            logger.info(f"Successfully created index with {len(nodes)} nodes")
            return index
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            raise RAGFrameworkError(f"Failed to create index: {e}") from e

    def load_index(self) -> tuple:
        """
        Load an existing index without re-ingesting documents.

        This is useful when you want to query an existing index without
        reprocessing the source documents.

        Returns:
            Tuple of (index, nodes) if loaded successfully.

        Raises:
            IndexNotFoundError: If no index exists and loading fails.
        """
        index = None
        try:
            index, nodes = self.framework._index_manager.load_index()
            DisplayFormatter.print_index_load_status(
                has_nodes=bool(nodes),
                node_count=len(nodes) if nodes else None,
            )

            if not nodes:
                print("   Tip: Run 'ingest' to enable hybrid search")

            return index, nodes

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            print(f"Error loading index: {e}")

            if index is None:
                error_msg = "No index found. Run ingest() first."
                logger.error(error_msg)
                print(f"Error: {error_msg}")
                raise IndexNotFoundError(error_msg)

            return None, None

    def clear_index(self) -> None:
        """
        Clear the vector index and reset framework state.

        This removes all indexed data and requires re-ingestion.
        """
        try:
            self.framework._index_manager.clear_index()
            logger.info("Index cleared successfully")
            print("Success: Index cleared")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            print(f"Warning: Error clearing index: {e}")

    def ensure_index_loaded(self) -> None:
        """
        Ensure an index is loaded, attempting to load if necessary.

        Raises:
            IndexNotFoundError: If index cannot be loaded.
        """
        if self.framework._index is None:
            try:
                index, nodes = self.load_index()
                self.framework._index = index
                self.framework._nodes = nodes
            except IndexNotFoundError:
                raise IndexNotFoundError(
                    "No index available. Run ingest() first to create an index."
                )
