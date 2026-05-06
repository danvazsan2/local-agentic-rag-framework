"""
Lifecycle Management for RAG Framework.

Handles initialization, setup, and factory methods for creating RAGFramework
instances from various configuration sources.
"""

from typing import Optional, TYPE_CHECKING
import logging

from rag_framework.config import RAGConfig, ConfigLoader
from rag_framework.core.ingestion import DocumentIngestion
from rag_framework.core.indexing import IndexManager
from rag_framework.core.query_engine import QueryEngineManager
from rag_framework.exceptions import RAGFrameworkError
from rag_framework.display.formatters import DisplayFormatter

if TYPE_CHECKING:
    from rag_framework.framework import RAGFramework

logger = logging.getLogger(__name__)


class LifecycleManager:
    """
    Manages the lifecycle of RAG Framework components.

    Responsibilities:
    - Configuration initialization and validation
    - Core component initialization (ingestion, index, query managers)
    - State management (reset, initialization tracking)
    - Factory methods for creating framework instances
    """

    def __init__(self, framework: "RAGFramework") -> None:
        """
        Initialize lifecycle manager.

        Args:
            framework: Reference to parent RAGFramework instance.
        """
        self.framework = framework
        self._initialized = False

    def initialize_from_config(self, config: Optional[RAGConfig] = None) -> RAGConfig:
        """
        Initialize framework configuration.

        Args:
            config: RAG configuration. If None, loads from default sources.

        Returns:
            Validated RAGConfig instance.

        Raises:
            RAGFrameworkError: If configuration validation fails.
        """
        try:
            config = config or ConfigLoader.load()
            config.validate()
            return config
        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            raise RAGFrameworkError(f"Configuration initialization failed: {e}") from e

    def initialize_managers(self, config: RAGConfig) -> tuple:
        """
        Initialize the core framework managers.

        Args:
            config: Validated RAG configuration.

        Returns:
            Tuple of (ingestion, index_manager, query_manager).

        Raises:
            RAGFrameworkError: If manager initialization fails.
        """
        try:
            ingestion = DocumentIngestion(config)
            index_manager = IndexManager(config)
            query_manager = QueryEngineManager(config)

            self._initialized = True
            logger.info("Core managers initialized successfully")

            return ingestion, index_manager, query_manager
        except Exception as e:
            logger.error(f"Failed to initialize managers: {e}")
            raise RAGFrameworkError(f"Manager initialization failed: {e}") from e

    def reset_state(self) -> tuple:
        """
        Reset internal state variables.

        Returns:
            Tuple of (None, None, None) for (index, nodes, query_engine).
        """
        logger.debug("Resetting framework state")
        return None, None, None

    def display_initialization_summary(self, config: RAGConfig) -> None:
        """
        Display configuration summary and hybrid mode status.

        Args:
            config: Current RAG configuration.
        """
        DisplayFormatter.print_config_summary(config)

        if config.router.enabled:
            print(f"   Routing: ENABLED (SQL: {'ON' if config.sql.enabled else 'OFF'})")

    @staticmethod
    def create_from_yaml(config_path: str) -> RAGConfig:
        """
        Create configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Loaded RAGConfig instance.

        Raises:
            RAGFrameworkError: If config file cannot be loaded.
        """
        try:
            config = ConfigLoader.load_from_yaml(config_path)
            logger.info(f"Configuration loaded from YAML: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from YAML: {e}")
            raise RAGFrameworkError(f"YAML config loading failed: {e}") from e

    @property
    def is_initialized(self) -> bool:
        """Check if managers have been initialized."""
        return self._initialized
