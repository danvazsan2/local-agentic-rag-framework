"""
Configuration Operations for RAG Framework.

Handles configuration management including prompt template manipulation,
configuration persistence, and model validation.
"""

from typing import TYPE_CHECKING
import logging

from rag_framework.config import ConfigLoader
from rag_framework.exceptions import RAGFrameworkError
from rag_framework.display.formatters import DisplayFormatter
from rag_framework.validators import ModelValidator

if TYPE_CHECKING:
    from rag_framework.framework import RAGFramework

logger = logging.getLogger(__name__)


class ConfigOperations:
    """
    Manages configuration-related operations.

    Responsibilities:
    - Prompt template management
    - Custom prompt configuration
    - Configuration persistence (save/load)
    - Model validation
    """

    def __init__(self, framework: "RAGFramework") -> None:
        """
        Initialize configuration operations manager.

        Args:
            framework: Reference to parent RAGFramework instance.
        """
        self.framework = framework

    def set_prompt_template(self, template_name: str) -> None:
        """
        Change the active prompt template.

        Args:
            template_name: Name of the template to use.
        """
        self.framework.config.prompt_template = template_name
        self.framework._query_engine = None  # Force recreation with new template

        DisplayFormatter.print_success(f"Prompt template changed to: {template_name}")
        logger.info(f"Prompt template changed to: {template_name}")

    def set_custom_prompt(self, prompt_template: str) -> None:
        """
        Set a custom prompt template.

        The template should include {context_str} and {query_str} placeholders.

        Args:
            prompt_template: Custom prompt string with required placeholders.
        """
        self.framework.config.prompt_template = "custom"
        self.framework.config.custom_prompt = prompt_template
        self.framework._query_engine = None  # Force recreation with new template

        DisplayFormatter.print_success("Custom prompt template set")
        logger.info("Custom prompt template configured")

    def save_config(self, path: str) -> None:
        """
        Save current configuration to a YAML file.

        Args:
            path: Path where the configuration should be saved.

        Raises:
            RAGFrameworkError: If saving fails.
        """
        try:
            ConfigLoader.save_yaml(self.framework.config, path)
            DisplayFormatter.print_success(f"Configuration saved to: {path}")
            logger.info(f"Configuration saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise RAGFrameworkError(f"Failed to save configuration: {e}") from e

    def validate_models(self) -> bool:
        """
        Validate that all configured models are available and accessible.

        This checks:
        - LLM provider and model availability
        - Embedding provider and model availability

        Returns:
            True if all models are valid and accessible, False otherwise.
        """
        validator = ModelValidator(self.framework.config)
        is_valid = validator.validate_all_models()

        if is_valid:
            logger.info("All models validated successfully")
        else:
            logger.warning("Model validation failed for one or more models")

        return is_valid

    def reload_config(self) -> None:
        """
        Reload configuration from default sources.

        Useful for picking up configuration changes without recreating
        the entire framework instance.

        Raises:
            RAGFrameworkError: If configuration reload fails.
        """
        try:
            new_config = ConfigLoader.load()
            new_config.validate()

            self.framework.config = new_config

            # Reset query engine to use new config
            self.framework._query_engine = None

            DisplayFormatter.print_success("Configuration reloaded")
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            raise RAGFrameworkError(f"Configuration reload failed: {e}") from e

    def get_config_summary(self) -> dict:
        """
        Get a summary of current configuration.

        Returns:
            Dictionary containing key configuration parameters.
        """
        return {
            "llm_provider": self.framework.config.llm.provider,
            "llm_model": self.framework.config.llm.model,
            "embedding_provider": self.framework.config.embedding.provider,
            "embedding_model": self.framework.config.embedding.model,
            "chunk_size": self.framework.config.chunking.chunk_size,
            "chunk_overlap": self.framework.config.chunking.chunk_overlap,
            "top_k": self.framework.config.retrieval.top_k,
            "reranking_enabled": self.framework.config.retrieval.reranker.enabled,
            "hybrid_search_enabled": self.framework.config.retrieval.use_hybrid_search,
            "routing_enabled": self.framework.config.router.enabled,
            "sql_enabled": self.framework.config.sql.enabled,
            "debug_mode": self.framework.config.debug,
        }
