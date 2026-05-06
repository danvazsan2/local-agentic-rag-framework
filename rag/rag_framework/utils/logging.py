"""
Centralized logging configuration for the RAG Framework.

Provides consistent logging setup across all entry points
and suppresses verbose library logs.
"""

import logging
import warnings
import os
from typing import Optional, List

from rag_framework.utils.constants import VERBOSE_LOGGERS


def setup_logging(
    level: int = logging.WARNING,
    format_string: Optional[str] = None,
    suppress_warnings: bool = True,
    debug: bool = False,
) -> logging.Logger:
    """
    Configure logging for the RAG Framework.

    This function should be called once at the start of the application
    to ensure consistent logging behavior.

    Args:
        level: Base logging level (default: WARNING)
        format_string: Custom format string for log messages
        suppress_warnings: Whether to suppress tokenizer and other warnings
        debug: If True, sets level to DEBUG and shows more output

    Returns:
        Root logger instance

    Example:
        >>> from rag_framework.utils import setup_logging
        >>> setup_logging(debug=True)
    """
    # Set effective level
    effective_level = logging.DEBUG if debug else level

    # Default format
    if format_string is None:
        if debug:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(asctime)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=effective_level,
        format=format_string,
        force=True,  # Override existing configuration
    )

    # Suppress verbose library logs
    for logger_name in VERBOSE_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Ensure rag_framework loggers always show INFO (even when root is WARNING)
    framework_logger = logging.getLogger("rag_framework")
    if (
        framework_logger.level == logging.NOTSET
        or framework_logger.level > logging.INFO
    ):
        framework_logger.setLevel(logging.INFO)

    # Suppress transformers' own logging system (separate from stdlib)
    try:
        import transformers as _tf

        _tf.logging.set_verbosity_error()
    except Exception:
        pass

    # Suppress warnings if requested
    if suppress_warnings:
        _suppress_warnings()

    # Set tokenizers parallelism to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return logging.getLogger()


def _suppress_warnings():
    """Suppress common library warnings."""
    # Tokenizer warnings
    warnings.filterwarnings("ignore", message=".*tokenizer.*")
    warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
    warnings.filterwarnings("ignore", message=".*regex pattern.*")

    # HuggingFace warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="bm25s")
    warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")
    warnings.filterwarnings("ignore", message=".*fast tokenizer.*")

    # Torch warnings
    warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a named logger for a specific module.

    Args:
        name: Logger name (typically __name__)
        level: Optional specific level for this logger

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing documents...")
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger


class LoggerMixin:
    """
    Mixin class that provides a logger property.

    Usage:
        class MyClass(LoggerMixin):
            def method(self):
                self.logger.info("Doing something")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
