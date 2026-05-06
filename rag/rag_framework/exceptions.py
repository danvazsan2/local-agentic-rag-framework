"""
RAG Framework - Custom Exceptions.

Defines custom exception hierarchy for comprehensive error handling
throughout the RAG framework.
"""


class RAGFrameworkError(Exception):
    """
    Base exception for all RAG Framework errors.

    All framework-specific exceptions should inherit from this class
    to allow for easy exception filtering and handling.
    """

    pass


class IndexNotFoundError(RAGFrameworkError):
    """
    Raised when a vector index is not available or cannot be loaded.

    This typically occurs when:
    - Attempting to query before ingesting documents
    - Index files are missing or corrupted
    - Index loading fails due to configuration mismatch
    """

    pass


class DocumentIngestionError(RAGFrameworkError):
    """
    Raised when document ingestion fails.

    This can occur due to:
    - Invalid document formats
    - File system access errors
    - Document parsing failures
    - Encoding issues
    """

    pass


class ConfigurationError(RAGFrameworkError):
    """
    Raised when configuration validation fails.

    This occurs when:
    - Required configuration fields are missing
    - Configuration values are invalid
    - Configuration file cannot be parsed
    """

    pass


class ModelValidationError(RAGFrameworkError):
    """
    Raised when model validation fails.

    This occurs when:
    - LLM provider is not accessible
    - Embedding model is not available
    - API credentials are invalid
    - Model configuration is incorrect
    """

    pass


# ============================================================================
# SQL-related Exceptions
# ============================================================================


class SQLError(RAGFrameworkError):
    """
    Base exception for SQL-related errors.

    All SQL-specific exceptions should inherit from this class.
    """

    pass


class SQLConnectionError(SQLError):
    """
    Raised when database connection fails.

    This occurs when:
    - Database server is unreachable
    - Connection credentials are invalid
    - Database does not exist
    - Connection pool is exhausted
    """

    pass


class SQLQueryError(SQLError):
    """
    Raised when SQL query execution fails.

    This occurs when:
    - Syntax error in generated SQL
    - Referenced table/column does not exist
    - Query timeout exceeded
    - Constraint violations
    """

    pass


class SQLSecurityError(SQLError):
    """
    Raised when a potentially unsafe SQL operation is detected.

    This occurs when:
    - Non-SELECT statement is attempted
    - SQL injection patterns detected
    - Forbidden tables/columns accessed
    - Query exceeds safety limits
    """

    pass


class SQLGenerationError(SQLError):
    """
    Raised when LLM fails to generate valid SQL.

    This occurs when:
    - LLM produces malformed SQL
    - Generated SQL doesn't match schema
    - Query validation fails
    - Maximum retry attempts exceeded
    """

    pass


class RoutingError(RAGFrameworkError):
    """
    Raised when query routing fails.

    This occurs when:
    - Router cannot classify query
    - Invalid source type specified
    - Required source is unavailable
    """

    pass
