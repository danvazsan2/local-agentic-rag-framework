"""
SQL Query Executor.

Safely executes validated SQL queries against the database
with proper timeout handling, connection management, and
result formatting.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from rag_framework.config.models import SQLConfig
from rag_framework.exceptions import SQLQueryError, SQLSecurityError
from rag_framework.sql.validator import QueryValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a SQL query execution."""

    success: bool
    columns: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    query: str = ""  # The executed query (for debugging)

    def to_markdown_table(self, max_rows: int = 20) -> str:
        """
        Convert results to a Markdown table format.

        Args:
            max_rows: Maximum rows to include

        Returns:
            Markdown-formatted table string
        """
        if not self.success:
            return f"**Error:** {self.error}"

        if not self.rows:
            return "*No results found*"

        lines = []

        # Header
        lines.append("| " + " | ".join(self.columns) + " |")
        lines.append("|" + "|".join(["---"] * len(self.columns)) + "|")

        # Rows
        for row in self.rows[:max_rows]:
            values = [str(row.get(col, "")) for col in self.columns]
            # Escape pipe characters
            values = [v.replace("|", "\\|") for v in values]
            lines.append("| " + " | ".join(values) + " |")

        # Show truncation notice
        if len(self.rows) > max_rows:
            lines.append(f"\n*Showing {max_rows} of {len(self.rows)} rows*")

        return "\n".join(lines)

    def to_natural_language(self) -> str:
        """
        Convert results to natural language description.

        This format is useful for LLM synthesis.
        """
        if not self.success:
            return f"Query failed: {self.error}"

        if not self.rows:
            return "The query returned no results."

        if self.row_count == 1 and len(self.columns) == 1:
            # Single value result (e.g., COUNT)
            value = self.rows[0].get(self.columns[0])
            return f"Result: {value}"

        if self.row_count == 1:
            # Single row - describe as attributes
            row = self.rows[0]
            parts = [f"{col}: {row.get(col)}" for col in self.columns]
            return "Found: " + ", ".join(parts)

        # Multiple rows - list all
        summary = (
            f"Found {self.row_count} results with columns: {', '.join(self.columns)}.\n"
        )

        for i, row in enumerate(self.rows, 1):
            parts = [f"{col}={row.get(col)}" for col in self.columns]
            summary += f"{i}. {', '.join(parts)}\n"

        return summary


class QueryExecutor:
    """
    Safely executes SQL queries with validation and timeout handling.

    This executor provides:
    - Pre-execution validation
    - Connection pooling via SQLAlchemy
    - Timeout enforcement
    - Result formatting for LLM consumption
    """

    def __init__(
        self,
        engine: Engine,
        config: SQLConfig,
        validator: Optional[QueryValidator] = None,
    ):
        """
        Initialize query executor.

        Args:
            engine: SQLAlchemy engine for database connection
            config: SQL configuration
            validator: Query validator (created if not provided)
        """
        self.engine = engine
        self.config = config
        self._validator = validator

    @property
    def validator(self) -> QueryValidator:
        """Lazy-load validator."""
        if self._validator is None:
            self._validator = QueryValidator(self.config.security)
        return self._validator

    def execute(
        self,
        query: str,
        skip_validation: bool = False,
    ) -> QueryResult:
        """
        Execute a SQL query safely.

        Args:
            query: SQL query to execute
            skip_validation: Skip validation (use with caution)

        Returns:
            QueryResult with execution results
        """
        start_time = datetime.now()
        warnings = []

        # Validate query unless skipped
        if not skip_validation:
            validation = self.validator.validate(query)

            if not validation.is_valid:
                logger.warning(f"Query validation failed: {validation.errors}")
                raise SQLSecurityError(
                    f"Query validation failed: {'; '.join(validation.errors)}"
                )

            query = validation.query  # Use potentially sanitized query
            warnings.extend(validation.warnings)

        try:
            with self.engine.connect() as conn:
                # Execute with timeout
                result = conn.execute(text(query))

                # Fetch results
                if result.returns_rows:
                    columns = list(result.keys())
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]
                else:
                    columns = []
                    rows = []

                execution_time = (datetime.now() - start_time).total_seconds() * 1000

                logger.info(
                    f"Query executed successfully: {len(rows)} rows in {execution_time:.1f}ms"
                )

                return QueryResult(
                    success=True,
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_time_ms=execution_time,
                    warnings=warnings,
                    query=query,
                )

        except SQLAlchemyError as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = str(e)

            logger.error(f"Query execution failed: {error_msg}")

            return QueryResult(
                success=False,
                error=self._sanitize_error(error_msg),
                execution_time_ms=execution_time,
                query=query,
            )

    def execute_with_retry(
        self,
        query: str,
        max_retries: int = 2,
    ) -> QueryResult:
        """
        Execute query with retry on transient failures.

        Args:
            query: SQL query to execute
            max_retries: Maximum retry attempts

        Returns:
            QueryResult from successful execution or final failure
        """
        last_result = None

        for attempt in range(max_retries + 1):
            result = self.execute(query)

            if result.success:
                return result

            last_result = result

            # Check if error is retryable
            if not self._is_retryable_error(result.error):
                break

            logger.info(f"Retrying query (attempt {attempt + 2}/{max_retries + 1})")

        return last_result or QueryResult(
            success=False,
            error="Query execution failed after retries",
        )

    def _is_retryable_error(self, error: Optional[str]) -> bool:
        """Check if an error is transient and worth retrying."""
        if not error:
            return False

        error_lower = error.lower()
        retryable_patterns = [
            "connection",
            "timeout",
            "deadlock",
            "lock wait",
            "too many connections",
        ]

        return any(pattern in error_lower for pattern in retryable_patterns)

    def _sanitize_error(self, error: str) -> str:
        """
        Sanitize error message to avoid leaking sensitive information.
        """
        # Remove potential file paths
        error = error.split("\n")[0]  # Only first line

        # Truncate long errors
        if len(error) > 200:
            error = error[:200] + "..."

        return error

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
