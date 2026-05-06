"""
SQL and database configuration dataclasses.

Groups all SQL-related configs: connection, schema, security, and main SQLConfig.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict

from rag_framework.config.enums import DatabaseType


@dataclass
class DatabaseConnectionConfig:
    """Configuration for database connection."""

    # Connection type
    db_type: str = "sqlite"  # sqlite, postgresql, mysql

    # SQLite settings
    sqlite_path: Optional[str] = None  # Path to SQLite database file

    # Server-based DB settings (PostgreSQL, MySQL)
    host: str = "localhost"
    port: Optional[int] = None  # 5432 for PostgreSQL, 3306 for MySQL
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0

    # Query settings
    query_timeout: float = 30.0

    def __post_init__(self):
        """Validate configuration and set defaults."""
        valid_types = [t.value for t in DatabaseType]
        if self.db_type not in valid_types:
            raise ValueError(
                f"Invalid database type: {self.db_type}. " f"Supported: {valid_types}"
            )

        # Set default ports if not specified
        if self.port is None:
            if self.db_type == "postgresql":
                self.port = 5432
            elif self.db_type == "mysql":
                self.port = 3306

    def get_connection_string(self) -> str:
        """Build database connection string."""
        if self.db_type == "sqlite":
            if not self.sqlite_path:
                raise ValueError("sqlite_path is required for SQLite connections")
            return f"sqlite:///{self.sqlite_path}"

        elif self.db_type == "postgresql":
            return (
                f"postgresql://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )

        elif self.db_type == "mysql":
            return (
                f"mysql+pymysql://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )

        raise ValueError(f"Unknown database type: {self.db_type}")


@dataclass
class SchemaConfig:
    """Configuration for database schema handling."""

    # Tables to include (if empty, include all)
    include_tables: List[str] = field(default_factory=list)

    # Tables to exclude (blacklist)
    exclude_tables: List[str] = field(default_factory=list)

    # Include table descriptions in schema
    include_descriptions: bool = True

    # Include sample values for categorical columns
    include_sample_values: bool = True
    sample_values_limit: int = 5

    # Include foreign key relationships
    include_relationships: bool = True

    # Custom table descriptions (override auto-detected)
    table_descriptions: Dict[str, str] = field(default_factory=dict)

    # Custom column descriptions
    column_descriptions: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass
class SQLSecurityConfig:
    """Configuration for SQL security settings."""

    # Only allow SELECT statements
    allow_only_select: bool = True

    # Maximum rows to return
    max_rows: int = 100

    # Maximum query execution time (seconds)
    max_execution_time: float = 30.0

    # Forbidden SQL patterns (regex)
    forbidden_patterns: List[str] = field(
        default_factory=lambda: [
            r";\s*DROP\s+",
            r";\s*DELETE\s+",
            r";\s*UPDATE\s+",
            r";\s*INSERT\s+",
            r";\s*ALTER\s+",
            r";\s*CREATE\s+",
            r";\s*TRUNCATE\s+",
            r"--",  # SQL comments
            r"/\*",  # Block comments
            r"EXEC\s*\(",
            r"EXECUTE\s*\(",
            r"xp_",  # SQL Server extended procedures
            r"sp_",  # SQL Server stored procedures
        ]
    )

    # Allowed aggregate functions
    allowed_functions: List[str] = field(
        default_factory=lambda: [
            "COUNT",
            "SUM",
            "AVG",
            "MIN",
            "MAX",
            "ROUND",
            "COALESCE",
            "NULLIF",
            "CAST",
            "LOWER",
            "UPPER",
            "TRIM",
            "SUBSTRING",
            "DATE",
            "YEAR",
            "MONTH",
            "DAY",
            "LENGTH",
            "CONCAT",
            "REPLACE",
        ]
    )


@dataclass
class SQLConfig:
    """
    Main SQL configuration combining all SQL-related settings.
    """

    # Enable/disable SQL functionality
    enabled: bool = False

    # Database connection
    connection: DatabaseConnectionConfig = field(
        default_factory=DatabaseConnectionConfig
    )

    # Schema handling
    schema: SchemaConfig = field(default_factory=SchemaConfig)

    # Security settings
    security: SQLSecurityConfig = field(default_factory=SQLSecurityConfig)

    # LLM prompts for SQL generation
    max_retries: int = 3  # Max attempts to generate valid SQL

    # Example queries (for few-shot learning)
    example_queries: List[Dict[str, str]] = field(default_factory=list)
    # Format: [{"question": "...", "sql": "..."}, ...]
