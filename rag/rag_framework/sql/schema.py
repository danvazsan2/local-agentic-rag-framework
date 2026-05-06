"""
Database Schema Loader.

Extracts and manages database schema information for SQL generation.
Provides optimized schema representations for LLM consumption.
"""

from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
import logging
from sqlalchemy import create_engine, inspect, MetaData, text
from sqlalchemy.engine import Engine

from rag_framework.config.models import (
    SQLConfig,
    DatabaseConnectionConfig,
    SchemaConfig,
)
from rag_framework.exceptions import SQLConnectionError

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a database column."""

    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: Optional[str] = None  # "table.column" if FK
    description: Optional[str] = None
    sample_values: List[Any] = field(default_factory=list)

    def to_compact_string(self) -> str:
        """Generate compact string representation for LLM."""
        parts = [f"{self.name} ({self.data_type})"]

        if self.primary_key:
            parts.append("PK")
        if self.foreign_key:
            parts.append(f"FK→{self.foreign_key}")
        if not self.nullable:
            parts.append("NOT NULL")

        return " ".join(parts)


@dataclass
class TableInfo:
    """Information about a database table."""

    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    description: Optional[str] = None
    row_count: Optional[int] = None
    primary_key: Optional[List[str]] = None
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)

    def to_compact_string(self, include_samples: bool = False) -> str:
        """
        Generate compact string representation for LLM.

        Args:
            include_samples: Include sample values for columns

        Returns:
            Compact schema string
        """
        lines = [f"TABLE {self.name}"]

        if self.description:
            lines.append(f"  -- {self.description}")

        for col in self.columns:
            col_str = f"  {col.to_compact_string()}"
            if col.description:
                col_str += f" -- {col.description}"
            if include_samples and col.sample_values:
                samples = ", ".join(str(v) for v in col.sample_values[:3])
                col_str += f" [e.g., {samples}]"
            lines.append(col_str)

        if self.foreign_keys:
            fk_strs = [
                f"{fk['column']}→{fk['ref_table']}.{fk['ref_column']}"
                for fk in self.foreign_keys
            ]
            lines.append(f"  FOREIGN KEYS: {', '.join(fk_strs)}")

        if self.row_count is not None:
            lines.append(f"  (~{self.row_count} rows)")

        return "\n".join(lines)

    def get_column_names(self) -> List[str]:
        """Get list of column names."""
        return [col.name for col in self.columns]


class SchemaLoader:
    """
    Loads and manages database schema information.

    This class extracts schema metadata from the database and provides
    optimized representations for LLM consumption, minimizing token usage
    while preserving essential information.
    """

    def __init__(self, config: SQLConfig):
        """
        Initialize schema loader.

        Args:
            config: SQL configuration including connection details
        """
        self.config = config
        self.connection_config = config.connection
        self.schema_config = config.schema

        self._engine: Optional[Engine] = None
        self._tables: Dict[str, TableInfo] = {}
        self._loaded = False

    @property
    def engine(self) -> Engine:
        """Lazy-load database engine."""
        if self._engine is None:
            try:
                conn_string = self.connection_config.get_connection_string()
                self._engine = create_engine(
                    conn_string,
                    pool_size=self.connection_config.pool_size,
                    max_overflow=self.connection_config.max_overflow,
                    pool_timeout=self.connection_config.pool_timeout,
                )
                # Test connection
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info(
                    f"Database connection established: {self.connection_config.db_type}"
                )
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise SQLConnectionError(f"Database connection failed: {e}") from e
        return self._engine

    def load_schema(self, force_reload: bool = False) -> Dict[str, TableInfo]:
        """
        Load complete schema information from the database.

        Args:
            force_reload: Force reload even if already loaded

        Returns:
            Dictionary of table names to TableInfo objects
        """
        if self._loaded and not force_reload:
            return self._tables

        logger.info("Loading database schema...")

        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()

        # Apply include/exclude filters
        filtered_tables = self._filter_tables(table_names)

        for table_name in filtered_tables:
            try:
                table_info = self._load_table_info(inspector, table_name)
                self._tables[table_name] = table_info
            except Exception as e:
                logger.warning(f"Could not load schema for table {table_name}: {e}")

        self._loaded = True
        logger.info(f"Loaded schema for {len(self._tables)} tables")

        return self._tables

    def _filter_tables(self, table_names: List[str]) -> List[str]:
        """Apply include/exclude filters to table list."""
        # If include list is specified, only include those
        if self.schema_config.include_tables:
            table_names = [
                t for t in table_names if t in self.schema_config.include_tables
            ]

        # Apply exclusions
        if self.schema_config.exclude_tables:
            table_names = [
                t for t in table_names if t not in self.schema_config.exclude_tables
            ]

        return table_names

    def _load_table_info(self, inspector, table_name: str) -> TableInfo:
        """Load detailed information for a single table."""
        # Get columns
        columns = []
        column_info_list = inspector.get_columns(table_name)
        pk_columns = set(
            inspector.get_pk_constraint(table_name).get("constrained_columns", [])
        )

        # Get foreign keys
        fk_info = inspector.get_foreign_keys(table_name)
        fk_map = {}
        foreign_keys = []

        for fk in fk_info:
            for i, col in enumerate(fk.get("constrained_columns", [])):
                ref_table = fk.get("referred_table", "")
                ref_cols = fk.get("referred_columns", [])
                ref_col = ref_cols[i] if i < len(ref_cols) else ""
                fk_map[col] = f"{ref_table}.{ref_col}"
                foreign_keys.append(
                    {
                        "column": col,
                        "ref_table": ref_table,
                        "ref_column": ref_col,
                    }
                )

        for col_info in column_info_list:
            col_name = col_info["name"]

            column = ColumnInfo(
                name=col_name,
                data_type=str(col_info["type"]),
                nullable=col_info.get("nullable", True),
                primary_key=col_name in pk_columns,
                foreign_key=fk_map.get(col_name),
                description=self.schema_config.column_descriptions.get(
                    table_name, {}
                ).get(col_name),
            )

            # Load sample values if enabled
            if self.schema_config.include_sample_values:
                column.sample_values = self._get_sample_values(table_name, col_name)

            columns.append(column)

        # Get row count estimate
        row_count = self._get_row_count(table_name)

        return TableInfo(
            name=table_name,
            columns=columns,
            description=self.schema_config.table_descriptions.get(table_name),
            row_count=row_count,
            primary_key=list(pk_columns) if pk_columns else None,
            foreign_keys=foreign_keys,
        )

    def _get_sample_values(self, table_name: str, column_name: str) -> List[Any]:
        """Get sample values for a column."""
        try:
            limit = self.schema_config.sample_values_limit
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        f"SELECT DISTINCT {column_name} FROM {table_name} LIMIT {limit}"
                    )
                )
                return [row[0] for row in result if row[0] is not None]
        except Exception as e:
            logger.debug(f"Could not get samples for {table_name}.{column_name}: {e}")
            return []

    def _get_row_count(self, table_name: str) -> Optional[int]:
        """Get approximate row count for a table."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.scalar()
        except Exception as e:
            logger.debug(f"Could not get row count for {table_name}: {e}")
            return None

    def get_compact_schema(self, include_samples: bool = True) -> str:
        """
        Generate compact schema representation for LLM consumption.

        This method creates a token-efficient schema summary that includes
        essential information while minimizing prompt size.

        Args:
            include_samples: Include sample values

        Returns:
            Compact schema string
        """
        if not self._loaded:
            self.load_schema()

        parts = ["DATABASE SCHEMA:", "=" * 40]

        for table in self._tables.values():
            parts.append(table.to_compact_string(include_samples))
            parts.append("")

        return "\n".join(parts)

    def get_table_names(self) -> List[str]:
        """Get list of available table names."""
        if not self._loaded:
            self.load_schema()
        return list(self._tables.keys())

    def get_all_column_names(self) -> List[str]:
        """Get list of all column names across all tables."""
        if not self._loaded:
            self.load_schema()
        columns = []
        for table in self._tables.values():
            columns.extend(table.get_column_names())
        return columns

    def get_schema_for_router(self) -> Dict[str, Any]:
        """Get schema info formatted for the query router."""
        if not self._loaded:
            self.load_schema()

        return {
            "tables": list(self._tables.keys()),
            "columns": self.get_all_column_names(),
            "table_details": [
                {
                    "name": t.name,
                    "description": t.description,
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.data_type,
                            "primary_key": c.primary_key,
                            "foreign_key": c.foreign_key,
                        }
                        for c in t.columns
                    ],
                    "row_count": t.row_count,
                }
                for t in self._tables.values()
            ],
        }

    def close(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connection closed")
