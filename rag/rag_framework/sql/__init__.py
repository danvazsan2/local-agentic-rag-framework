"""
SQL Module for RAG Framework.

Provides SQL query generation, execution, and database interaction
capabilities for the hybrid RAG system.

Components:
- SQLAgent: Main interface for natural language to SQL translation
- SchemaLoader: Extracts and manages database schema information
- QueryValidator: Validates SQL for safety before execution
- QueryExecutor: Safely executes validated queries
"""

from rag_framework.sql.agent import SQLAgent, SQLQueryResult
from rag_framework.sql.schema import SchemaLoader, TableInfo, ColumnInfo
from rag_framework.sql.validator import QueryValidator, ValidationResult
from rag_framework.sql.executor import QueryExecutor

__all__ = [
    "SQLAgent",
    "SQLQueryResult",
    "SchemaLoader",
    "TableInfo",
    "ColumnInfo",
    "QueryValidator",
    "ValidationResult",
    "QueryExecutor",
]
