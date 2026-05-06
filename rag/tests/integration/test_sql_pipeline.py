"""Integration tests for SQL RAG pipeline.

Tests the hybrid system that combines SQL queries with document retrieval.
Migrated from prueba.py to pytest format.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSQLRAGPipeline:
    """Integration tests for SQL + RAG hybrid queries."""

    @pytest.fixture
    def sample_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "test.db"

    @pytest.fixture
    def sample_config_data(self, tmp_path, sample_db_path):
        """Sample configuration for SQL-enabled RAG."""
        return {
            "llm": {
                "provider": "ollama",
                "model": "llama3-instruct-8k",
                "base_url": "http://localhost:11434",
            },
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text:v1.5",
            },
            "sql": {
                "enabled": True,
                "database_path": str(sample_db_path),
            },
            "router": {
                "enabled": True,
            },
        }

    def test_sql_query_classification(self):
        """Test that SQL queries are correctly classified by the LLM router."""
        from unittest.mock import MagicMock
        from rag_framework.routing.router import QueryRouter, RoutingResult
        from rag_framework.config.models import SourceType
        from rag_framework.config.retrieval_config import RouterConfig

        config = MagicMock()
        config.router = RouterConfig(enabled=True, default_source="unstructured")
        config.sql.enabled = True

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "STRUCTURED"

        router = QueryRouter(
            config=config,
            llm=mock_llm,
            schema_info={
                "tables": ["products", "users"],
                "table_details": [
                    {
                        "name": "products",
                        "description": "Product catalog",
                        "columns": [
                            {
                                "name": "id",
                                "type": "INTEGER",
                                "primary_key": True,
                                "foreign_key": None,
                            },
                            {
                                "name": "name",
                                "type": "VARCHAR",
                                "primary_key": False,
                                "foreign_key": None,
                            },
                            {
                                "name": "price",
                                "type": "FLOAT",
                                "primary_key": False,
                                "foreign_key": None,
                            },
                        ],
                        "row_count": 100,
                    },
                ],
            },
        )

        sql_queries = [
            "¿Cuántas asignaturas hay en total?",
            "¿Qué profesores imparten asignaturas de Inteligencia Artificial?",
            "Lista las asignaturas del cuarto curso",
        ]

        for query in sql_queries:
            result = router.route(query)
            assert result.source == SourceType.STRUCTURED
            assert result.method == "llm"

    def test_document_query_classification(self):
        """Test that document queries are routed to unstructured by the LLM router."""
        from unittest.mock import MagicMock
        from rag_framework.routing.router import QueryRouter
        from rag_framework.config.models import SourceType
        from rag_framework.config.retrieval_config import RouterConfig

        config = MagicMock()
        config.router = RouterConfig(enabled=True, default_source="unstructured")
        config.sql.enabled = True

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "UNSTRUCTURED"

        router = QueryRouter(
            config=config,
            llm=mock_llm,
            schema_info={
                "tables": ["products"],
                "table_details": [],
            },
        )

        doc_queries = [
            "¿Cuál es la política de plagio?",
            "¿Cuáles son los requisitos para el TFG?",
            "Explica el procedimiento de matrícula",
        ]

        for query in doc_queries:
            result = router.route(query)
            assert result.source == SourceType.UNSTRUCTURED
            assert result.method == "llm"

    @pytest.mark.skip(reason="Requires running Ollama server")
    def test_hybrid_query_execution(self, sample_config_data):
        """Test full hybrid query execution."""
        from rag_framework import RAGFramework

        # This test requires:
        # 1. Ollama running
        # 2. Documents indexed
        # 3. Database with data
        # Skip for CI, run manually with --run-integration
        pass

    def test_query_types_classification(self):
        """Test that various query types are routed correctly by the LLM router."""
        from unittest.mock import MagicMock
        from rag_framework.routing.router import QueryRouter
        from rag_framework.config.models import SourceType
        from rag_framework.config.retrieval_config import RouterConfig

        config = MagicMock()
        config.router = RouterConfig(enabled=True, default_source="unstructured")
        config.sql.enabled = True

        # The LLM responses for each query in order
        llm_responses = [
            "STRUCTURED",
            "STRUCTURED",
            "STRUCTURED",
            "UNSTRUCTURED",
            "HYBRID",
        ]
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = llm_responses

        router = QueryRouter(
            config=config,
            llm=mock_llm,
            schema_info={
                "tables": ["products", "users"],
                "table_details": [],
            },
        )

        test_queries = [
            "¿Cuántas asignaturas hay en total?",
            "¿Qué profesores imparten asignaturas de Inteligencia Artificial?",
            "Lista las asignaturas del cuarto curso",
            "¿Cuál es la política de plagio?",
            "¿Cuántos créditos necesito para matricularme del TFG?",
        ]

        expected = [
            SourceType.STRUCTURED,
            SourceType.STRUCTURED,
            SourceType.STRUCTURED,
            SourceType.UNSTRUCTURED,
            SourceType.HYBRID,
        ]

        for query, exp_source in zip(test_queries, expected):
            result = router.route(query)
            assert (
                result.source == exp_source
            ), f"Expected {exp_source.value} for: {query}"
            assert result.method == "llm"
