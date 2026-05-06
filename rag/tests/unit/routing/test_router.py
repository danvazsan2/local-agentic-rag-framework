"""Unit tests for QueryRouter (LLM-based query routing).

Validates that the router correctly delegates classification to the LLM
and handles edge cases: disabled router, unavailable SQL, and LLM failure.
"""

import pytest
from unittest.mock import MagicMock

from rag_framework.routing.router import QueryRouter, RoutingResult
from rag_framework.config.models import SourceType
from rag_framework.config.retrieval_config import RouterConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(router_enabled=True, sql_enabled=True, default_source="unstructured"):
    config = MagicMock()
    config.router = RouterConfig(enabled=router_enabled, default_source=default_source)
    config.sql = MagicMock()
    config.sql.enabled = sql_enabled
    config.llm = MagicMock()
    return config


@pytest.fixture
def sample_schema():
    return {
        "tables": ["users", "products", "orders"],
        "columns": ["id", "name", "email", "price", "total"],
        "table_details": [
            {
                "name": "users",
                "description": "Registered users",
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True, "foreign_key": None},
                    {"name": "name", "type": "VARCHAR", "primary_key": False, "foreign_key": None},
                ],
                "row_count": 500,
            },
            {
                "name": "products",
                "description": "Product catalog",
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True, "foreign_key": None},
                    {"name": "price", "type": "FLOAT", "primary_key": False, "foreign_key": None},
                    {"name": "category_id", "type": "INTEGER", "primary_key": False, "foreign_key": "categories.id"},
                ],
                "row_count": 200,
            },
        ],
    }


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete.return_value = "STRUCTURED"
    return llm


@pytest.fixture
def router(sample_schema, mock_llm):
    return QueryRouter(config=_make_config(), llm=mock_llm, schema_info=sample_schema)


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------


class TestLLMClassification:
    """Router delegates classification to the LLM and parses the response."""

    def test_routes_structured_query(self, router, mock_llm):
        mock_llm.complete.return_value = "STRUCTURED"
        result = router.route("¿Cuántos usuarios activos hay?")
        assert result.source == SourceType.STRUCTURED
        assert result.method == "llm"

    def test_routes_unstructured_query(self, router, mock_llm):
        mock_llm.complete.return_value = "UNSTRUCTURED"
        result = router.route("¿Cuál es la política de plagio?")
        assert result.source == SourceType.UNSTRUCTURED
        assert result.method == "llm"

    def test_routes_hybrid_query(self, router, mock_llm):
        mock_llm.complete.return_value = "HYBRID"
        result = router.route("Lista los pedidos y explica la política de devoluciones")
        assert result.source == SourceType.HYBRID
        assert result.method == "llm"

    def test_prompt_contains_schema_and_query(self, router, mock_llm):
        query = "¿Cuántos productos hay?"
        router.route(query)
        prompt = mock_llm.complete.call_args[0][0]
        assert "users" in prompt
        assert "products" in prompt
        assert query in prompt

    def test_handles_whitespace_and_mixed_case_response(self, router, mock_llm):
        mock_llm.complete.return_value = "  Structured  \n"
        result = router.route("¿Cuántos usuarios hay?")
        assert result.source == SourceType.STRUCTURED


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


class TestFallbackBehavior:
    """When the LLM is unavailable or returns nonsense, the router falls back."""

    def test_llm_failure_falls_back_to_default(self, sample_schema):
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("LLM unavailable")
        router = QueryRouter(config=_make_config(), llm=mock_llm, schema_info=sample_schema)

        result = router.route("¿Cuántos usuarios hay?")
        assert result.source == SourceType.UNSTRUCTURED
        assert result.method == "default"

    def test_unparseable_response_falls_back(self, router, mock_llm):
        mock_llm.complete.return_value = "I don't understand"
        result = router.route("¿Cuántos usuarios hay?")
        assert result.method == "default"


# ---------------------------------------------------------------------------
# Router state
# ---------------------------------------------------------------------------


class TestRouterState:
    def test_disabled_router_skips_llm(self, sample_schema):
        mock_llm = MagicMock()
        router = QueryRouter(config=_make_config(router_enabled=False), llm=mock_llm, schema_info=sample_schema)
        result = router.route("¿Cuántos usuarios hay?")
        assert result.method == "default"
        mock_llm.complete.assert_not_called()

    def test_sql_disabled_returns_unstructured_default(self, sample_schema):
        mock_llm = MagicMock()
        router = QueryRouter(config=_make_config(sql_enabled=False), llm=mock_llm, schema_info=sample_schema)
        result = router.route("¿Cuántos usuarios hay?")
        assert result.source == SourceType.UNSTRUCTURED
        assert result.method == "default"
        mock_llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Schema summary
# ---------------------------------------------------------------------------


class TestSchemaSummary:
    def test_summary_contains_table_and_column_info(self, router):
        summary = router._build_schema_summary()
        assert "users" in summary
        assert "products" in summary
        assert "INTEGER" in summary
        assert "FLOAT" in summary
        assert "categories.id" in summary  # FK reference

    def test_empty_schema_produces_fallback_message(self):
        router = QueryRouter(config=_make_config(), llm=MagicMock(), schema_info={})
        summary = router._build_schema_summary()
        assert "No hay esquema" in summary


# ---------------------------------------------------------------------------
# RoutingResult
# ---------------------------------------------------------------------------


class TestRoutingResult:
    def test_serialization_to_dict(self):
        result = RoutingResult(
            source=SourceType.STRUCTURED,
            confidence=0.9,
            method="llm",
            reasoning="LLM classified as structured",
        )
        d = result.to_dict()
        assert d["source"] == "structured"
        assert d["confidence"] == 0.9
        assert d["method"] == "llm"
