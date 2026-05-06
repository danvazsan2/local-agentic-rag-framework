"""Tests for ConfigOperations.get_config_summary."""

from unittest.mock import MagicMock

import pytest

from rag_framework.operations.config import ConfigOperations


class TestGetConfigSummary:
    """ConfigOperations.get_config_summary returns a complete configuration snapshot."""

    @pytest.fixture
    def config_ops(self, sample_rag_config):
        framework = MagicMock()
        framework.config = sample_rag_config
        ops = ConfigOperations.__new__(ConfigOperations)
        ops.framework = framework
        return ops

    def test_summary_is_dict(self, config_ops):
        assert isinstance(config_ops.get_config_summary(), dict)

    def test_summary_returns_all_expected_keys(self, config_ops):
        result = config_ops.get_config_summary()
        expected_keys = {
            "llm_provider", "llm_model", "embedding_provider", "embedding_model",
            "chunk_size", "chunk_overlap", "top_k", "reranking_enabled",
            "hybrid_search_enabled", "routing_enabled", "sql_enabled", "debug_mode",
        }
        assert set(result.keys()) == expected_keys

    def test_summary_values_match_config(self, config_ops):
        result = config_ops.get_config_summary()
        assert result["llm_model"] == "llama3-instruct-8k"
        assert result["embedding_provider"] == "ollama"
        assert result["chunk_size"] == 512
        assert result["chunk_overlap"] == 50
        assert result["reranking_enabled"] is False
        assert result["hybrid_search_enabled"] is True
        assert result["top_k"] == 5
