"""Unit tests for configuration dataclasses.

Covers LLMConfig provider validation, RAGConfig deserialization,
DirectoryConfig filesystem side-effects, and CorrectiveRAG config integration.
"""

import os
import pytest

from rag_framework.config.models import (
    RAGConfig,
    LLMConfig,
    DirectoryConfig,
)
from rag_framework.config.corrective_rag_config import CorrectiveRAGConfig


class TestLLMConfig:
    """LLMConfig provider validation and default values."""

    def test_defaults(self):
        config = LLMConfig()
        assert config.provider == "ollama"
        assert config.model == "llama3-instruct-8k"
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == 0.0
        assert config.max_tokens == 512

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Invalid LLM provider"):
            LLMConfig(provider="nonexistent")

    def test_huggingface_sets_hf_model_id_from_model(self):
        config = LLMConfig(provider="huggingface", model="my-model")
        assert config.hf_model_id == "my-model"

    def test_huggingface_local_path_sets_is_local(self):
        config = LLMConfig(provider="huggingface", local_model_path="/some/path")
        assert config.is_local is True


class TestRAGConfigFromDict:
    """RAGConfig.from_dict should build a fully populated config."""

    def test_from_empty_dict_uses_defaults(self, tmp_path):
        config = RAGConfig.from_dict({
            "directories": {
                "documents_dir": str(tmp_path / "docs"),
                "vector_store_dir": str(tmp_path / "vs"),
            }
        })
        assert isinstance(config, RAGConfig)
        assert config.llm.provider == "ollama"
        assert config.prompt_template == "default"
        assert config.debug is False

    def test_from_dict_overrides_llm_fields(self, tmp_path):
        config = RAGConfig.from_dict({
            "llm": {"provider": "ollama", "model": "custom-model", "temperature": 0.7},
            "directories": {
                "documents_dir": str(tmp_path / "docs"),
                "vector_store_dir": str(tmp_path / "vs"),
            },
        })
        assert config.llm.model == "custom-model"
        assert config.llm.temperature == 0.7

    def test_from_dict_parses_nested_sql(self, tmp_path):
        config = RAGConfig.from_dict({
            "sql": {
                "enabled": True,
                "connection": {"db_type": "sqlite", "sqlite_path": "/tmp/test.db"},
                "security": {"allow_only_select": True, "max_rows": 50},
            },
            "directories": {
                "documents_dir": str(tmp_path / "docs"),
                "vector_store_dir": str(tmp_path / "vs"),
            },
        })
        assert config.sql.enabled is True
        assert config.sql.connection.sqlite_path == "/tmp/test.db"
        assert config.sql.security.max_rows == 50

    def test_from_dict_preserves_prompt_template(self, tmp_path):
        config = RAGConfig.from_dict({
            "prompt_template": "academic",
            "directories": {
                "documents_dir": str(tmp_path / "docs"),
                "vector_store_dir": str(tmp_path / "vs"),
            },
        })
        assert config.prompt_template == "academic"

    def test_from_dict_parses_corrective_rag(self, tmp_path):
        config = RAGConfig.from_dict({
            "corrective_rag": {
                "enabled": True,
                "relevance_threshold": 0.7,
                "max_retries": 2,
            },
            "directories": {
                "documents_dir": str(tmp_path / "docs"),
                "vector_store_dir": str(tmp_path / "vs"),
            },
        })
        assert config.corrective_rag.enabled is True
        assert config.corrective_rag.relevance_threshold == 0.7
        assert config.corrective_rag.max_retries == 2


class TestRAGConfigValidation:
    """RAGConfig.validate() cross-field checks."""

    def test_validate_syncs_vector_store_dir(self, sample_rag_config):
        sample_rag_config.vector_store.persist_directory = "/old/path"
        expected_dir = sample_rag_config.directories.vector_store_dir
        result = sample_rag_config.validate()
        assert result is True
        assert sample_rag_config.vector_store.persist_directory == expected_dir

    def test_corrective_rag_disabled_by_default(self):
        config = RAGConfig()
        assert config.corrective_rag.enabled is False


class TestDirectoryConfig:
    """DirectoryConfig creates directories on initialization."""

    def test_creates_directories_on_init(self, tmp_path):
        docs = str(tmp_path / "new_docs")
        vs = str(tmp_path / "new_vs")
        DirectoryConfig(documents_dir=docs, vector_store_dir=vs)
        assert os.path.isdir(docs)
        assert os.path.isdir(vs)
