"""Tests for the generic ProviderFactory and concrete factory classes."""

import pytest
from dataclasses import dataclass

from rag_framework.providers.base import ProviderFactory, BaseLLMProvider
from rag_framework.exceptions import ConfigurationError
from rag_framework.config import LLMConfig, EmbeddingConfig, RerankerConfig
from rag_framework.config.vector_store_config import VectorStoreConfig


# ---------------------------------------------------------------------------
# Minimal mock providers for factory-level tests
# ---------------------------------------------------------------------------


@dataclass
class _SimpleConfig:
    provider: str = "mock"
    model: str = "test"


class _MockLLMProvider(BaseLLMProvider):
    def __init__(self, config):
        self.config = config

    def get_llm(self):
        return None

    def validate(self):
        return True


class _FailingProvider(BaseLLMProvider):
    def __init__(self, config):
        self.config = config

    def get_llm(self):
        return None

    def validate(self):
        return False


# ---------------------------------------------------------------------------
# ProviderFactory base class
# ---------------------------------------------------------------------------


class TestProviderFactoryCore:
    """Generic ProviderFactory registration, creation, and validation."""

    def _make_factory(self):
        class TestFactory(ProviderFactory):
            _registry = {}
        return TestFactory

    def test_register_and_create(self):
        Factory = self._make_factory()
        Factory.register("mock", _MockLLMProvider)

        provider = Factory.create(_SimpleConfig(provider="mock"))

        assert isinstance(provider, _MockLLMProvider)

    def test_create_unknown_provider_raises(self):
        Factory = self._make_factory()
        with pytest.raises(ConfigurationError, match="Unknown provider"):
            Factory.create(_SimpleConfig(provider="does_not_exist"))

    def test_create_missing_provider_attr_raises(self):
        Factory = self._make_factory()

        class BareConfig:
            pass

        with pytest.raises(ConfigurationError, match="missing 'provider'"):
            Factory.create(BareConfig())

    def test_validate_delegates_to_provider(self):
        Factory = self._make_factory()
        Factory.register("mock", _MockLLMProvider)
        assert Factory.validate(_SimpleConfig(provider="mock")) is True

    def test_validate_returns_false_for_failing_provider(self):
        Factory = self._make_factory()
        Factory.register("fail", _FailingProvider)
        assert Factory.validate(_SimpleConfig(provider="fail")) is False

    def test_registries_are_isolated_between_subclasses(self):
        FactoryA = self._make_factory()
        FactoryB = self._make_factory()
        FactoryA.register("mock", _MockLLMProvider)

        assert "mock" in FactoryA._registry
        assert "mock" not in FactoryB._registry


# ---------------------------------------------------------------------------
# Concrete factory classes
# ---------------------------------------------------------------------------


class TestLLMFactory:
    def test_create_ollama_provider(self):
        from rag_framework.providers.llm import LLMFactory, OllamaLLMProvider

        config = LLMConfig(provider="ollama", model="llama3")
        assert isinstance(LLMFactory.create(config), OllamaLLMProvider)

    def test_create_huggingface_provider(self):
        from rag_framework.providers.llm import LLMFactory, HuggingFaceLLMProvider

        config = LLMConfig(provider="huggingface", model="tiiuae/falcon-7b")
        assert isinstance(LLMFactory.create(config), HuggingFaceLLMProvider)


class TestEmbeddingFactory:
    def test_create_ollama_provider(self):
        from rag_framework.providers.embeddings import EmbeddingFactory, OllamaEmbeddingProvider

        config = EmbeddingConfig(provider="ollama", model="nomic-embed-text")
        assert isinstance(EmbeddingFactory.create(config), OllamaEmbeddingProvider)


class TestVectorStoreFactory:
    def test_create_lancedb_provider(self, tmp_path):
        from rag_framework.providers.vector_stores import VectorStoreFactory, LanceDBVectorStoreProvider

        config = VectorStoreConfig(provider="lancedb", persist_directory=str(tmp_path / "lance"))
        assert isinstance(VectorStoreFactory.create(config), LanceDBVectorStoreProvider)

    def test_list_providers_includes_expected_backends(self):
        from rag_framework.providers.vector_stores import VectorStoreFactory

        providers = VectorStoreFactory.list_providers()
        assert "lancedb" in providers
        assert "chroma" in providers


# ---------------------------------------------------------------------------
# Validator convenience functions
# ---------------------------------------------------------------------------


class TestValidators:
    def test_validate_reranker_disabled_skips(self, sample_rag_config):
        from rag_framework.validators import validate_reranker_config

        sample_rag_config.retrieval.reranker.enabled = False
        assert validate_reranker_config(sample_rag_config) is True

    def test_validate_all_providers_returns_bool(self, sample_rag_config):
        from rag_framework.validators import validate_all_providers

        assert isinstance(validate_all_providers(sample_rag_config), bool)
