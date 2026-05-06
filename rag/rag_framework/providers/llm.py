"""
LLM Provider abstraction for multiple backends.

Supports:
- Ollama (local models)
- HuggingFace (local/remote models)
- OpenAI (future support)
"""

import logging
from typing import Any, Dict, Optional, Type

from rag_framework.config.llm_config import LLMConfig
from rag_framework.providers.base import BaseLLMProvider, ProviderFactory
from rag_framework.providers.common import OllamaValidator, HuggingFaceValidator

logger = logging.getLogger(__name__)


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama LLM provider for local models."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def get_llm(self) -> Any:
        """Get Ollama LLM instance."""
        from llama_index.llms.ollama import Ollama

        logger.info(f"Configuring Ollama LLM: {self.config.model}")
        logger.debug(f"  temperature: {self.config.temperature}")
        logger.debug(f"  top_k: {self.config.top_k}, top_p: {self.config.top_p}")

        llm = Ollama(
            model=self.config.model,
            base_url=self.config.base_url,
            request_timeout=self.config.request_timeout,
            context_window=self.config.context_window,
            temperature=self.config.temperature,
            thinking=self.config.thinking,
            additional_kwargs={
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_predict": self.config.max_tokens,
                "repeat_penalty": self.config.repeat_penalty,
                "stop": self.config.stop_sequences,
            },
        )

        return llm

    def validate(self) -> bool:
        """Validate Ollama connection and model availability."""
        if not OllamaValidator.check_connection(self.config.base_url):
            logger.error(f"Cannot connect to Ollama at {self.config.base_url}")
            return False

        if not OllamaValidator.check_model(self.config.base_url, self.config.model):
            logger.warning(
                f"Model {self.config.model} not found. Run: ollama pull {self.config.model}"
            )
            return False

        return True


class HuggingFaceLLMProvider(BaseLLMProvider):
    """HuggingFace LLM provider for local/remote models."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def get_llm(self) -> Any:
        """Get HuggingFace LLM instance."""
        from llama_index.llms.huggingface import HuggingFaceLLM

        # Determine if using local or remote model
        if self.config.is_local and self.config.local_model_path:
            model_id = self.config.local_model_path
            logger.info(f"Configuring Local HuggingFace LLM: {model_id}")
            logger.debug("  Source: Local model (offline)")
        else:
            model_id = self.config.hf_model_id or self.config.model
            logger.info(f"Configuring HuggingFace LLM: {model_id}")
            logger.debug("  Source: HuggingFace Hub")

        logger.debug(f"  device: {self.config.device}")
        logger.debug(f"  temperature: {self.config.temperature}")

        # Determine device
        device = self._get_device()

        # Configure generation kwargs
        generate_kwargs = {
            "temperature": (
                self.config.temperature if self.config.temperature > 0 else None
            ),
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_new_tokens": self.config.max_tokens,
            "repetition_penalty": self.config.repeat_penalty,
            "do_sample": self.config.temperature > 0,
        }

        # Remove None values
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        # Model kwargs for loading
        model_kwargs = {
            "torch_dtype": "auto",
        }

        if self.config.hf_token and not self.config.is_local:
            model_kwargs["token"] = self.config.hf_token

        llm = HuggingFaceLLM(
            model_name=model_id,
            tokenizer_name=model_id,
            context_window=self.config.context_window,
            max_new_tokens=self.config.max_tokens,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            device_map=device,
        )

        return llm

    def _get_device(self) -> str:
        """Determine the best available device."""
        return HuggingFaceValidator.get_best_device(self.config.device)

    def validate(self) -> bool:
        """Validate HuggingFace model availability."""
        # If using local model, check if path exists
        if self.config.is_local and self.config.local_model_path:
            from pathlib import Path

            model_path = Path(self.config.local_model_path)
            if model_path.exists():
                logger.info(f"Local model found at: {self.config.local_model_path}")
                return True
            else:
                logger.warning(
                    f"Local model not found at: {self.config.local_model_path}"
                )
                return False

        # For remote models, check HuggingFace Hub
        model_id = self.config.hf_model_id or self.config.model
        if HuggingFaceValidator.check_model(model_id, token=self.config.hf_token):
            return True

        logger.warning(f"Cannot validate HuggingFace model: {model_id}")
        return True  # Allow to proceed, may work with cached model


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM provider (for future support)."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def get_llm(self) -> Any:
        """Get OpenAI LLM instance."""
        raise NotImplementedError(
            "OpenAI support is not yet implemented. "
            "Please use 'ollama' or 'huggingface' provider."
        )

    def validate(self) -> bool:
        """Validate OpenAI API key."""
        import os

        return bool(os.getenv("OPENAI_API_KEY"))


class LLMFactory(ProviderFactory[LLMConfig, BaseLLMProvider]):
    """Factory for creating LLM provider instances."""

    _registry: Dict[str, Type[BaseLLMProvider]] = {}

    @classmethod
    def get_llm(cls, config: LLMConfig) -> Any:
        """
        Convenience method to directly get the LLM instance.

        Args:
            config: LLM configuration

        Returns:
            LLM instance compatible with LlamaIndex
        """
        provider = cls.create(config)
        return provider.get_llm()


# Auto-registration
LLMFactory.register("ollama", OllamaLLMProvider)
LLMFactory.register("huggingface", HuggingFaceLLMProvider)
LLMFactory.register("openai", OpenAILLMProvider)
