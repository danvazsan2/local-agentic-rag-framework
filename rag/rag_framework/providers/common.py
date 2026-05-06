"""Shared utilities for provider validation and health checks."""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class OllamaValidator:
    """Shared Ollama connectivity and model validation."""

    @staticmethod
    def check_connection(base_url: str, timeout: int = 5) -> bool:
        """Verify Ollama server is reachable.

        Args:
            base_url: Ollama server URL (e.g., "http://localhost:11434")
            timeout: Connection timeout in seconds

        Returns:
            True if server responds, False otherwise
        """
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=timeout)
            return response.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"Ollama connection failed: {e}")
            return False

    @staticmethod
    def check_model(base_url: str, model_name: str, timeout: int = 5) -> bool:
        """Verify a model is available in Ollama.

        Args:
            base_url: Ollama server URL
            model_name: Name of the model to check
            timeout: Connection timeout in seconds

        Returns:
            True if model exists, False otherwise
        """
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=timeout)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m.get("name") == model_name for m in models)
            return False
        except requests.RequestException as e:
            logger.warning(f"Failed to check Ollama model: {e}")
            return False


class HuggingFaceValidator:
    """Shared HuggingFace model validation."""

    @staticmethod
    def check_model(
        model_id: str, token: Optional[str] = None, timeout: int = 10
    ) -> bool:
        """Verify a model exists on HuggingFace Hub.

        Args:
            model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b")
            token: Optional HuggingFace API token for gated models
            timeout: Request timeout in seconds

        Returns:
            True if model exists and is accessible, False otherwise
        """
        try:
            from huggingface_hub import model_info

            model_info(model_id, token=token, timeout=timeout)
            return True
        except Exception as e:
            logger.warning(f"HuggingFace model check failed for {model_id}: {e}")
            return False

    @staticmethod
    def get_best_device(preference: str = "auto") -> str:
        """Determine the best available compute device.

        Args:
            preference: "auto", "cuda", "mps", or "cpu"

        Returns:
            Device string: "cuda", "mps", or "cpu"
        """
        if preference != "auto":
            return preference

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"
