"""
Embedding model configuration dataclass.
"""

from dataclasses import dataclass
from typing import Optional

from rag_framework.config.enums import EmbeddingProvider


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    # Provider selection
    provider: str = "ollama"
    model: str = "nomic-embed-text:v1.5"

    # Ollama-specific
    base_url: str = "http://localhost:11434"

    # HuggingFace-specific
    hf_model_id: Optional[str] = None  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
    hf_token: Optional[str] = None
    device: str = "auto"

    # Local model support (for already downloaded models)
    local_model_path: Optional[str] = None  # Path to local model directory
    is_local: bool = False  # True if using a local model instead of downloading

    # Embedding dimensions (auto-detected if not specified)
    dimensions: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_providers = [p.value for p in EmbeddingProvider]
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid embedding provider: {self.provider}. "
                f"Supported: {valid_providers}"
            )

        if self.provider == "huggingface":
            if self.local_model_path:
                self.is_local = True
            elif not self.hf_model_id:
                self.hf_model_id = self.model
