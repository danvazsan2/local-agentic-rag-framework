"""
Reranker model configuration dataclass.
"""

from dataclasses import dataclass
from typing import Optional

from rag_framework.utils.constants import DEFAULT_RERANKER_TOP_N


@dataclass
class RerankerConfig:
    """Configuration for reranking models."""

    enabled: bool = True
    provider: str = "huggingface"  # "huggingface" | "ollama"
    model: str = "BAAI/bge-reranker-base"
    top_n: int = DEFAULT_RERANKER_TOP_N
    device: str = "auto"  # used by HuggingFace provider

    # Ollama-specific (used when provider="ollama")
    base_url: str = "http://localhost:11434"

    # Local model support (for already downloaded models)
    local_model_path: Optional[str] = None  # Path to local reranker model directory
    is_local: bool = False  # True if using a local model instead of downloading

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.local_model_path:
            self.is_local = True
