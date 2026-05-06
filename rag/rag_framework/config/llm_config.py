"""
LLM (Language Model) configuration dataclass.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from rag_framework.config.enums import LLMProvider


@dataclass
class LLMConfig:
    """Configuration for the LLM (Language Model)."""

    # Provider selection
    provider: str = "ollama"
    model: str = "llama3-instruct-8k"

    # Ollama-specific
    base_url: str = "http://localhost:11434"

    # HuggingFace-specific
    hf_model_id: Optional[str] = None  # e.g., "meta-llama/Llama-2-7b-chat-hf"
    hf_token: Optional[str] = None
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"

    # Local model support (for already downloaded models)
    local_model_path: Optional[str] = None  # Path to local model directory
    is_local: bool = False  # True if using a local model instead of downloading

    # Generation parameters
    context_window: int = 8192
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 512
    repeat_penalty: float = 1.15
    stop_sequences: List[str] = field(
        default_factory=lambda: ["\n\nPregunta:", "\n\nUsuario:", "\n\nContexto:"]
    )

    # Request settings
    request_timeout: float = 120.0

    # Thinking mode (Ollama only — for models like qwen3 that support chain-of-thought)
    # False = disable thinking (faster, direct responses — recommended for RAG)
    # True  = enable default thinking
    # None  = use model's built-in default behaviour
    thinking: Optional[bool] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_providers = [p.value for p in LLMProvider]
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid LLM provider: {self.provider}. "
                f"Supported: {valid_providers}"
            )

        if self.provider == "huggingface":
            if self.local_model_path:
                self.is_local = True
            elif not self.hf_model_id:
                self.hf_model_id = self.model
