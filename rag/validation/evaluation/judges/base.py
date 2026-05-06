"""
Abstract base class for LLM judges.

All judges implement faithfulness and answer relevancy scoring
with a consistent interface. Judges are model-agnostic — the
constructor accepts a model name for cross-validation.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseJudge(ABC):
    """Abstract LLM judge for generation quality evaluation."""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir

    @abstractmethod
    def score_faithfulness(self, context: str, response: str) -> float:
        """Score faithfulness: is the response supported by the context?

        Returns:
            Float in [0, 1]. 1 = fully faithful.
        """
        ...

    @abstractmethod
    def score_relevancy(self, question: str, response: str) -> float:
        """Score answer relevancy: does the response answer the question?

        Returns:
            Float in [0, 1]. 1 = fully relevant.
        """
        ...

    @abstractmethod
    def get_prompt_version(self) -> str:
        """Return a string identifying the prompt strategy used."""
        ...
