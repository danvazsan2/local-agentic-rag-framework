"""
Corrective RAG configuration dataclass.
"""

from dataclasses import dataclass


@dataclass
class CorrectiveRAGConfig:
    """
    Configuration for Corrective RAG (CRAG).

    Corrective RAG adds a document grading step after retrieval
    to filter out irrelevant documents before synthesis. When too
    few relevant documents remain, the query is rewritten and
    retrieval retried.

    Attributes:
        enabled: Enable/disable CRAG pipeline.
        relevance_threshold: Minimum ratio of relevant documents
            required to skip query rewriting. Value between 0 and 1.
            If the fraction of relevant docs is below this threshold,
            the query is rewritten and re-retrieved.
        max_retries: Number of query rewrite + re-retrieval attempts.
            Set to 0 to disable rewriting (grading only).
    """

    # Enable/disable Corrective RAG
    enabled: bool = False

    # Minimum relevance ratio to skip query rewriting (0.0 - 1.0)
    relevance_threshold: float = 0.5

    # Maximum rewrite + re-retrieval attempts
    max_retries: int = 1

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError(
                f"relevance_threshold must be between 0.0 and 1.0, "
                f"got {self.relevance_threshold}"
            )
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
