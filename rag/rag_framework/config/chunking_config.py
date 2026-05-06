"""
Document chunking configuration dataclass.
"""

from dataclasses import dataclass

from rag_framework.utils.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    separator: str = " "

    # Semantic chunking: use DoclingNodeParser to split by document structure
    # (sections, tables, paragraphs) instead of fixed-size windows.
    # Chunks that exceed chunk_size * semantic_oversized_factor are re-split
    # with SentenceSplitter to keep a controlled size.
    use_semantic_chunking: bool = True
    semantic_oversized_factor: float = 1.5

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.semantic_oversized_factor <= 0:
            raise ValueError("semantic_oversized_factor must be positive")
