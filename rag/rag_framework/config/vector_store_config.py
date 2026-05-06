"""
Vector store configuration dataclass.
"""

from dataclasses import dataclass
from typing import Optional

from rag_framework.config.enums import VectorStoreProvider


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""

    # Provider selection
    provider: str = "lancedb"

    # Storage path
    persist_directory: str = "./vector_store"
    collection_name: str = "documents"

    # LanceDB-specific
    lance_mode: str = "overwrite"  # "overwrite", "append"

    # Chroma-specific
    chroma_host: Optional[str] = None  # For remote Chroma
    chroma_port: Optional[int] = None

    # FAISS-specific
    faiss_index_type: str = "flat"  # "flat", "ivf", "hnsw"

    # Qdrant-specific
    qdrant_url: Optional[str] = None  # For remote Qdrant
    qdrant_api_key: Optional[str] = None

    # Pinecone-specific
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_providers = [p.value for p in VectorStoreProvider]
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid vector store provider: {self.provider}. "
                f"Supported: {valid_providers}"
            )
