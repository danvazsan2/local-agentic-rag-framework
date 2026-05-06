"""FAISS vector store provider (local, efficient for large datasets)."""

import logging
from pathlib import Path
from typing import Any

from rag_framework.config.vector_store_config import VectorStoreConfig
from rag_framework.providers.vector_stores.base import BaseVectorStoreProvider

logger = logging.getLogger(__name__)


class FAISSVectorStoreProvider(BaseVectorStoreProvider):
    """FAISS vector store provider (local, efficient for large datasets)."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure the persist directory exists."""
        Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)

    def get_vector_store(self) -> Any:
        """Get FAISS vector store instance."""
        from llama_index.vector_stores.faiss import FaissVectorStore
        import faiss

        logger.info(f"Configuring FAISS at: {self.config.persist_directory}")
        logger.debug(f"  index type: {self.config.faiss_index_type}")

        # FAISS requires dimension at creation time
        # This will be set when first vectors are added
        # For now, use a reasonable default that will be overwritten
        dimension = 768  # Common embedding dimension

        if self.config.faiss_index_type == "flat":
            faiss_index = faiss.IndexFlatL2(dimension)
        elif self.config.faiss_index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            faiss_index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        elif self.config.faiss_index_type == "hnsw":
            faiss_index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            faiss_index = faiss.IndexFlatL2(dimension)

        vector_store = FaissVectorStore(faiss_index=faiss_index)

        return vector_store

    def exists(self) -> bool:
        """Check if FAISS index exists."""
        index_path = Path(self.config.persist_directory) / "index.faiss"
        return index_path.exists()

    def clear(self) -> None:
        """Clear FAISS index files."""
        try:
            index_path = Path(self.config.persist_directory) / "index.faiss"
            if index_path.exists():
                index_path.unlink()

            docstore_path = Path(self.config.persist_directory) / "docstore.json"
            if docstore_path.exists():
                docstore_path.unlink()

            logger.info("Cleared FAISS index")
        except Exception as e:
            logger.warning(f"Could not clear FAISS: {e}")
