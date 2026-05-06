"""Qdrant vector store provider (local or cloud)."""

import logging
from pathlib import Path
from typing import Any

from rag_framework.config.vector_store_config import VectorStoreConfig
from rag_framework.providers.vector_stores.base import BaseVectorStoreProvider

logger = logging.getLogger(__name__)


class QdrantVectorStoreProvider(BaseVectorStoreProvider):
    """Qdrant vector store provider (local or cloud)."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure the persist directory exists for local mode."""
        if not self.config.qdrant_url:
            Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)

    def get_vector_store(self) -> Any:
        """Get Qdrant vector store instance."""
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        logger.info("Configuring Qdrant")

        if self.config.qdrant_url:
            # Remote Qdrant
            logger.debug(f"  remote: {self.config.qdrant_url}")
            client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
            )
        else:
            # Local Qdrant
            logger.debug(f"  local: {self.config.persist_directory}")
            client = QdrantClient(path=str(self.config.persist_directory))

        logger.debug(f"  collection: {self.config.collection_name}")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.config.collection_name,
        )

        return vector_store

    def exists(self) -> bool:
        """Check if Qdrant collection exists."""
        from qdrant_client import QdrantClient

        try:
            if self.config.qdrant_url:
                client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key,
                )
            else:
                client = QdrantClient(path=str(self.config.persist_directory))

            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            return self.config.collection_name in collection_names
        except Exception:
            return False

    def clear(self) -> None:
        """Clear Qdrant collection."""
        from qdrant_client import QdrantClient

        try:
            if self.config.qdrant_url:
                client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key,
                )
            else:
                client = QdrantClient(path=str(self.config.persist_directory))

            client.delete_collection(self.config.collection_name)
            logger.info(f"Cleared Qdrant collection: {self.config.collection_name}")
        except Exception as e:
            logger.warning(f"Could not clear Qdrant: {e}")
