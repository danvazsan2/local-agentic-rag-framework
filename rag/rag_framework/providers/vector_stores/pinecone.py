"""Pinecone vector store provider (cloud)."""

import logging
from typing import Any

from rag_framework.config.vector_store_config import VectorStoreConfig
from rag_framework.providers.vector_stores.base import BaseVectorStoreProvider

logger = logging.getLogger(__name__)


class PineconeVectorStoreProvider(BaseVectorStoreProvider):
    """Pinecone vector store provider (cloud)."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config

    def get_vector_store(self) -> Any:
        """Get Pinecone vector store instance."""
        from llama_index.vector_stores.pinecone import PineconeVectorStore
        from pinecone import Pinecone

        if not self.config.pinecone_api_key:
            raise ValueError(
                "Pinecone API key is required. Set PINECONE_API_KEY environment variable."
            )

        logger.info("Configuring Pinecone")
        logger.debug(
            f"  index: {self.config.pinecone_index_name or self.config.collection_name}"
        )

        pc = Pinecone(api_key=self.config.pinecone_api_key)

        index_name = self.config.pinecone_index_name or self.config.collection_name
        index = pc.Index(index_name)

        vector_store = PineconeVectorStore(pinecone_index=index)

        return vector_store

    def exists(self) -> bool:
        """Check if Pinecone index exists."""
        from pinecone import Pinecone

        try:
            if not self.config.pinecone_api_key:
                return False

            pc = Pinecone(api_key=self.config.pinecone_api_key)
            indexes = pc.list_indexes()
            index_name = self.config.pinecone_index_name or self.config.collection_name

            return index_name in [idx.name for idx in indexes]
        except Exception:
            return False

    def clear(self) -> None:
        """Clear Pinecone index (delete all vectors)."""
        from pinecone import Pinecone

        try:
            pc = Pinecone(api_key=self.config.pinecone_api_key)
            index_name = self.config.pinecone_index_name or self.config.collection_name
            index = pc.Index(index_name)

            # Delete all vectors
            index.delete(delete_all=True)
            logger.info(f"Cleared Pinecone index: {index_name}")
        except Exception as e:
            logger.warning(f"Could not clear Pinecone: {e}")
