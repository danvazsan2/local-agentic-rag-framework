"""Chroma vector store provider (local or remote)."""

import logging
from pathlib import Path
from typing import Any

from rag_framework.config.vector_store_config import VectorStoreConfig
from rag_framework.providers.vector_stores.base import BaseVectorStoreProvider

logger = logging.getLogger(__name__)


class ChromaVectorStoreProvider(BaseVectorStoreProvider):
    """Chroma vector store provider (local or remote)."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure the persist directory exists for local mode."""
        if not self.config.chroma_host:
            Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)

    def get_vector_store(self) -> Any:
        """Get Chroma vector store instance."""
        try:
            from llama_index.vector_stores.chroma import ChromaVectorStore
            import chromadb
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Chroma dependencies are not installed. "
                "Install with: pip install llama-index-vector-stores-chroma chromadb"
            ) from e

        logger.info("Configuring Chroma")

        if self.config.chroma_host and self.config.chroma_port:
            # Remote Chroma
            logger.debug(
                f"  remote: {self.config.chroma_host}:{self.config.chroma_port}"
            )
            client = chromadb.HttpClient(
                host=self.config.chroma_host,
                port=self.config.chroma_port,
            )
        else:
            # Local Chroma
            logger.debug(f"  local: {self.config.persist_directory}")
            client = chromadb.PersistentClient(path=str(self.config.persist_directory))

        # Get or create collection
        collection = client.get_or_create_collection(name=self.config.collection_name)

        logger.debug(f"  collection: {self.config.collection_name}")

        vector_store = ChromaVectorStore(chroma_collection=collection)

        return vector_store

    def exists(self) -> bool:
        """Check if Chroma collection exists and has data."""
        import chromadb

        try:
            if self.config.chroma_host and self.config.chroma_port:
                client = chromadb.HttpClient(
                    host=self.config.chroma_host,
                    port=self.config.chroma_port,
                )
            else:
                client = chromadb.PersistentClient(
                    path=str(self.config.persist_directory)
                )

            collections = client.list_collections()
            collection_names = [c.name for c in collections]

            if self.config.collection_name in collection_names:
                collection = client.get_collection(self.config.collection_name)
                return collection.count() > 0

            return False
        except Exception:
            return False

    def clear(self) -> None:
        """Clear Chroma collection."""
        import chromadb

        try:
            if self.config.chroma_host and self.config.chroma_port:
                client = chromadb.HttpClient(
                    host=self.config.chroma_host,
                    port=self.config.chroma_port,
                )
            else:
                client = chromadb.PersistentClient(
                    path=str(self.config.persist_directory)
                )

            client.delete_collection(self.config.collection_name)
            logger.info(f"Cleared Chroma collection: {self.config.collection_name}")
        except Exception as e:
            logger.warning(f"Could not clear Chroma: {e}")
