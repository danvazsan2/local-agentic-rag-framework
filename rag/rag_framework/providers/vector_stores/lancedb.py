"""LanceDB vector store provider (local, fast, default)."""

import logging
from pathlib import Path
from typing import Any

from rag_framework.config.vector_store_config import VectorStoreConfig
from rag_framework.providers.vector_stores.base import BaseVectorStoreProvider

logger = logging.getLogger(__name__)


class LanceDBVectorStoreProvider(BaseVectorStoreProvider):
    """LanceDB vector store provider (local, fast, default)."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure the persist directory exists."""
        Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)

    def get_vector_store(self) -> Any:
        """Get LanceDB vector store instance for indexing."""
        from llama_index.vector_stores.lancedb import LanceDBVectorStore
        import lancedb

        logger.info(f"Configuring LanceDB at: {self.config.persist_directory}")
        logger.debug(f"  collection: {self.config.collection_name}")
        logger.debug(f"  mode: {self.config.lance_mode}")

        # If overwrite mode, drop existing table first
        if self.config.lance_mode == "overwrite":
            try:
                db = lancedb.connect(str(self.config.persist_directory))
                if self.config.collection_name in db.table_names():
                    db.drop_table(self.config.collection_name)
                    logger.debug("  dropped existing table")
            except Exception:
                pass

        vector_store = LanceDBVectorStore(
            uri=str(self.config.persist_directory),
            table_name=self.config.collection_name,
        )

        return vector_store

    def get_vector_store_for_query(self) -> Any:
        """Get LanceDB vector store for querying (no overwrite)."""
        from llama_index.vector_stores.lancedb import LanceDBVectorStore

        vector_store = LanceDBVectorStore(
            uri=str(self.config.persist_directory),
            table_name=self.config.collection_name,
        )

        return vector_store

    def exists(self) -> bool:
        """Check if LanceDB table exists."""
        import lancedb

        try:
            db = lancedb.connect(str(self.config.persist_directory))
            tables = db.table_names()
            return self.config.collection_name in tables
        except Exception:
            return False

    def clear(self) -> None:
        """Clear LanceDB table."""
        import lancedb

        try:
            db = lancedb.connect(str(self.config.persist_directory))
            if self.config.collection_name in db.table_names():
                db.drop_table(self.config.collection_name)
                logger.info(f"Cleared LanceDB table: {self.config.collection_name}")
        except Exception as e:
            logger.warning(f"Could not clear LanceDB: {e}")
