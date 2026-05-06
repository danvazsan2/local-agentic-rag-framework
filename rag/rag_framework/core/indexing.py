"""
Index management module.

Handles creation and loading of vector indexes using configurable
embedding models and vector stores.
"""

import os
import pickle
from typing import List, Optional, Any, Tuple

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode

from rag_framework.config.models import RAGConfig
from rag_framework.providers.embeddings import EmbeddingFactory
from rag_framework.providers.vector_stores import (
    VectorStoreFactory,
    LanceDBVectorStoreProvider,
)


class IndexManager:
    """
    Manages vector index creation, storage, and retrieval.

    Works with multiple vector store backends and embedding models.
    """

    NODES_FILENAME = "nodes.pkl"

    def __init__(self, config: RAGConfig):
        """
        Initialize the index manager.

        Args:
            config: RAG configuration
        """
        self.config = config
        self._embed_model = None
        self._vector_store_provider = None

    @property
    def embed_model(self):
        """Lazily load the embedding model."""
        if self._embed_model is None:
            self._embed_model = EmbeddingFactory.get_embedding_model(
                self.config.embedding
            )
        return self._embed_model

    @property
    def vector_store_provider(self):
        """Lazily load the vector store provider."""
        if self._vector_store_provider is None:
            self._vector_store_provider = VectorStoreFactory.create(
                self.config.vector_store
            )
        return self._vector_store_provider

    @property
    def nodes_path(self) -> str:
        """Path to save/load nodes for hybrid search."""
        persist_dir = self.config.vector_store.persist_directory
        return os.path.join(persist_dir, self.NODES_FILENAME)

    def _save_nodes(self, nodes: List[TextNode]) -> None:
        """Save nodes to disk for later hybrid search."""
        os.makedirs(os.path.dirname(self.nodes_path), exist_ok=True)
        with open(self.nodes_path, "wb") as f:
            pickle.dump(nodes, f)
        print(f"Saved {len(nodes)} nodes for hybrid search")

    def _load_nodes(self) -> Optional[List[TextNode]]:
        """Load nodes from disk for hybrid search."""
        if os.path.exists(self.nodes_path):
            try:
                with open(self.nodes_path, "rb") as f:
                    nodes = pickle.load(f)
                print(f"Loaded {len(nodes)} nodes for hybrid search")
                return nodes
            except Exception as e:
                print(f"Could not load nodes: {e}")
        return None

    def create_index(
        self, nodes: List[TextNode], show_progress: bool = True
    ) -> VectorStoreIndex:
        """
        Create a new vector index from nodes.

        Args:
            nodes: List of text nodes to index
            show_progress: Whether to show progress bar

        Returns:
            VectorStoreIndex instance
        """
        if not nodes:
            raise ValueError("No nodes provided for indexing")

        print("=" * 50)
        print("CREATING VECTOR INDEX")
        print("=" * 50)

        # Save nodes for hybrid search
        self._save_nodes(nodes)

        # Get vector store
        vector_store = self.vector_store_provider.get_vector_store()

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index
        print(f"Indexing {len(nodes)} nodes...")

        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=show_progress,
        )

        print("=" * 50)
        print("INDEX CREATED SUCCESSFULLY")
        print("=" * 50)

        return index

    def load_index(self) -> Tuple[Optional[VectorStoreIndex], Optional[List[TextNode]]]:
        """
        Load an existing index from the vector store.

        Returns:
            Tuple of (VectorStoreIndex, nodes) if exists, (None, None) otherwise
        """
        if not self.vector_store_provider.exists():
            print("[AVISO] No existing index found")
            return None, None

        print("📂 Loading existing index...")

        try:
            # For LanceDB, we need to use a different method to load
            if isinstance(self.vector_store_provider, LanceDBVectorStoreProvider):
                vector_store = self.vector_store_provider.get_vector_store_for_query()
            else:
                vector_store = self.vector_store_provider.get_vector_store()

            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model,
            )

            # Load nodes for hybrid search
            nodes = self._load_nodes()

            print("Index loaded successfully")
            return index, nodes

        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return None, None

    def index_exists(self) -> bool:
        """Check if an index already exists."""
        return self.vector_store_provider.exists()

    def clear_index(self) -> None:
        """Clear the existing index."""
        self.vector_store_provider.clear()

    def get_or_create_index(
        self, nodes: Optional[List[TextNode]] = None, force_recreate: bool = False
    ) -> Optional[VectorStoreIndex]:
        """
        Get existing index or create a new one.

        Args:
            nodes: Nodes to index if creating new index
            force_recreate: Force recreation even if index exists

        Returns:
            VectorStoreIndex instance or None
        """
        if force_recreate and nodes:
            self.clear_index()
            return self.create_index(nodes)

        existing = self.load_index()
        if existing:
            return existing

        if nodes:
            return self.create_index(nodes)

        return None
