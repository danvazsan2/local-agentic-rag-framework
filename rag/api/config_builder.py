"""Configuration builder for API sessions.

Constructs RAGConfig instances from API request parameters,
applying defaults and validations.
"""

from pathlib import Path
from typing import Optional

from rag_framework.utils.logging import get_logger
from rag_framework.utils.constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
)
from rag_framework.config import (
    RAGConfig,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    ChunkingConfig,
    RetrievalConfig,
)
from rag_framework.config.models import DirectoryConfig

logger = get_logger(__name__)


def _resolve_retrieval_mode(retrieval_mode: Optional[str]):
    """Map retrieval_mode string to (use_hybrid_search, alpha)."""
    if retrieval_mode == "semantic":
        return False, 1.0
    elif retrieval_mode == "bm25":
        return True, 0.0
    else:
        return True, 0.5


def write_session_yaml(session_id: str, config: RAGConfig, sessions_dir: Path) -> Path:
    """Write the effective session config as YAML for reproducibility.

    The file is placed at sessions_dir/{session_id}.yaml so it can be inspected
    or reused as a preset in future sessions.
    """
    import yaml

    sessions_dir.mkdir(parents=True, exist_ok=True)
    output_path = sessions_dir / f"{session_id}.yaml"

    data = {
        "llm": {
            "provider": config.llm.provider,
            "model": config.llm.model,
        },
        "embedding": {
            "provider": config.embedding.provider,
            "model": config.embedding.model,
        },
        "vector_store": {
            "provider": config.vector_store.provider,
        },
        "prompt_template": config.prompt_template,
        "sql": {
            "enabled": config.sql.enabled,
            "sqlite_path": config.sql.connection.sqlite_path,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    logger.info("Session config written to %s", output_path)
    return output_path


def build_session_config(
    base_config: Optional[RAGConfig],
    documents_dir: Path,
    vector_store_dir: Path,
    llm_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    vector_store_type: Optional[str] = None,
    retrieval_mode: Optional[str] = None,
) -> RAGConfig:
    """Build a RAGConfig from session parameters.

    Args:
        base_config: Optional base configuration to inherit values from
        documents_dir: Directory path for session documents
        vector_store_dir: Directory path for session vector store
        llm_model: Optional LLM model override
        embedding_model: Optional embedding model override
        vector_store_type: Optional vector store provider ("lancedb", "chroma", "faiss")
        retrieval_mode: Optional retrieval mode ("semantic", "bm25", "hybrid")

    Returns:
        Configured RAGConfig instance
    """
    resolved_vector_store = vector_store_type or "lancedb"
    use_hybrid, alpha = _resolve_retrieval_mode(retrieval_mode)

    # Use base config values if available
    if base_config:
        if retrieval_mode is None:
            use_hybrid = base_config.retrieval.use_hybrid_search
            alpha = base_config.retrieval.alpha
        return RAGConfig(
            llm=LLMConfig(
                provider=base_config.llm.provider,
                model=llm_model or base_config.llm.model,
                base_url=base_config.llm.base_url,
                temperature=base_config.llm.temperature,
                max_tokens=base_config.llm.max_tokens,
            ),
            embedding=EmbeddingConfig(
                provider=base_config.embedding.provider,
                model=embedding_model or base_config.embedding.model,
                base_url=base_config.embedding.base_url,
            ),
            vector_store=VectorStoreConfig(
                provider=resolved_vector_store,
                persist_directory=str(vector_store_dir),
                collection_name="session_docs",
                lance_mode="overwrite",
            ),
            chunking=ChunkingConfig(
                chunk_size=base_config.chunking.chunk_size,
                chunk_overlap=base_config.chunking.chunk_overlap,
            ),
            retrieval=RetrievalConfig(
                use_hybrid_search=use_hybrid,
                alpha=alpha,
                top_k=base_config.retrieval.top_k,
                reranker=base_config.retrieval.reranker,
            ),
            directories=DirectoryConfig(
                documents_dir=str(documents_dir),
                vector_store_dir=str(vector_store_dir),
            ),
            prompt_template=base_config.prompt_template,
            corrective_rag=base_config.corrective_rag,
            debug=base_config.debug,
        )

    # Default configuration
    return RAGConfig(
        llm=LLMConfig(
            provider="ollama",
            model=llm_model or DEFAULT_LLM_MODEL,
            base_url="http://localhost:11434",
            temperature=0.0,
            max_tokens=1024,
        ),
        embedding=EmbeddingConfig(
            provider="ollama",
            model=embedding_model or DEFAULT_EMBEDDING_MODEL,
            base_url="http://localhost:11434",
        ),
        vector_store=VectorStoreConfig(
            provider=resolved_vector_store,
            persist_directory=str(vector_store_dir),
            collection_name="session_docs",
            lance_mode="overwrite",
        ),
        chunking=ChunkingConfig(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        ),
        retrieval=RetrievalConfig(
            use_hybrid_search=use_hybrid,
            alpha=alpha,
            top_k=DEFAULT_TOP_K,
        ),
        directories=DirectoryConfig(
            documents_dir=str(documents_dir),
            vector_store_dir=str(vector_store_dir),
        ),
        prompt_template="default",
        debug=False,
    )
