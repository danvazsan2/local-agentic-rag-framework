"""
Root RAG configuration and directory configuration dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from rag_framework.config.llm_config import LLMConfig
from rag_framework.config.embedding_config import EmbeddingConfig
from rag_framework.config.vector_store_config import VectorStoreConfig
from rag_framework.config.chunking_config import ChunkingConfig
from rag_framework.config.reranker_config import RerankerConfig
from rag_framework.config.retrieval_config import (
    RetrievalConfig,
    RouterConfig,
)
from rag_framework.config.sql_config import (
    DatabaseConnectionConfig,
    SchemaConfig,
    SQLSecurityConfig,
    SQLConfig,
)
from rag_framework.config.corrective_rag_config import CorrectiveRAGConfig
from rag_framework.config.metadata_config import (
    MetadataExtractionConfig,
    MetadataFilterConfig,
    DbEnrichmentConfig,
    FilenamePatternConfig,
)


@dataclass
class DirectoryConfig:
    """Configuration for directories."""

    documents_dir: str = "./documents"
    vector_store_dir: str = "./vector_store"

    def __post_init__(self):
        """Create directories if they don't exist."""
        from pathlib import Path

        Path(self.documents_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class RAGConfig:
    """
    Main configuration class for the RAG Framework.

    Combines all configuration sections into a single object.
    """

    # Sub-configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)

    # SQL configuration (new)
    sql: SQLConfig = field(default_factory=SQLConfig)

    # Router configuration (new)
    router: RouterConfig = field(default_factory=RouterConfig)

    # Corrective RAG configuration
    corrective_rag: CorrectiveRAGConfig = field(default_factory=CorrectiveRAGConfig)

    # Metadata extraction & filtering configuration
    metadata: MetadataExtractionConfig = field(default_factory=MetadataExtractionConfig)

    # Prompt template
    prompt_template: str = "default"  # Name of template or "custom"
    custom_prompt: Optional[str] = None  # Custom prompt if prompt_template="custom"

    # Debug mode
    debug: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        """Create configuration from dictionary."""
        # Handle nested configurations
        llm_data = data.get("llm", {})
        embedding_data = data.get("embedding", {})
        vector_store_data = data.get("vector_store", {})
        chunking_data = data.get("chunking", {})
        retrieval_data = data.get("retrieval", {})
        directories_data = data.get("directories", {})
        sql_data = data.get("sql", {})
        router_data = data.get("router", {})
        corrective_rag_data = data.get("corrective_rag", {})
        metadata_data = data.get("metadata", {})

        # Handle reranker nested in retrieval
        if "reranker" in retrieval_data:
            retrieval_data["reranker"] = RerankerConfig(**retrieval_data["reranker"])

        # Handle SQL nested configs
        if sql_data:
            if "connection" in sql_data:
                sql_data["connection"] = DatabaseConnectionConfig(
                    **sql_data["connection"]
                )
            if "schema" in sql_data:
                sql_data["schema"] = SchemaConfig(**sql_data["schema"])
            if "security" in sql_data:
                sql_data["security"] = SQLSecurityConfig(**sql_data["security"])

        # Handle metadata nested configs
        metadata_obj = MetadataExtractionConfig()
        if metadata_data:
            # filename_patterns
            raw_patterns = metadata_data.get("filename_patterns", [])
            patterns = [
                (
                    FilenamePatternConfig(**p)
                    if isinstance(p, dict)
                    else FilenamePatternConfig(pattern=p)
                )
                for p in raw_patterns
            ]
            # db_enrichment
            db_enr_raw = metadata_data.get("db_enrichment", {})
            db_enr = (
                DbEnrichmentConfig(**db_enr_raw) if db_enr_raw else DbEnrichmentConfig()
            )
            # filtering
            filter_raw = metadata_data.get("filtering", {})
            filter_obj = (
                MetadataFilterConfig(**filter_raw)
                if filter_raw
                else MetadataFilterConfig()
            )
            metadata_obj = MetadataExtractionConfig(
                enabled=metadata_data.get("enabled", False),
                filename_patterns=patterns,
                db_enrichment=db_enr,
                embed_visible_fields=metadata_data.get("embed_visible_fields", []),
                llm_visible_fields=metadata_data.get("llm_visible_fields", []),
                filtering=filter_obj,
            )

        return cls(
            llm=LLMConfig(**llm_data) if llm_data else LLMConfig(),
            embedding=(
                EmbeddingConfig(**embedding_data)
                if embedding_data
                else EmbeddingConfig()
            ),
            vector_store=(
                VectorStoreConfig(**vector_store_data)
                if vector_store_data
                else VectorStoreConfig()
            ),
            chunking=(
                ChunkingConfig(**chunking_data) if chunking_data else ChunkingConfig()
            ),
            retrieval=(
                RetrievalConfig(**retrieval_data)
                if retrieval_data
                else RetrievalConfig()
            ),
            directories=(
                DirectoryConfig(**directories_data)
                if directories_data
                else DirectoryConfig()
            ),
            sql=SQLConfig(**sql_data) if sql_data else SQLConfig(),
            router=RouterConfig(**router_data) if router_data else RouterConfig(),
            corrective_rag=(
                CorrectiveRAGConfig(**corrective_rag_data)
                if corrective_rag_data
                else CorrectiveRAGConfig()
            ),
            metadata=metadata_obj,
            prompt_template=data.get("prompt_template", "default"),
            custom_prompt=data.get("custom_prompt"),
            debug=data.get("debug", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    def validate(self) -> bool:
        """Validate the entire configuration."""
        # All __post_init__ validations are called automatically
        # This method is for additional cross-field validation

        # Ensure vector store directory matches
        if self.vector_store.persist_directory != self.directories.vector_store_dir:
            self.vector_store.persist_directory = self.directories.vector_store_dir

        return True
