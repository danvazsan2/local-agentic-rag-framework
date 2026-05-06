"""Session management for RAG API.

Handles creation, storage, and lifecycle of RAG framework sessions
with thread-safe access and automatic cleanup.
"""

import shutil
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from rag_framework.utils.logging import get_logger
from rag_framework import RAGFramework, RAGConfig

from api.config_builder import build_session_config

logger = get_logger(__name__)


# =============================================================================
# SESSION MODEL
# =============================================================================


@dataclass
class RAGSession:
    """Represents a RAG session with its own isolated index."""

    session_id: str
    rag: RAGFramework
    documents_dir: Path
    vector_store_dir: Path
    default_llm_model: str
    default_embedding_model: str
    default_vector_store_provider: str
    default_prompt_template: str
    default_sql_path: str
    default_sql_enabled: bool
    data_source_mode: str = "documents"
    first_query_executed: bool = False
    configuration_locked: bool = False
    ingested_files: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure directories exist."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SESSION MANAGER
# =============================================================================


class SessionManager:
    """Manages RAG sessions with isolated vector stores.

    Each session has its own:
    - Documents directory
    - Vector store
    - RAG framework instance
    """

    def __init__(
        self, base_dir: str = "sessions", base_config: Optional[RAGConfig] = None
    ):
        """Initialize session manager.

        Args:
            base_dir: Base directory for session storage
            base_config: Base configuration to use for new sessions
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.base_config = base_config
        self._sessions: Dict[str, RAGSession] = {}

    def get_or_create(
        self,
        session_id: str,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        config_override: Optional[RAGConfig] = None,
        vector_store_type: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
        data_source_mode: Optional[str] = None,
    ) -> RAGSession:
        """Get existing session or create a new one.

        Args:
            session_id: Unique session identifier
            llm_model: Optional LLM model override
            config_override: Optional RAGConfig that takes priority
                over the manager's base_config for this session
            vector_store_type: Optional vector store provider override
            retrieval_mode: Optional retrieval mode ("semantic", "bm25", "hybrid")
            data_source_mode: Data source mode ("documents", "sql", "auto")

        Returns:
            RAGSession instance
        """
        if session_id in self._sessions:
            return self._sessions[session_id]

        session = self._create_session(
            session_id, llm_model, embedding_model, config_override,
            vector_store_type, retrieval_mode, data_source_mode,
        )
        self._sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")

        return session

    def _create_session(
        self,
        session_id: str,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        config_override: Optional[RAGConfig] = None,
        vector_store_type: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
        data_source_mode: Optional[str] = None,
    ) -> RAGSession:
        """Create a new RAG session."""
        session_dir = self.base_dir / session_id
        documents_dir = session_dir / "documents"
        vector_store_dir = session_dir / "vector_store"

        # Use config_override if provided, otherwise fall back to base_config
        effective_base = config_override if config_override else self.base_config

        # Build configuration for this session
        config = build_session_config(
            base_config=effective_base,
            documents_dir=documents_dir,
            vector_store_dir=vector_store_dir,
            llm_model=llm_model,
            embedding_model=embedding_model,
            vector_store_type=vector_store_type,
            retrieval_mode=retrieval_mode,
        )

        rag = RAGFramework(config)

        return RAGSession(
            session_id=session_id,
            rag=rag,
            documents_dir=documents_dir,
            vector_store_dir=vector_store_dir,
            default_llm_model=config.llm.model,
            default_embedding_model=config.embedding.model,
            default_vector_store_provider=config.vector_store.provider,
            default_prompt_template=config.prompt_template,
            default_sql_path=config.sql.connection.sqlite_path,
            default_sql_enabled=config.sql.enabled,
            data_source_mode=data_source_mode or "documents",
        )

    def clear(self, session_id: str) -> bool:
        """Clear a session and its data.

        Args:
            session_id: Session to clear

        Returns:
            True if session was cleared
        """
        if session_id not in self._sessions:
            # Check if directory exists from previous run
            session_dir = self.base_dir / session_id
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)
                logger.info(f"Cleared orphaned session directory: {session_id}")
                return True
            return False

        session = self._sessions[session_id]

        # Clear the RAG index
        try:
            session.rag.clear_index()
        except Exception as e:
            logger.warning(f"Error clearing index: {e}")

        # Remove session directories
        session_dir = self.base_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)

        del self._sessions[session_id]
        logger.info(f"Cleared session: {session_id}")
        return True

    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        if session_id in self._sessions:
            return True
        return (self.base_dir / session_id).exists()

    def get(self, session_id: str) -> Optional[RAGSession]:
        """Get session if it exists in memory."""
        return self._sessions.get(session_id)

    @property
    def session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)
