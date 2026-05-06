"""Unit tests for api.sessions module.

Tests session creation, lifecycle management, and config override behaviour.
"""

import pytest
from unittest.mock import MagicMock, patch

from api.sessions import RAGSession, SessionManager


def _make_session(tmp_path, session_id="test-123"):
    """Create a RAGSession with all required fields."""
    return RAGSession(
        session_id=session_id,
        rag=MagicMock(),
        documents_dir=tmp_path / "documents",
        vector_store_dir=tmp_path / "vector_store",
        default_llm_model="llama3",
        default_embedding_model="nomic-embed-text",
        default_vector_store_provider="lancedb",
        default_prompt_template="default",
        default_sql_path="",
        default_sql_enabled=False,
    )


class TestRAGSession:
    def test_session_creation_initializes_state(self, tmp_path):
        session = _make_session(tmp_path)
        assert session.session_id == "test-123"
        assert session.documents_dir.exists()
        assert session.vector_store_dir.exists()
        assert session.ingested_files == []

    def test_session_tracks_ingested_files(self, tmp_path):
        session = _make_session(tmp_path, session_id="test-456")
        session.ingested_files.append("doc1.pdf")
        session.ingested_files.append("doc2.txt")
        assert len(session.ingested_files) == 2
        assert "doc1.pdf" in session.ingested_files


class TestSessionManager:
    @patch("api.sessions.build_session_config")
    @patch("api.sessions.RAGFramework")
    def test_get_or_create_new_session(self, mock_framework_cls, mock_build_config, tmp_path):
        mock_build_config.return_value = MagicMock()
        mock_framework_cls.return_value = MagicMock()

        manager = SessionManager(base_dir=str(tmp_path))
        session = manager.get_or_create("session-001")

        assert session.session_id == "session-001"
        assert manager.session_count == 1
        mock_build_config.assert_called_once()

    @patch("api.sessions.build_session_config")
    @patch("api.sessions.RAGFramework")
    def test_get_or_create_returns_existing_session(self, mock_framework_cls, mock_build_config, tmp_path):
        mock_build_config.return_value = MagicMock()
        mock_framework_cls.return_value = MagicMock()

        manager = SessionManager(base_dir=str(tmp_path))
        session1 = manager.get_or_create("session-001")
        session2 = manager.get_or_create("session-001")

        assert session1 is session2
        assert manager.session_count == 1
        mock_build_config.assert_called_once()

    @patch("api.sessions.build_session_config")
    @patch("api.sessions.RAGFramework")
    def test_session_clear_removes_session(self, mock_framework_cls, mock_build_config, tmp_path):
        mock_build_config.return_value = MagicMock()
        mock_framework_cls.return_value = MagicMock()

        manager = SessionManager(base_dir=str(tmp_path))
        manager.get_or_create("session-to-clear")

        result = manager.clear("session-to-clear")

        assert result is True
        assert manager.session_count == 0
        assert manager.get("session-to-clear") is None

    def test_clear_nonexistent_session_returns_false(self, tmp_path):
        manager = SessionManager(base_dir=str(tmp_path))
        assert manager.clear("nonexistent-session") is False

    def test_clear_orphaned_directory(self, tmp_path):
        orphan_dir = tmp_path / "orphaned-session"
        orphan_dir.mkdir()

        manager = SessionManager(base_dir=str(tmp_path))
        result = manager.clear("orphaned-session")

        assert result is True
        assert not orphan_dir.exists()

    @patch("api.sessions.build_session_config")
    @patch("api.sessions.RAGFramework")
    def test_exists_for_in_memory_session(self, mock_framework_cls, mock_build_config, tmp_path):
        mock_build_config.return_value = MagicMock()
        mock_framework_cls.return_value = MagicMock()

        manager = SessionManager(base_dir=str(tmp_path))
        manager.get_or_create("existing-session")

        assert manager.exists("existing-session") is True
        assert manager.exists("nonexistent") is False

    @patch("api.sessions.build_session_config")
    @patch("api.sessions.RAGFramework")
    def test_config_override_takes_priority_over_base_config(
        self, mock_framework_cls, mock_build_config, tmp_path
    ):
        mock_base = MagicMock(name="base_config")
        mock_override = MagicMock(name="override_config")
        mock_build_config.return_value = MagicMock()
        mock_framework_cls.return_value = MagicMock()

        manager = SessionManager(base_dir=str(tmp_path), base_config=mock_base)
        manager.get_or_create("override-session", config_override=mock_override)

        call_kwargs = mock_build_config.call_args.kwargs
        assert call_kwargs["base_config"] == mock_override

    @patch("api.sessions.build_session_config")
    @patch("api.sessions.RAGFramework")
    def test_no_override_uses_base_config(self, mock_framework_cls, mock_build_config, tmp_path):
        mock_base = MagicMock(name="base_config")
        mock_build_config.return_value = MagicMock()
        mock_framework_cls.return_value = MagicMock()

        manager = SessionManager(base_dir=str(tmp_path), base_config=mock_base)
        manager.get_or_create("default-session")

        call_kwargs = mock_build_config.call_args.kwargs
        assert call_kwargs["base_config"] == mock_base
