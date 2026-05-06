"""Unit tests for api.handlers module.

Tests the /configs and /sessions HTTP endpoints.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from api.handlers import RAGAPIHandler


class FakeRequest:
    def __init__(self, body: dict = None):
        self.body = json.dumps(body or {}).encode("utf-8")

    def read(self, n):
        data = self.body[:n]
        self.body = self.body[n:]
        return data


def _make_handler(session_manager=None):
    handler = object.__new__(RAGAPIHandler)
    handler.session_manager = session_manager or MagicMock()
    handler.base_config = None
    handler._responses = []

    def mock_send_json(data, status_code=200):
        handler._responses.append({"data": data, "status": status_code})

    def mock_error(message, status_code=400):
        handler._responses.append({"data": {"error": message, "success": False}, "status": status_code})

    handler._send_json = mock_send_json
    handler._error = mock_error
    return handler


# ---------------------------------------------------------------------------
# GET /configs
# ---------------------------------------------------------------------------


class TestHandleListConfigs:
    @patch("api.handlers.discover_config_files")
    def test_list_configs_returns_files_with_normalized_paths(self, mock_discover):
        mock_discover.return_value = [
            "config/rag_config.yaml",
            "config\\local_models.yaml",
        ]
        handler = _make_handler()
        handler._handle_list_configs()

        resp = handler._responses[0]
        assert resp["status"] == 200
        configs = resp["data"]["configs"]
        assert configs[0]["name"] == "rag_config.yaml"
        assert configs[1]["path"] == "config/local_models.yaml"  # backslash normalized

    @patch("api.handlers.discover_config_files")
    def test_list_configs_empty(self, mock_discover):
        mock_discover.return_value = []
        handler = _make_handler()
        handler._handle_list_configs()

        assert handler._responses[0]["data"]["configs"] == []


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------


class TestHandleCreateSession:
    def test_requires_session_id(self):
        handler = _make_handler()
        handler._read_json = lambda: {}
        handler._handle_create_session()
        assert handler._responses[0]["status"] == 400
        assert "session_id" in handler._responses[0]["data"]["error"]

    def test_rejects_existing_session(self):
        mock_sm = MagicMock()
        mock_sm.exists.return_value = True
        handler = _make_handler(session_manager=mock_sm)
        handler._read_json = lambda: {"session_id": "existing"}
        handler._handle_create_session()
        assert handler._responses[0]["status"] == 409

    def test_create_with_default_config(self):
        mock_sm = MagicMock()
        mock_sm.exists.return_value = False
        mock_rag = MagicMock()
        mock_rag.config.llm.provider = "ollama"
        mock_rag.config.llm.model = "llama3"
        mock_rag.config.embedding.provider = "ollama"
        mock_rag.config.embedding.model = "nomic"
        mock_session = MagicMock()
        mock_session.rag = mock_rag
        mock_sm.get_or_create.return_value = mock_session

        handler = _make_handler(session_manager=mock_sm)
        handler._read_json = lambda: {"session_id": "new-session"}
        handler._handle_create_session()

        resp = handler._responses[0]
        assert resp["status"] == 201
        assert resp["data"]["success"] is True
        assert resp["data"]["session_id"] == "new-session"
        assert resp["data"]["config_source"] == "default"

    @patch("api.handlers.ConfigLoader")
    def test_create_with_config_file(self, mock_config_loader):
        mock_config_loader.load_from_yaml.return_value = MagicMock()
        mock_sm = MagicMock()
        mock_sm.exists.return_value = False
        mock_rag = MagicMock()
        mock_rag.config.llm.provider = "huggingface"
        mock_rag.config.llm.model = "gpt2"
        mock_rag.config.embedding.provider = "huggingface"
        mock_rag.config.embedding.model = "all-MiniLM"
        mock_session = MagicMock()
        mock_session.rag = mock_rag
        mock_sm.get_or_create.return_value = mock_session

        handler = _make_handler(session_manager=mock_sm)
        handler._read_json = lambda: {"session_id": "custom", "config_name": "config/hf.yaml"}
        handler._handle_create_session()

        resp = handler._responses[0]
        assert resp["status"] == 201
        assert resp["data"]["config_source"] == "config/hf.yaml"
        mock_config_loader.load_from_yaml.assert_called_once_with("config/hf.yaml")
