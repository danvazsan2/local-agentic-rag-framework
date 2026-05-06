"""HTTP request handlers for RAG API.

Implements the HTTP routing and request/response handling for all
RAG framework operations via REST API.
"""

import json
import base64
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

from rag_framework.utils.logging import get_logger
from rag_framework.utils.constants import FRAMEWORK_VERSION
from rag_framework.config import RAGConfig, ConfigLoader

from api.sessions import SessionManager
from cli.discovery import discover_config_files

logger = get_logger(__name__)


class RAGAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RAG API."""

    SUPPORTED_RUNTIME_VECTOR_STORES = {"lancedb", "chroma", "faiss"}

    # Class-level session manager (set by server)
    session_manager: SessionManager = None
    base_config: RAGConfig = None

    def _send_json(self, data: dict, status_code: int = 200):
        """Send JSON response with CORS headers."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _read_json(self) -> dict:
        """Read JSON from request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        return json.loads(body) if body else {}

    def _error(self, message: str, status_code: int = 400):
        """Send error response."""
        self._send_json({"error": message, "success": False}, status_code)

    @staticmethod
    def _normalize_optional_str(value):
        """Normalize optional string values from request payloads."""
        if value is None:
            return None
        if not isinstance(value, str):
            return str(value)
        normalized = value.strip()
        return normalized

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path

        handlers = {
            "/health": self._handle_health,
            "/config": self._handle_config,
            "/configs": self._handle_list_configs,
            "/templates": self._handle_list_templates,
            "/session-config": self._handle_session_config,
        }

        handler = handlers.get(path)
        if handler:
            try:
                handler()
            except Exception as e:
                logger.exception("Error handling GET request")
                self._error(str(e), 500)
        else:
            self._error("Not found", 404)

    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path

        handlers = {
            "/ingest": self._handle_ingest,
            "/query": self._handle_query,
            "/clear": self._handle_clear,
            "/sessions": self._handle_create_session,
            "/configure": self._handle_configure,
        }

        handler = handlers.get(path)
        if handler:
            try:
                handler()
            except Exception as e:
                logger.exception("Error handling POST request")
                self._error(str(e), 500)
        else:
            self._error("Not found", 404)

    # -------------------------------------------------------------------------
    # Endpoint Handlers
    # -------------------------------------------------------------------------

    def _handle_health(self):
        """Health check endpoint."""
        self._send_json(
            {
                "status": "ok",
                "version": FRAMEWORK_VERSION,
                "sessions": self.session_manager.session_count,
            }
        )

    def _handle_config(self):
        """Get current configuration."""
        from rag_framework.utils.constants import (
            DEFAULT_LLM_MODEL,
            DEFAULT_EMBEDDING_MODEL,
        )

        if self.base_config:
            config_info = {
                "llm": {
                    "provider": self.base_config.llm.provider,
                    "model": self.base_config.llm.model,
                },
                "embedding": {
                    "provider": self.base_config.embedding.provider,
                    "model": self.base_config.embedding.model,
                },
                "retrieval": {
                    "use_hybrid_search": self.base_config.retrieval.use_hybrid_search,
                    "top_k": self.base_config.retrieval.top_k,
                    "reranker_enabled": self.base_config.retrieval.reranker.enabled,
                },
            }
        else:
            config_info = {
                "llm": {"provider": "ollama", "model": DEFAULT_LLM_MODEL},
                "embedding": {"provider": "ollama", "model": DEFAULT_EMBEDDING_MODEL},
            }

        self._send_json({"config": config_info, "version": FRAMEWORK_VERSION})

    def _handle_ingest(self):
        """Handle file ingestion.

        Expected JSON body:
        {
            "session_id": "chat-uuid",
            "files": [
                {"name": "file.pdf", "content": "base64-encoded-content"},
                ...
            ],
            "llm_model": "optional-model-name"
        }
        """
        data = self._read_json()

        session_id = data.get("session_id")
        files = data.get("files", [])
        llm_model = data.get("llm_model")

        if not session_id:
            return self._error("session_id is required")

        if not files:
            return self._error("files array is required")

        # Get or create session
        session = self.session_manager.get_or_create(session_id, llm_model)

        # Save files to session documents directory
        saved_files = []
        for file_data in files:
            file_name = file_data.get("name")
            file_content = file_data.get("content")

            if not file_name or not file_content:
                continue

            try:
                content_bytes = base64.b64decode(file_content)
                file_path = session.documents_dir / file_name
                file_path.write_bytes(content_bytes)

                saved_files.append(file_name)
                session.ingested_files.append(file_name)
                logger.info(f"Saved file: {file_path}")
            except Exception as e:
                logger.error(f"Error saving file {file_name}: {e}")

        if not saved_files:
            return self._error("No files could be saved")

        # Ingest documents
        try:
            logger.info(f"Starting ingestion for session {session_id}")
            session.rag.ingest(force_reindex=True)

            if session.rag._index is None:
                return self._error("Ingestion completed but index was not created", 500)

            logger.info(f"Index created successfully for session {session_id}")

            self._send_json(
                {
                    "success": True,
                    "session_id": session_id,
                    "files_ingested": saved_files,
                    "total_files": len(session.ingested_files),
                }
            )

        except Exception as e:
            logger.exception("Error during ingestion")
            self._error(f"Ingestion failed: {str(e)}", 500)

    def _handle_query(self):
        """Handle RAG query.

        Expected JSON body:
        {
            "session_id": "chat-uuid",
            "query": "user question",
            "llm_model": "optional-model-name"
        }
        """
        data = self._read_json()

        session_id = data.get("session_id")
        query = data.get("query")
        llm_model = self._normalize_optional_str(data.get("llm_model"))

        if not session_id:
            return self._error("session_id is required")

        if not query:
            return self._error("query is required")

        # Get or recreate session
        if not self.session_manager.exists(session_id):
            return self._error("Session not found. Please ingest documents first.", 404)

        session = self.session_manager.get(session_id)

        if session is None:
            # Try to recreate from existing data
            session = self.session_manager.get_or_create(session_id, llm_model)
            try:
                session.rag.load_index()
                logger.info(f"Loaded existing index for session {session_id}")
            except Exception as e:
                logger.warning(f"Could not load index: {e}")
                return self._error(
                    "Session index not found. Please ingest documents first.", 404
                )

        # Query-time LLM overrides are only allowed before the session is locked.
        # This avoids changing engines mid-conversation and causing runtime instability.
        if llm_model and session.rag.config.llm.model != llm_model:
            if session.configuration_locked:
                return self._error(
                    "Configuration is locked for this session after starting the conversation. "
                    "Create/clear the session to apply a different LLM.",
                    409,
                )
            session.rag.config.llm.model = llm_model
            session.rag._query_engine = None
            logger.info(
                "LLM model set to %s for session %s before lock", llm_model, session_id
            )

        if not session.first_query_executed:
            session.first_query_executed = True
            session.configuration_locked = True
            logger.info("Session %s configuration locked after first query", session_id)

        # Perform query
        try:
            logger.info(f"Querying session {session_id} [{session.data_source_mode}]: {query[:50]}...")

            if session.data_source_mode == "sql":
                response = session.rag.query_sql(query)
            elif session.data_source_mode == "auto":
                # Router enabled — query() delegates to the router
                response = session.rag.query(query)
            else:
                response = session.rag.query(query)

            self._send_json(
                {
                    "success": True,
                    "session_id": session_id,
                    "query": query,
                    "response": response,
                    "model": session.rag.config.llm.model,
                }
            )

        except Exception as e:
            logger.exception("Error during query")
            self._error(f"Query failed: {str(e)}", 500)

    def _handle_clear(self):
        """Clear a session.

        Expected JSON body:
        {
            "session_id": "chat-uuid"
        }
        """
        data = self._read_json()
        session_id = data.get("session_id")

        if not session_id:
            return self._error("session_id is required")

        success = self.session_manager.clear(session_id)

        self._send_json(
            {
                "success": success,
                "session_id": session_id,
                "message": "Session cleared" if success else "Session not found",
            }
        )

    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info("%s - %s", self.address_string(), format % args)

    # -------------------------------------------------------------------------
    # Configuration Endpoints
    # -------------------------------------------------------------------------

    def _handle_list_configs(self):
        """List available configuration files.

        Returns a list of YAML configuration files found in the project.
        """
        try:
            config_files = discover_config_files()
            configs = []
            for path in config_files:
                configs.append(
                    {
                        "name": path.replace("\\", "/").split("/")[-1],
                        "path": path.replace("\\", "/"),
                    }
                )

            self._send_json({"configs": configs})
        except Exception as e:
            logger.exception("Error listing configurations")
            self._error(f"Failed to list configurations: {str(e)}", 500)

    def _handle_configure(self):
        """Update runtime configuration for an existing session.

        Expected JSON body:
        {
            "session_id": "chat-uuid",
            "llm_model": "optional-model-name",
            "custom_prompt": "optional-prompt-template",
            "prompt_template": "template-name",
            "sql_path": "path/to/db.sqlite",
            "embedding_model": "model-name",
            "vector_store_type": "lancedb"
        }
        """
        data = self._read_json()
        session_id = data.get("session_id")

        if not session_id:
            return self._error("session_id is required")

        if not self.session_manager.exists(session_id):
            return self._error("Session not found", 404)

        session = self.session_manager.get(session_id)
        if session is None:
            return self._error("Session not in memory", 404)

        if session.configuration_locked:
            return self._error(
                "Configuration is locked for this session because the conversation already started. "
                "Clear/create a new session to change RAG settings.",
                409,
            )

        requires_reindex = False

        llm_model = self._normalize_optional_str(data.get("llm_model"))
        custom_prompt = data.get("custom_prompt")
        prompt_template = self._normalize_optional_str(data.get("prompt_template"))
        sql_path = self._normalize_optional_str(data.get("sql_path"))
        embedding_model = self._normalize_optional_str(data.get("embedding_model"))
        vector_store_type = self._normalize_optional_str(data.get("vector_store_type"))

        if llm_model is not None:
            resolved_llm_model = llm_model or session.default_llm_model
            session.rag.config.llm.model = resolved_llm_model
            session.rag._query_engine = None
            logger.info(
                "LLM model configured to %s for session %s",
                resolved_llm_model,
                session_id,
            )

        if prompt_template is not None:
            if prompt_template == "":
                session.rag._config_ops.set_prompt_template(
                    session.default_prompt_template
                )
                logger.info(
                    "Prompt template reset to default (%s) for session %s",
                    session.default_prompt_template,
                    session_id,
                )
            elif prompt_template != "custom":
                from rag_framework.prompts.templates import PromptTemplates

                available_templates = PromptTemplates.list_templates()
                if prompt_template not in available_templates:
                    return self._error(
                        f"Unknown prompt template '{prompt_template}'.",
                        400,
                    )
                session.rag._config_ops.set_prompt_template(prompt_template)
                logger.info(
                    "Prompt template configured to %s for session %s",
                    prompt_template,
                    session_id,
                )

        if custom_prompt is not None:
            normalized_custom_prompt = (
                custom_prompt.strip()
                if isinstance(custom_prompt, str)
                else str(custom_prompt)
            )

            if normalized_custom_prompt == "":
                session.rag.config.custom_prompt = None
                session.rag._config_ops.set_prompt_template(
                    session.default_prompt_template
                )
                logger.info("Custom prompt cleared for session %s", session_id)
            else:
                missing_vars = []
                if "{context_str}" not in normalized_custom_prompt:
                    missing_vars.append("{context_str}")
                if "{query_str}" not in normalized_custom_prompt:
                    missing_vars.append("{query_str}")
                if missing_vars:
                    return self._error(
                        "Custom prompt must include placeholders: "
                        + ", ".join(missing_vars),
                        400,
                    )

                session.rag._config_ops.set_custom_prompt(normalized_custom_prompt)
                logger.info("Custom prompt configured for session %s", session_id)

        if sql_path is not None:
            resolved_sql_path = sql_path or session.default_sql_path
            session.rag.config.sql.connection.sqlite_path = resolved_sql_path
            session.rag.config.sql.enabled = (
                bool(resolved_sql_path)
                if sql_path != ""
                else session.default_sql_enabled
            )
            session.rag._sql_agent = None
            session.rag._router = None
            logger.info(
                "SQL path configured to '%s' for session %s",
                resolved_sql_path,
                session_id,
            )

        if embedding_model is not None:
            resolved_embedding_model = (
                embedding_model or session.default_embedding_model
            )
            session.rag.config.embedding.model = resolved_embedding_model
            session.rag._query_engine = None
            session.rag._index = None
            session.rag._nodes = None
            requires_reindex = True
            logger.info(
                "Embedding model configured to %s for session %s",
                resolved_embedding_model,
                session_id,
            )

        if vector_store_type is not None:
            resolved_vector_store = (
                vector_store_type or session.default_vector_store_provider
            )
            if resolved_vector_store not in self.SUPPORTED_RUNTIME_VECTOR_STORES:
                return self._error(
                    "Unsupported vector store for runtime session: "
                    f"'{resolved_vector_store}'. Supported: "
                    f"{', '.join(sorted(self.SUPPORTED_RUNTIME_VECTOR_STORES))}",
                    400,
                )

            session.rag.config.vector_store.provider = resolved_vector_store
            session.rag._query_engine = None
            session.rag._index = None
            session.rag._nodes = None
            requires_reindex = True
            logger.info(
                "Vector store type configured to %s for session %s",
                resolved_vector_store,
                session_id,
            )

        cfg = session.rag.config
        self._send_json(
            {
                "success": True,
                "session_id": session_id,
                "requires_reindex": requires_reindex,
                "configuration_locked": session.configuration_locked,
                "config": {
                    "llm": {"provider": cfg.llm.provider, "model": cfg.llm.model},
                    "embedding": {
                        "provider": cfg.embedding.provider,
                        "model": cfg.embedding.model,
                    },
                    "prompt_template": getattr(cfg, "prompt_template", "default"),
                    "custom_prompt": getattr(cfg, "custom_prompt", None),
                    "vector_store_type": cfg.vector_store.provider,
                    "sql_enabled": cfg.sql.enabled,
                    "sql_path": cfg.sql.connection.sqlite_path,
                },
            }
        )

    def _handle_list_templates(self):
        """List available prompt templates.

        Returns all registered prompt templates with their names and descriptions.
        """
        try:
            from rag_framework.prompts.templates import PromptTemplates

            templates_dict = PromptTemplates.list_templates()
            templates = [
                {"name": k, "description": v} for k, v in templates_dict.items()
            ]
            self._send_json({"templates": templates})
        except Exception as e:
            logger.exception("Error listing templates")
            self._error(f"Failed to list templates: {str(e)}", 500)

    def _handle_session_config(self):
        """Get per-session configuration.

        Query params:
            session_id: The session ID to retrieve config for.
        """
        from urllib.parse import parse_qs

        query_string = urlparse(self.path).query
        params = parse_qs(query_string)
        session_id = params.get("session_id", [None])[0]

        if not session_id:
            return self._error("session_id is required")

        if not self.session_manager.exists(session_id):
            return self._error("Session not found", 404)

        session = self.session_manager.get(session_id)
        if session is None:
            return self._error("Session not in memory", 404)

        cfg = session.rag.config
        self._send_json(
            {
                "session_id": session_id,
                "llm_model": cfg.llm.model,
                "embedding_model": cfg.embedding.model,
                "vector_store_provider": cfg.vector_store.provider,
                "prompt_template": getattr(cfg, "prompt_template", "default"),
                "custom_prompt": getattr(cfg, "custom_prompt", None),
                "sql_enabled": cfg.sql.enabled,
                "sql_path": cfg.sql.connection.sqlite_path,
                "first_query_executed": session.first_query_executed,
                "configuration_locked": session.configuration_locked,
            }
        )

    def _handle_create_session(self):
        """Create a new session with optional specific configuration.

        Expected JSON body:
        {
            "session_id": "my-session",
            "config_name": "config/huggingface.yaml",  // optional
            "llm_model": "model-name"                   // optional
        }
        """
        data = self._read_json()

        session_id = data.get("session_id")
        config_name = data.get("config_name")
        llm_model = self._normalize_optional_str(data.get("llm_model"))
        embedding_model = self._normalize_optional_str(data.get("embedding_model"))
        prompt_template = self._normalize_optional_str(data.get("prompt_template"))
        sql_path = self._normalize_optional_str(data.get("sql_path"))
        custom_prompt = data.get("custom_prompt")
        vector_store = self._normalize_optional_str(data.get("vector_store"))
        retrieval_mode = self._normalize_optional_str(data.get("retrieval_mode"))
        data_source_mode = self._normalize_optional_str(data.get("data_source_mode"))

        if not session_id:
            return self._error("session_id is required")

        # Check if session already exists
        if self.session_manager.exists(session_id):
            return self._error(
                f"Session '{session_id}' already exists. Use /clear first.", 409
            )

        # Load specific configuration if requested
        config_override = None
        if config_name:
            try:
                config_override = ConfigLoader.load_from_yaml(config_name)
                logger.info(f"Loaded config '{config_name}' for session {session_id}")
            except FileNotFoundError:
                return self._error(f"Configuration file not found: {config_name}", 404)
            except Exception as e:
                return self._error(f"Error loading configuration: {str(e)}", 400)

        # Create the session with all model overrides applied before RAGFramework init
        try:
            session = self.session_manager.get_or_create(
                session_id,
                llm_model=llm_model,
                embedding_model=embedding_model,
                config_override=config_override,
                vector_store_type=vector_store,
                retrieval_mode=retrieval_mode,
                data_source_mode=data_source_mode,
            )

            if prompt_template and prompt_template != "default":
                from rag_framework.prompts.templates import PromptTemplates

                if prompt_template in PromptTemplates.list_templates():
                    session.rag._config_ops.set_prompt_template(prompt_template)
                    logger.info(
                        "Prompt template set to %s for session %s",
                        prompt_template,
                        session_id,
                    )

            if custom_prompt and isinstance(custom_prompt, str):
                normalized = custom_prompt.strip()
                if (
                    normalized
                    and "{context_str}" in normalized
                    and "{query_str}" in normalized
                ):
                    session.rag._config_ops.set_custom_prompt(normalized)
                    logger.info("Custom prompt set for session %s", session_id)

            if sql_path:
                session.rag.config.sql.connection.sqlite_path = sql_path
                session.rag.config.sql.enabled = True
                logger.info(
                    "SQL path set to %s for session %s", sql_path, session_id
                )

            # Apply data source routing config
            resolved_data_source = data_source_mode or "documents"
            if resolved_data_source == "sql":
                session.rag.config.sql.enabled = True
            elif resolved_data_source == "auto":
                session.rag.config.sql.enabled = True
                session.rag.config.router.enabled = True
                session.rag.config.router.default_source = "unstructured"
                session.rag._router = None
            elif resolved_data_source == "documents":
                session.rag.config.router.enabled = False

            # Write effective config YAML for reproducibility
            try:
                from api.config_builder import write_session_yaml

                sessions_config_dir = self.session_manager.base_dir / "session_configs"
                write_session_yaml(session_id, session.rag.config, sessions_config_dir)
            except Exception as yaml_err:
                logger.warning("Failed to write session YAML: %s", yaml_err)

            # Build response config info
            cfg = session.rag.config
            config_info = {
                "llm": {
                    "provider": cfg.llm.provider,
                    "model": cfg.llm.model,
                },
                "embedding": {
                    "provider": cfg.embedding.provider,
                    "model": cfg.embedding.model,
                },
                "prompt_template": cfg.prompt_template,
                "sql_enabled": cfg.sql.enabled,
            }

            self._send_json(
                {
                    "success": True,
                    "session_id": session_id,
                    "config": config_info,
                    "config_source": config_name or "default",
                },
                201,
            )

        except Exception as e:
            logger.exception("Error creating session")
            self._error(f"Failed to create session: {str(e)}", 500)
