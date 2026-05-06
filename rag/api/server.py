"""HTTP server setup for RAG API.

Configures and runs the HTTP server with session management
and provides graceful shutdown.
"""

from http.server import HTTPServer

from rag_framework.utils.logging import setup_logging, get_logger
from rag_framework.utils.constants import (
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_SESSIONS_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    FRAMEWORK_VERSION,
)

setup_logging()
logger = get_logger(__name__)

from api.handlers import RAGAPIHandler
from api.sessions import SessionManager


def run_server(
    host: str = DEFAULT_API_HOST,
    port: int = DEFAULT_API_PORT,
    config_path: str = None,
):
    """Run the RAG API server.

    Args:
        host: Host to bind to
        port: Port to listen on
        config_path: Optional path to configuration file
    """
    # Load base configuration
    base_config = None
    if config_path:
        from rag_framework.config import ConfigLoader

        base_config = ConfigLoader.load_from_yaml(config_path)
        logger.info(f"Loaded configuration from: {config_path}")
    else:
        try:
            from rag_framework.config import ConfigLoader

            base_config = ConfigLoader.load()
        except Exception:
            pass

    # Initialize session manager
    session_manager = SessionManager(
        base_dir=DEFAULT_SESSIONS_DIR,
        base_config=base_config,
    )

    # Set class-level attributes for handler
    RAGAPIHandler.session_manager = session_manager
    RAGAPIHandler.base_config = base_config

    # Create and start server
    server = HTTPServer((host, port), RAGAPIHandler)

    print("=" * 60)
    print(f"🚀 RAG API Server v{FRAMEWORK_VERSION}")
    print("=" * 60)
    print(f"   Listening on http://{host}:{port}")
    print()
    print("   Endpoints:")
    print("     POST /ingest    - Ingest files for a session")
    print("     POST /query     - Query the RAG system")
    print("     POST /clear     - Clear a session")
    print("     POST /sessions  - Create session with specific config")
    print("     GET  /health    - Health check")
    print("     GET  /config    - Get current configuration")
    print("     GET  /configs   - List available configurations")
    print()
    if base_config:
        print(f"   LLM: {base_config.llm.provider}/{base_config.llm.model}")
        print(
            f"   Embedding: {base_config.embedding.provider}/{base_config.embedding.model}"
        )
    else:
        print(f"   Using defaults: {DEFAULT_LLM_MODEL}, {DEFAULT_EMBEDDING_MODEL}")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
