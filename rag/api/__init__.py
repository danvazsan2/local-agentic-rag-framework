"""HTTP API server for Custom-RAG Framework.

This package provides a REST API for interacting with the RAG framework
without the CLI interface. Supports session management and concurrent requests.
"""

from api.server import run_server

__all__ = ["run_server"]
