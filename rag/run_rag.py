#!/usr/bin/env python3
"""
TFG Chatbot - Unified Entry Point
============================================================

Single entry point for CLI, API and model-download modes.

Usage:
    python run_rag.py cli                                    # Interactive CLI (default)
    python run_rag.py api                                    # Start API server
    python run_rag.py api --port 8080                        # API on custom port
    python run_rag.py download llm tiny-llama                # Download a LLM
    python run_rag.py download embedding all-MiniLM-L6-v2    # Download an embedding model
    python run_rag.py download --list-popular                # List popular models
"""

import os
import sys
import argparse

# Necessary to ensure the script can be run from any location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rag_framework.utils.logging import setup_logging
from rag_framework.utils.constants import (
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    FRAMEWORK_VERSION,
)


def run_cli() -> None:
    """Run the interactive CLI mode."""
    from rag_framework import RAGFramework
    from cli.ui import print_banner, print_success, print_error, print_info
    from cli.menu import show_startup_menu, interactive_menu

    print_banner()
    config = show_startup_menu()

    print_info("Inicializando framework RAG...")
    try:
        rag = RAGFramework(config)
    except Exception as e:
        print_error(f"Error al inicializar el framework: {e}")
        print_info("Intenta revisar la configuración o los modelos disponibles.")
        sys.exit(1)

    print_success("Framework inicializado correctamente\n")

    try:
        interactive_menu(rag)
    except KeyboardInterrupt:  # Friendly exit on Ctrl+C
        print(f"\n\n¡Hasta pronto!\n")
        sys.exit(0)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        sys.exit(1)


def run_api(args: argparse.Namespace) -> None:
    """Run the API server mode."""
    from api.server import run_server

    run_server(args.host, args.port, args.config)


def run_download(args: argparse.Namespace) -> None:
    """Run the model download mode."""
    from rag_framework.utils.model_downloader import run_download as _run_download

    _run_download(args)


def main() -> None:
    """Unified entry point for TFG Chatbot."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description=f"TFG Chatbot v{FRAMEWORK_VERSION} - RAG Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
            Examples:
                python run_rag.py                                         # Start CLI (default)
                python run_rag.py cli                                     # Start CLI explicitly
                python run_rag.py api                                     # Start API server
                python run_rag.py api --port 8080                         # API on custom port
                python run_rag.py api --config cfg.yaml                   # API with custom config
                python run_rag.py download llm tiny-llama                 # Download a LLM
                python run_rag.py download embedding all-MiniLM-L6-v2    # Download embedding
                python run_rag.py download reranker bge-reranker-base    # Download reranker
                python run_rag.py download --list-popular                # List popular models
                    """,
    )
    subparsers = parser.add_subparsers(dest="mode")

    # --- CLI subcommand ---
    subparsers.add_parser(
        "cli",
        help="Start the interactive CLI (default)",
        description="Interactive command-line interface for the RAG system.",
    )

    # --- API subcommand ---
    api_parser = subparsers.add_parser(
        "api",
        help="Start the REST API server",
        description="HTTP REST API server for the RAG system.",
    )
    api_parser.add_argument(
        "--host",
        default=DEFAULT_API_HOST,
        help=f"Host to bind to (default: {DEFAULT_API_HOST})",
    )
    api_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=DEFAULT_API_PORT,
        help=f"Port to listen on (default: {DEFAULT_API_PORT})",
    )
    api_parser.add_argument(
        "--config",
        "-c",
        help="Path to configuration YAML file",
    )

    # --- Download subcommand ---
    dl_parser = subparsers.add_parser(
        "download",
        help="Download HuggingFace models for offline use",
        description="Download and cache HuggingFace models (LLMs, embeddings, rerankers).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
            Examples:
            python run_rag.py download llm tiny-llama
            python run_rag.py download embedding all-MiniLM-L6-v2
            python run_rag.py download reranker bge-reranker-base
            python run_rag.py download llm meta-llama/Llama-2-7b-chat-hf
            python run_rag.py download llm tiny-llama --output ./my_models/llama
            python run_rag.py download llm meta-llama/Llama-2-7b-chat-hf --token YOUR_HF_TOKEN
            python run_rag.py download --list-popular
        """,
    )
    dl_parser.add_argument(
        "model_type",
        nargs="?",
        choices=["llm", "embedding", "reranker"],
        help="Type of model to download",
    )
    dl_parser.add_argument(
        "model_id",
        nargs="?",
        help="HuggingFace model ID or shortcut name",
    )
    dl_parser.add_argument(
        "--output",
        "-o",
        help="Custom output directory for the model",
    )
    dl_parser.add_argument(
        "--token",
        "-t",
        help="HuggingFace token for private/gated models",
    )
    dl_parser.add_argument(
        "--list-popular",
        "-l",
        action="store_true",
        help="List popular models for each category",
    )

    args = parser.parse_args()

    if args.mode == "api":
        run_api(args)
    elif args.mode == "download":
        run_download(args)
    else:
        # Default to CLI if no subcommand is given
        run_cli()


if __name__ == "__main__":
    main()
