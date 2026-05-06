"""
Output Formatters
==================

Console output functions for the CLI.
"""

import pyfiglet


def print_banner() -> None:
    """Display the TFG Chatbot banner using pyfiglet."""
    try:
        banner = pyfiglet.figlet_format("TFG Chatbot", font="slant")
    except pyfiglet.FontNotFound:
        banner = "TFG CHATBOT\n"

    print(f"\n{banner}")
    print(f"{'═' * 60}")
    print(f"  Sistema RAG - Retrieval-Augmented Generation")
    print(f"  Autor: Daniel Vázquez Sánchez")
    print(f"{'═' * 60}\n")


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'─' * 50}")
    print(f"  {text}")
    print(f"{'─' * 50}\n")


def _print_message(text: str, prefix: str) -> None:
    """Helper to print messages with a prefix label."""
    print(f"{prefix} {text}")


def print_success(text: str) -> None:
    """Print a success message."""
    _print_message(text, "[OK]")


def print_error(text: str) -> None:
    """Print an error message."""
    _print_message(text, "[ERROR]")


def print_warning(text: str) -> None:
    """Print a warning message."""
    _print_message(text, "[AVISO]")


def print_info(text: str) -> None:
    """Print an info message."""
    _print_message(text, "[INFO]")
