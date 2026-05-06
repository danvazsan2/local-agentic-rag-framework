"""
RAG Framework - Console Output Utilities.

Provides console I/O utilities and helpers for interactive features.
"""

from typing import Optional


class ConsoleOutput:
    """
    Handles console input/output operations.

    This class provides utilities for interactive console operations,
    input validation, and formatted output management.
    """

    @staticmethod
    def get_user_input(prompt: str = "Tú: ", prefix: str = "") -> Optional[str]:
        """
        Get user input with a formatted prompt.

        Args:
            prompt: The prompt message to display.
            prefix: Optional prefix before the prompt.

        Returns:
            User input string stripped of whitespace, or None if empty.
        """
        try:
            user_input = input(f"\n{prefix}{prompt}").strip()
            return user_input if user_input else None
        except EOFError:
            return None
        except KeyboardInterrupt:
            raise

    @staticmethod
    def confirm_action(message: str, default: bool = False) -> bool:
        """
        Ask user for yes/no confirmation.

        Args:
            message: The confirmation message.
            default: Default value if user just presses Enter.

        Returns:
            True if user confirms, False otherwise.
        """
        default_str = "Y/n" if default else "y/N"
        response = input(f"{message} [{default_str}]: ").strip().lower()

        if not response:
            return default

        return response in ["y", "yes", "si", "sí"]

    @staticmethod
    def clear_screen() -> None:
        """Clear the console screen (platform-independent)."""
        import os

        os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def print_separator(char: str = "=", length: int = 60) -> None:
        """
        Print a separator line.

        Args:
            char: Character to use for the separator.
            length: Length of the separator line.
        """
        print(char * length)
