"""
RAG Framework - Chat Interface.

Provides interactive chat functionality for the RAG system,
including command processing and conversation management.
"""

import logging
from typing import Optional, Callable
from rag_framework.display.formatters import DisplayFormatter
from rag_framework.display.console import ConsoleOutput

logger = logging.getLogger(__name__)


class ChatInterface:
    """
    Interactive chat interface for the RAG system.

    This class handles all chat-related functionality including:
    - Interactive conversation loop
    - Command processing (help, config, templates, exit)
    - User input handling
    - Response formatting

    The interface is designed to be framework-agnostic and can be
    used with different query engines or backends.
    """

    # Exit commands recognized by the chat interface
    EXIT_COMMANDS = {"exit", "quit", "q", "salir"}

    def __init__(
        self,
        query_callback: Callable[[str], str],
        config_callback: Optional[Callable[[], None]] = None,
        templates_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the chat interface.

        Args:
            query_callback: Function to call for processing queries.
                           Should accept a question string and return a response string.
            config_callback: Optional function to display current configuration.
            templates_callback: Optional function to display available templates.
        """
        self.query_callback = query_callback
        self.config_callback = config_callback
        self.templates_callback = templates_callback

    def start_interactive_session(self) -> None:
        """
        Start an interactive chat session.

        Runs a continuous loop accepting user input and processing
        queries until the user exits with an exit command.
        """
        DisplayFormatter.print_chat_header()

        while True:
            try:
                if not self._handle_interaction():
                    break
            except KeyboardInterrupt:
                print("\nSesión finalizada.")
                break
            except Exception as e:
                logger.error(f"Chat error: {e}")
                DisplayFormatter.print_error(str(e))

    def process_single_message(self, message: str) -> str:
        """
        Process a single message programmatically (non-interactive).

        Args:
            message: The message to process.

        Returns:
            Generated response string.
        """
        try:
            return self.query_callback(message)
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            return f"Error processing message: {e}"

    def _handle_interaction(self) -> bool:
        """
        Handle a single chat interaction.

        Returns:
            True to continue the chat loop, False to exit.
        """
        question = ConsoleOutput.get_user_input()

        if not question:
            print("Por favor, introduce una pregunta.")
            return True

        # Check if it's a command
        command_result = self._process_command(question.lower())
        if command_result is not None:
            return command_result

        # Process as a regular query
        self._process_and_display_response(question)
        return True

    def _process_command(self, command: str) -> Optional[bool]:
        """
        Process chat commands.

        Args:
            command: The command string (lowercase).

        Returns:
            True to continue chat, False to exit, None if not a command.
        """
        # Exit command
        if command in self.EXIT_COMMANDS:
            print("Sesión de chat finalizada.")
            return False

        # Help command
        if command == "help":
            DisplayFormatter.print_chat_help()
            return True

        # Config command
        if command == "config":
            if self.config_callback:
                self.config_callback()
            else:
                DisplayFormatter.print_warning(
                    "Visualización de configuración no disponible"
                )
            return True

        # Templates command
        if command == "templates":
            if self.templates_callback:
                self.templates_callback()
            else:
                DisplayFormatter.print_warning(
                    "Visualización de plantillas no disponible"
                )
            return True

        # Not a recognized command
        return None

    def _process_and_display_response(self, question: str) -> None:
        """
        Process a question and display the response.

        Args:
            question: The question to process.
        """
        try:
            response = self.query_callback(question)
            DisplayFormatter.print_response(response)
        except Exception as e:
            logger.error(f"Query failed in chat: {e}")
            DisplayFormatter.print_error(str(e))
