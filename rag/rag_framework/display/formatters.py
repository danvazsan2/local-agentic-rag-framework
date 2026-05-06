"""
RAG Framework - Display Formatters.

Handles all display formatting and output for the RAG framework,
including configuration summaries, chunk displays, and help messages.
"""

from typing import List, Dict, Any, Optional
from rag_framework.utils.constants import FRAMEWORK_VERSION, FRAMEWORK_NAME


class DisplayFormatter:
    """
    Handles all console display formatting for the RAG framework.

    This class centralizes all display logic to maintain consistent
    output formatting and make it easier to modify or extend display
    capabilities (e.g., different output formats).
    """

    # Constants for display formatting
    SEPARATOR_FULL = "=" * 60
    SEPARATOR_PARTIAL = "-" * 40
    SEPARATOR_CHUNKS = "=" * 70
    SEPARATOR_CHUNK = "-" * 70
    CHUNK_PREVIEW_LENGTH = 300

    @staticmethod
    def print_config_summary(config) -> None:
        """
        Display a formatted summary of the current configuration.

        Args:
            config: RAGConfig instance containing framework configuration.
        """
        print(f"\n{DisplayFormatter.SEPARATOR_FULL}")
        print(f"{FRAMEWORK_NAME} v{FRAMEWORK_VERSION}")
        print(DisplayFormatter.SEPARATOR_FULL)

        config_items = [
            ("LLM Provider", config.llm.provider),
            ("LLM Model", config.llm.model),
            ("Embedding Provider", config.embedding.provider),
            ("Embedding Model", config.embedding.model),
            ("Vector Store", config.vector_store.provider),
            ("Hybrid Search", config.retrieval.use_hybrid_search),
            ("Reranker", config.retrieval.reranker.enabled),
        ]

        for label, value in config_items:
            print(f"  {label:<20} {value}")

        if config.retrieval.reranker.enabled:
            print(f"  {'Reranker Model':<20} {config.retrieval.reranker.model}")

        print(f"  {'Corrective RAG':<20} {config.corrective_rag.enabled}")
        print(f"  {'Prompt Template':<20} {config.prompt_template}")
        print(f"{DisplayFormatter.SEPARATOR_FULL}\n")

    @staticmethod
    def print_chat_help() -> None:
        """Display available chat commands and their descriptions."""
        help_text = """
Comandos disponibles:
   exit/quit/q - Salir del modo chat
   help        - Mostrar este mensaje de ayuda
   config      - Mostrar la configuración actual
   templates   - Listar plantillas de prompts disponibles
"""
        print(help_text)

    @staticmethod
    def print_available_templates(current_template: str) -> None:
        """
        Display all available prompt templates with current selection marked.

        Args:
            current_template: Name of the currently active template.
        """
        from rag_framework.prompts import PromptTemplates

        print("\nPlantillas de Prompts Disponibles:")
        templates = PromptTemplates.list_templates()

        for name, description in templates.items():
            marker = "✓" if name == current_template else " "
            truncated_desc = (
                description[:60] + "..." if len(description) > 60 else description
            )
            print(f"   [{marker}] {name}: {truncated_desc}")
        print()

    @staticmethod
    def print_chat_header() -> None:
        """Print the chat mode header."""
        print("\n" + "-" * 50)
        print("  Modo chat activo")
        print("  Escribe 'exit' para salir, 'help' para ayuda")
        print("-" * 50)

    @staticmethod
    def print_retrieved_chunks_header() -> None:
        """Print header for retrieved chunks display."""
        print(f"\n{DisplayFormatter.SEPARATOR_CHUNKS}")
        print("CHUNKS RECUPERADOS (ordenados por relevancia)")
        print(DisplayFormatter.SEPARATOR_CHUNKS)

    @staticmethod
    def print_chunk_details(nodes: List) -> None:
        """
        Print detailed information for each retrieved chunk.

        Args:
            nodes: List of retrieved nodes with score and metadata.
        """
        for i, node in enumerate(nodes, start=1):
            print(f"\nChunk {i}/{len(nodes)}:")

            # Display score
            score = getattr(node, "score", None)
            print(f"   Score: {score:.4f}" if score else "   Score: N/A")

            # Display source metadata
            DisplayFormatter._print_chunk_metadata(node)

            # Display text preview
            DisplayFormatter._print_chunk_text_preview(node)

            print(DisplayFormatter.SEPARATOR_CHUNK)

    @staticmethod
    def _print_chunk_metadata(node) -> None:
        """
        Print metadata information for a chunk.

        Args:
            node: The node containing metadata.
        """
        if hasattr(node, "node") and hasattr(node.node, "metadata"):
            metadata = node.node.metadata
            filename = metadata.get("file_name", metadata.get("source", "Unknown"))
            print(f"   Archivo: {filename}")

    @staticmethod
    def _print_chunk_text_preview(node) -> None:
        """
        Print a preview of the chunk's text content.

        Args:
            node: The node containing text.
        """
        text = node.node.text if hasattr(node, "node") else str(node)
        preview = text[: DisplayFormatter.CHUNK_PREVIEW_LENGTH].replace("\n", " ")
        print(f"   Texto: {preview}...")

    @staticmethod
    def print_index_load_status(
        has_nodes: bool, node_count: Optional[int] = None
    ) -> None:
        """
        Print the status of loaded index including hybrid search capability.

        Args:
            has_nodes: Whether nodes are available for hybrid search.
            node_count: Number of nodes loaded (if available).
        """
        if has_nodes and node_count:
            print(f"   Hybrid search: enabled ({node_count} nodes)")
        elif has_nodes:
            print("   Hybrid search: enabled")
        else:
            print("   Hybrid search: disabled (no nodes file)")

    @staticmethod
    def print_question(question: str) -> None:
        """
        Print a formatted question.

        Args:
            question: The question being asked.
        """
        print(f"\nPregunta: {question}")

    @staticmethod
    def print_response(response: str, prefix: str = "Asistente:") -> None:
        """
        Print a formatted response.

        Args:
            response: The response text.
            prefix: Prefix to show before the response.
        """
        print(f"\n{prefix}\n{response}\n")

    @staticmethod
    def print_error(message: str) -> None:
        """
        Print a formatted error message.

        Args:
            message: The error message.
        """
        print(f"\nError: {message}")

    @staticmethod
    def print_success(message: str) -> None:
        """
        Print a formatted success message.

        Args:
            message: The success message.
        """
        print(f"Éxito: {message}")

    @staticmethod
    def print_warning(message: str) -> None:
        """
        Print a formatted warning message.

        Args:
            message: The warning message.
        """
        print(f"Advertencia: {message}")

    @staticmethod
    def print_info(message: str) -> None:
        """
        Print a formatted info message.

        Args:
            message: The info message.
        """
        print(f"Info: {message}")
