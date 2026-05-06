"""
Menu Controller
================

Interactive menu loop and startup configuration.

This module provides the main control flow for the RAG system's interactive menu,
including startup configuration and the main menu loop.
"""

from enum import Enum
from typing import Optional, List, Callable, Dict

from rag_framework import RAGFramework
from rag_framework.config import RAGConfig, ConfigLoader

from ..ui import (
    get_input,
    get_choice,
    confirm,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
)
from ..discovery import discover_config_files
from ..wizards import (
    run_configuration_wizard,
    show_config_summary,
    save_config_to_yaml,
)
from ..handlers import (
    handle_ingest,
    handle_chat,
    handle_single_query,
    handle_validate_models,
    handle_list_templates,
    handle_show_config,
    handle_edit_config,
    handle_toggle_debug,
    handle_toggle_features,
    handle_save_config,
    handle_download_models,
    handle_start_api,
)
from .display import display_main_menu


# ============================================================================
# Constants
# ============================================================================

EXIT_COMMANDS = frozenset(["0", "exit", "quit", "q"])
"""Commands that trigger exit from the interactive menu."""

DEFAULT_CONFIG_FILENAME = "config/rag_config_custom.yaml"
"""Default filename for saving custom configurations."""


# ============================================================================
# Enumerations
# ============================================================================


class MenuOption(str, Enum):
    """Main menu options with their corresponding input values."""

    INGEST = "1"
    CHAT = "2"
    QUERY = "3"
    VALIDATE = "4"
    TEMPLATES = "5"
    SHOW_CONFIG = "6"
    EDIT_CONFIG = "7"
    SAVE_CONFIG = "8"
    FEATURES = "9"
    DOWNLOAD = "10"
    API = "11"


class StartupOption(int, Enum):
    """Startup configuration menu options."""

    DEFAULT = 1
    CUSTOM = 2
    FROM_FILE = 3


# ============================================================================
# Type Aliases
# ============================================================================

MenuHandler = Callable[[], None]
"""Type alias for menu handler functions."""

MenuHandlerMap = Dict[str, MenuHandler]
"""Type alias for the mapping of menu options to handlers."""


# ============================================================================
# Interactive Menu
# ============================================================================


def interactive_menu(rag: RAGFramework) -> None:
    """Main interactive menu loop.

    Displays the main menu and processes user selections until
    the user chooses to exit. Each menu option is handled by a
    dedicated handler function with centralized error handling.

    Args:
        rag: RAGFramework instance
    """
    handlers = _create_menu_handlers(rag)

    while True:
        display_main_menu(rag)
        choice = get_input("\nSelecciona opción (0-11)")

        if _is_exit_command(choice):
            print("\n¡Hasta pronto!\n")
            break

        handler = handlers.get(choice)
        if handler:
            _execute_menu_option(handler)
        else:
            print_error("Opción inválida. Por favor selecciona 0-11.")


def _create_menu_handlers(rag: RAGFramework) -> MenuHandlerMap:
    """Create the mapping of menu options to their handlers.

    This function centralizes the definition of menu handlers, making it
    easier to maintain and modify menu options.

    Args:
        rag: RAGFramework instance

    Returns:
        Dictionary mapping menu option strings to handler functions
    """
    return {
        MenuOption.INGEST: lambda: handle_ingest(rag),
        MenuOption.CHAT: lambda: handle_chat(rag),
        MenuOption.QUERY: lambda: handle_single_query(rag),
        MenuOption.VALIDATE: lambda: handle_validate_models(rag),
        MenuOption.TEMPLATES: lambda: handle_list_templates(),
        MenuOption.SHOW_CONFIG: lambda: handle_show_config(rag),
        MenuOption.EDIT_CONFIG: lambda: handle_edit_config(rag),
        MenuOption.SAVE_CONFIG: lambda: handle_save_config(rag),
        MenuOption.FEATURES: lambda: handle_toggle_features(rag),
        MenuOption.DOWNLOAD: lambda: handle_download_models(),
        MenuOption.API: lambda: handle_start_api(),
    }


def _execute_menu_option(handler: MenuHandler) -> None:
    """Execute a menu handler with centralized error handling.

    This function provides consistent error handling across all menu options,
    catching both user interruptions and unexpected errors.

    Args:
        handler: The menu handler function to execute
    """
    try:
        handler()
    except KeyboardInterrupt:
        print("\n")
        print_warning("Operación cancelada por el usuario")
    except Exception as e:
        print_error(f"Error: {e}")
        # In production, consider logging the full traceback for debugging


def _is_exit_command(choice: str) -> bool:
    """Check if the user input is an exit command.

    Args:
        choice: User input string

    Returns:
        True if the choice is an exit command, False otherwise
    """
    return choice.lower() in EXIT_COMMANDS


# ============================================================================
# Startup Menu
# ============================================================================


def show_startup_menu() -> Optional[RAGConfig]:
    """Show the startup configuration menu allowing the user to choose a config method.
    Loops until a valid configuration is selected.

    The function delegates to specialized helpers for each option:
    - Default: Returns None to use default config
    - Custom: Runs the wizard to let user create a custom config interactively
    - From file: Loads an existing configuration

    Returns:
        RAGConfig if custom/loaded, None for default config
    """
    print_header("Configuración Inicial")

    # Search for config files in current directory
    config_files = discover_config_files()
    # Display the config method the user has available and get their choice
    options = _display_startup_options(bool(config_files))
    choice = get_choice("\nSelecciona opción", options, 1)

    if choice == StartupOption.DEFAULT.value:  # Default config selected
        return _load_default_config()

    elif choice == StartupOption.CUSTOM.value:  # Custom config wizard selected
        config = _create_custom_config()
        # If config is None, user rejected it, so restart
        return config if config is not None else show_startup_menu()

    elif (
        choice == StartupOption.FROM_FILE.value and config_files
    ):  # Load from existing file selected
        config = _load_config_from_file(config_files)
        # If loading failed, restart the menu
        return config if config is not None else show_startup_menu()

    # Fallback to default
    return None


def _display_startup_options(has_config_files: bool) -> List[str]:
    """Display startup configuration options.

    Shows available configuration methods based on whether existing
    configuration files were found.

    Args:
        has_config_files: Whether configuration files were found

    Returns:
        List of available option identifiers for get_choice
    """
    print("¿Cómo deseas configurar el sistema?\n")
    print("  1. Usar configuración por defecto (Recomendado)")
    print("  2. Configuración personalizada (asistente)")

    options = ["default", "custom"]

    if has_config_files:
        print("  3. Cargar desde archivo existente")
        options.append("file")

    return options


def _load_default_config() -> None:
    """Handle selection of default configuration.

    Returns:
        None to indicate default config should be used by ConfigLoader
    """
    print_info("Usando configuración por defecto...")
    return None


def _create_custom_config() -> Optional[RAGConfig]:
    """Create a custom configuration using the interactive wizard.

    Runs the configuration wizard, displays a summary for user review,
    and optionally saves the configuration to a YAML file.

    Returns:
        RAGConfig instance if user confirms the configuration,
        None if user rejects (triggering menu restart)
    """
    config = run_configuration_wizard()
    show_config_summary(config)

    if not confirm("\n¿La configuración es correcta?", default=True):
        return None  # Signal to restart the menu

    if confirm("¿Guardar configuración en archivo YAML?", default=True):
        path = get_input("Ruta del archivo", DEFAULT_CONFIG_FILENAME)
        save_config_to_yaml(config, path)

    return config


def _select_config_file(config_files: List[str]) -> Optional[str]:
    """Display available configuration files and let user select one.

    Args:
        config_files: List of available configuration file paths

    Returns:
        Selected configuration file path, or None if selection failed
    """
    print("\nArchivos de configuración encontrados:")
    for i, file in enumerate(config_files, 1):
        print(f"  {i}. {file}")

    file_choice = get_choice("Selecciona archivo", config_files, 1)
    return config_files[file_choice - 1]


def _load_config_from_file(config_files: List[str]) -> Optional[RAGConfig]:
    """Load configuration from an existing YAML file.

    Handles the complete flow of file selection, loading, validation,
    and error handling with specific error messages.

    Args:
        config_files: List of available configuration files

    Returns:
        Loaded RAGConfig instance on success,
        None on error or cancellation (triggering menu restart)
    """
    config_path = _select_config_file(config_files)
    if not config_path:
        return None

    try:
        config = ConfigLoader.load_from_yaml(config_path)
        print_success(f"Configuración cargada desde: {config_path}")
        show_config_summary(config)
        return config
    except FileNotFoundError:
        print_error(f"Archivo no encontrado: {config_path}")
        return None
    except ValueError as e:
        print_error(f"Error de validación en configuración: {e}")
        return None
    except Exception as e:
        print_error(f"Error al cargar configuración: {e}")
        return None
