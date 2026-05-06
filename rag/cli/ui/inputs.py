"""
Input Helpers
==============

User input utilities with defaults and validation.
"""

from typing import Optional, List, Set
from .formatters import print_error, print_header


def get_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value.

    Args:
        prompt: Prompt text to display
        default: Default value if user presses Enter

    Returns:
        User input or default value
    """
    if default:
        prompt_text = f"{prompt} [{default}]: "
    else:
        prompt_text = f"{prompt}: "

    try:
        value = input(prompt_text).strip()
        return value if value else (default or "")
    except EOFError:
        return default or ""


def get_choice(prompt: str, options: List[str], default: int = 1) -> int:
    """Get a numeric choice from user.

    Args:
        prompt: Prompt text to display
        options: List of available options (for validation)
        default: Default choice number (1-indexed)

    Returns:
        Selected option number (1-indexed)
    """
    while True:
        try:
            choice = input(f"{prompt} [{default}]: ").strip()
            if not choice:
                return default
            choice_int = int(choice)
            if 1 <= choice_int <= len(options):
                return choice_int
            print_error(f"Por favor, selecciona entre 1 y {len(options)}")
        except ValueError:
            print_error("Por favor, introduce un número válido")
        except EOFError:
            return default


def confirm(prompt: str, default: bool = True) -> bool:
    """Get yes/no confirmation from user.

    Args:
        prompt: Confirmation prompt text
        default: Default value if user presses Enter

    Returns:
        True if user confirmed, False otherwise
    """
    default_str = "S/n" if default else "s/N"
    try:
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not response:
            return default
        return response in ("s", "si", "sí", "y", "yes")
    except EOFError:
        return default


def get_multi_select(
    title: str,
    options: List[str],
    default_selected: Optional[Set[int]] = None,
) -> List[int]:
    """Interactive multi-select with toggle checkboxes.

    Displays a numbered list of options with checkboxes that the user
    can toggle on/off. Supports:
    - Individual numbers (e.g. ``1 3 5``) to toggle specific items
    - Ranges (e.g. ``2-4``) to toggle a range
    - ``*`` to select/deselect all
    - Empty input (Enter) to confirm the current selection

    Args:
        title: Header text for the selector
        options: List of option labels to display
        default_selected: Optional set of 0-indexed pre-selected indices

    Returns:
        Sorted list of 0-indexed selected indices
    """
    selected: Set[int] = set(default_selected) if default_selected else set()

    while True:
        # Display current state
        print_header(title)
        for i, option in enumerate(options):
            marker = "x" if i in selected else " "
            print(f"  [{marker}] {i + 1}. {option}")

        count = len(selected)
        total = len(options)
        print(f"\n  ({count}/{total} seleccionados)")
        print(
            "\n  Introduce números para alternar (ej: 1 3 5), "
            "rango (ej: 2-4), * = todos, Enter = confirmar"
        )

        try:
            user_input = input("\n  ▸ ").strip()
        except EOFError:
            return sorted(selected)

        # Confirm selection
        if not user_input:
            return sorted(selected)

        # Toggle all
        if user_input == "*":
            if len(selected) == total:
                selected.clear()
            else:
                selected = set(range(total))
            continue

        # Parse individual tokens
        _parse_multi_select_input(user_input, options, selected)


def _parse_multi_select_input(
    user_input: str,
    options: List[str],
    selected: Set[int],
) -> None:
    """Parse user input tokens and toggle items in *selected*.

    Handles space-separated numbers and dash-separated ranges.

    Args:
        user_input: Raw user input string
        options: Full list of options (used for bounds checking)
        selected: Mutable set of currently selected indices (0-indexed)
    """
    total = len(options)

    for token in user_input.split():
        # Range (e.g. "2-4")
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start = int(parts[0])
                end = int(parts[1])
                if 1 <= start <= end <= total:
                    for idx in range(start - 1, end):
                        _toggle(selected, idx)
                else:
                    print_error(
                        f"Rango inválido: {token} (usa 1-{total})"
                    )
            except ValueError:
                print_error(f"Rango inválido: {token}")
        else:
            # Single number
            try:
                num = int(token)
                if 1 <= num <= total:
                    _toggle(selected, num - 1)
                else:
                    print_error(
                        f"Número fuera de rango: {token} (usa 1-{total})"
                    )
            except ValueError:
                print_error(f"Entrada inválida: {token}")


def _toggle(selected: Set[int], index: int) -> None:
    """Toggle an index in/out of the selected set.

    Args:
        selected: Mutable set of selected indices
        index: 0-indexed item to toggle
    """
    if index in selected:
        selected.discard(index)
    else:
        selected.add(index)

