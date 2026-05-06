"""
Menu Handlers Package
======================

Action handlers for menu operations.
"""

from .menu_handlers import (
    handle_ingest,
    handle_chat,
    handle_single_query,
    handle_validate_models,
    handle_list_templates,
    handle_show_config,
    handle_edit_config,
    handle_toggle_debug,
    handle_toggle_features,
    get_features_status_line,
    handle_save_config,
    handle_download_models,
    handle_start_api,
)

__all__ = [
    "handle_ingest",
    "handle_chat",
    "handle_single_query",
    "handle_validate_models",
    "handle_list_templates",
    "handle_show_config",
    "handle_edit_config",
    "handle_toggle_debug",
    "handle_toggle_features",
    "get_features_status_line",
    "handle_save_config",
    "handle_download_models",
    "handle_start_api",
]
