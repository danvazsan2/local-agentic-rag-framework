"""
Main Configuration Wizard
===========================

Orchestrates the full configuration wizard and provides summary/save functions.
"""

from rag_framework.config import RAGConfig, ConfigLoader
from rag_framework.config.models import RetrievalConfig, RouterConfig
from typing import Any

from ..ui import get_input, print_header, print_success, print_error, get_multi_select
from .component_wizards import (
    configure_llm,
    configure_embedding,
    configure_reranker,
    configure_vector_store,
    configure_directories,
    configure_prompt_template,
    configure_sql,
)

# ---------------------------------------------------------------------------
# Component registry: (display_label, internal_key)
# The order here matches the order shown to the user.
# ---------------------------------------------------------------------------
_CONFIGURABLE_COMPONENTS = [
    ("LLM (Modelo de Lenguaje)", "llm"),
    ("Embeddings", "embedding"),
    ("Retrieval y Reranker", "retrieval"),
    ("Vector Store", "vector_store"),
    ("Directorios", "directories"),
    ("Plantilla de Prompts", "prompt"),
    ("SQL (Datos Estructurados)", "sql"),
]


def run_configuration_wizard() -> RAGConfig:
    """Run the interactive configuration wizard.

    First shows a component selector so the user can pick which parts
    of the configuration to customise. Components that are **not**
    selected keep their default values from ``RAGConfig``.

    Returns:
        Fully configured RAGConfig instance
    """
    print_header("Asistente de Configuración")
    print("Te guiaré paso a paso para configurar el sistema RAG.")
    print("Primero, selecciona qué componentes deseas personalizar.\n")

    # --- Component selector ------------------------------------------------
    labels = [label for label, _ in _CONFIGURABLE_COMPONENTS]
    selected_indices = get_multi_select(
        "Componentes a configurar",
        labels,
    )
    selected_keys = {_CONFIGURABLE_COMPONENTS[i][1] for i in selected_indices}

    # --- Run only the selected wizards ------------------------------------
    # LLM
    if "llm" in selected_keys:
        llm_config = configure_llm()
    else:
        llm_config = None  # will use RAGConfig default

    # Embedding
    if "embedding" in selected_keys:
        embedding_config = configure_embedding()
    else:
        embedding_config = None

    # Retrieval & Reranker
    if "retrieval" in selected_keys:
        retrieval_config = RetrievalConfig(
            use_hybrid_search=True,
            top_k=int(get_input("\n¿Cuántos documentos recuperar? (top_k)", "10")),
            reranker=configure_reranker(),
        )
    else:
        retrieval_config = None

    # Vector Store
    if "vector_store" in selected_keys:
        vector_store_config = configure_vector_store()
    else:
        vector_store_config = None

    # Directories
    if "directories" in selected_keys:
        directory_config = configure_directories()
    else:
        directory_config = None

    # Prompt Template
    if "prompt" in selected_keys:
        prompt_template, custom_prompt = configure_prompt_template()
    else:
        prompt_template, custom_prompt = "default", None

    # SQL (optional)
    if "sql" in selected_keys:
        sql_config = configure_sql()
    else:
        sql_config = None

    # Router follows SQL
    router_config = RouterConfig(enabled=sql_config.enabled if sql_config else False)

    # --- Build config using defaults for non-selected components ----------
    kwargs: dict[str, Any] = {}
    if llm_config is not None:
        kwargs["llm"] = llm_config
    if embedding_config is not None:
        kwargs["embedding"] = embedding_config
    if retrieval_config is not None:
        kwargs["retrieval"] = retrieval_config
    if vector_store_config is not None:
        kwargs["vector_store"] = vector_store_config
    if directory_config is not None:
        kwargs["directories"] = directory_config
    if sql_config is not None:
        kwargs["sql"] = sql_config

    kwargs["router"] = router_config
    kwargs["prompt_template"] = prompt_template
    kwargs["custom_prompt"] = custom_prompt
    kwargs["debug"] = False

    config = RAGConfig(**kwargs)
    return config


def show_config_summary(config: RAGConfig) -> None:
    """Display a formatted configuration summary.

    Args:
        config: RAGConfig instance to display
    """
    print_header("Resumen de Configuración")

    sections = [
        (
            "LLM",
            [
                ("Proveedor", config.llm.provider),
                (
                    "Modelo",
                    (
                        config.llm.model
                        if not config.llm.is_local
                        else config.llm.local_model_path
                    ),
                ),
                ("Temperature", config.llm.temperature),
                ("Max Tokens", config.llm.max_tokens),
            ],
        ),
        (
            "Embeddings",
            [
                ("Proveedor", config.embedding.provider),
                (
                    "Modelo",
                    (
                        config.embedding.model
                        if not config.embedding.is_local
                        else config.embedding.local_model_path
                    ),
                ),
            ],
        ),
        (
            "Vector Store",
            [
                ("Proveedor", config.vector_store.provider),
                ("Directorio", config.vector_store.persist_directory),
            ],
        ),
        (
            "Retrieval",
            [
                (
                    "Búsqueda Híbrida",
                    "Sí" if config.retrieval.use_hybrid_search else "No",
                ),
                ("Top K", config.retrieval.top_k),
                ("Reranker", "Sí" if config.retrieval.reranker.enabled else "No"),
            ],
        ),
        (
            "Directorios",
            [
                ("Documentos", config.directories.documents_dir),
                ("Vector Store", config.directories.vector_store_dir),
            ],
        ),
        (
            "Otros",
            [
                ("Plantilla", config.prompt_template),
                ("SQL Habilitado", "Sí" if config.sql.enabled else "No"),
                ("Routing", "Sí" if config.router.enabled else "No"),
                (
                    "Corrective RAG",
                    "Sí" if config.corrective_rag.enabled else "No",
                ),
            ],
        ),
    ]

    for section_name, items in sections:
        print(f"{section_name}:")
        for label, value in items:
            print(f"  {label:<20} {value}")
        print()


def save_config_to_yaml(config: RAGConfig, path: str) -> bool:
    """Save configuration to YAML file.

    Args:
        config: RAGConfig instance to save
        path: File path to save to

    Returns:
        True if successful, False otherwise
    """
    try:
        ConfigLoader.save_yaml(config, path)
        print_success(f"Configuración guardada en: {path}")
        return True
    except Exception as e:
        print_error(f"Error al guardar configuración: {e}")
        return False
