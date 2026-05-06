"""
Menu Action Handlers
=====================

Handlers for menu operations that interact with the RAG framework.
"""

from typing import Optional

from rag_framework import RAGFramework
from rag_framework.prompts import PromptTemplates

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
from ..wizards import (
    configure_llm,
    configure_embedding,
    configure_reranker,
    configure_vector_store,
    configure_directories,
    configure_prompt_template,
    configure_sql,
    show_config_summary,
    save_config_to_yaml,
)


def handle_ingest(rag: RAGFramework) -> None:
    """Handle document ingestion.

    Args:
        rag: RAGFramework instance
    """
    print_header("Ingestión de Documentos")

    force_reindex = confirm("¿Forzar re-indexación?", default=False)

    try:
        rag.ingest(force_reindex=force_reindex)
        print_success("Documentos ingestados correctamente")
    except Exception as e:
        print_error(f"Error en la ingestión: {e}")


def handle_chat(rag: RAGFramework) -> None:
    """Handle chat mode.

    Args:
        rag: RAGFramework instance
    """
    print_header("Modo Chat Interactivo")
    print_info("Escribe tu pregunta. Comandos: 'exit' para salir, 'help' para ayuda.\n")

    try:
        rag.ingest()
        rag.chat()
    except Exception as e:
        print_error(f"Error en modo chat: {e}")


def handle_single_query(rag: RAGFramework) -> None:
    """Handle single query.

    Args:
        rag: RAGFramework instance
    """
    print_header("Consulta Única")

    try:
        rag.ingest()
    except Exception as e:
        print_error(f"Error cargando índice: {e}")
        return

    question = get_input("Introduce tu pregunta")

    if question:
        print("\nProcesando consulta...")
        try:
            response = rag.query(question)
            print(f"\nRespuesta:")
            print(f"{response}\n")
        except Exception as e:
            print_error(f"Error en la consulta: {e}")


def handle_validate_models(rag: RAGFramework) -> None:
    """Handle model validation.

    Args:
        rag: RAGFramework instance
    """
    print_header("Validación de Modelos")

    try:
        result = rag.validate_models()
        if result:
            print_success("Todos los modelos están disponibles y funcionando")
        else:
            print_warning("Algunos modelos no están disponibles")
    except Exception as e:
        print_error(f"Error en validación: {e}")


def handle_list_templates() -> None:
    """Handle listing prompt templates."""
    print_header("Plantillas de Prompts Disponibles")

    templates = PromptTemplates.list_templates()
    for name, desc in templates.items():
        print(f"  {name}")
        print(f"    {desc}\n")


def handle_show_config(rag: RAGFramework) -> None:
    """Handle configuration display.

    Args:
        rag: RAGFramework instance
    """
    show_config_summary(rag.config)


def handle_edit_config(rag: RAGFramework) -> Optional[None]:
    """Handle configuration editing.

    Args:
        rag: RAGFramework instance

    Returns:
        None (modifies rag.config in place)
    """
    print_header("Editar Configuración")

    print("¿Qué deseas modificar?")
    options = [
        "LLM",
        "Embeddings",
        "Reranker",
        "Vector Store",
        "Plantilla de Prompts",
        "Directorios",
        "SQL",
        "Volver",
    ]

    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")

    choice = get_choice("Selecciona opción", options, len(options))

    config_updated = False

    if choice == 1:
        rag.config.llm = configure_llm()
        config_updated = True
    elif choice == 2:
        rag.config.embedding = configure_embedding()
        config_updated = True
    elif choice == 3:
        rag.config.retrieval.reranker = configure_reranker()
        config_updated = True
    elif choice == 4:
        rag.config.vector_store = configure_vector_store()
        config_updated = True
    elif choice == 5:
        template, custom = configure_prompt_template()
        rag.config.prompt_template = template
        rag.config.custom_prompt = custom
        config_updated = True
    elif choice == 6:
        rag.config.directories = configure_directories()
        config_updated = True
    elif choice == 7:
        rag.config.sql = configure_sql()
        rag.config.router.enabled = rag.config.sql.enabled
        config_updated = True

    if config_updated:
        print_success("Configuración actualizada")

        if confirm("¿Guardar configuración en archivo YAML?", default=False):
            path = get_input("Ruta del archivo", "config/rag_config_custom.yaml")
            save_config_to_yaml(rag.config, path)


# ============================================================================
# Feature Toggles
# ============================================================================

_FEATURE_LABELS = [
    "Corrective RAG",
    "Router SQL/Documentos",
    "Búsqueda Híbrida",
    "Reranker",
    "Modo Debug",
]


def _get_feature_states(rag: RAGFramework) -> list:
    """Return current on/off state for each feature (same order as _FEATURE_LABELS)."""
    return [
        rag.config.corrective_rag.enabled,
        rag.config.router.enabled,
        rag.config.retrieval.use_hybrid_search,
        rag.config.retrieval.reranker.enabled,
        rag.config.debug,
    ]


def get_features_status_line(rag: RAGFramework) -> str:
    """Build a compact one-line status string for the main menu display.

    Returns a string like: CRAG:✗  Router:✓  SQL:✗  Híbrida:✓  Rerank:✓  Debug:✗
    """
    labels_short = ["CRAG", "SQL", "Híbrida", "Rerank", "Debug"]
    states = _get_feature_states(rag)
    parts = [f"{l}:{'✓' if s else '✗'}" for l, s in zip(labels_short, states)]
    return "  ".join(parts)


def handle_toggle_features(rag: RAGFramework) -> None:  # noqa: C901
    """Interactive panel to enable/disable RAG features in real time.

    Displays all toggleable features with their current state using a
    checkbox-style list. The user selects which features to enable; the
    panel reflects changes immediately. When Corrective RAG is enabled
    the user is prompted for its key parameters.

    Args:
        rag: RAGFramework instance (config modified in place)
    """
    print_header("⚡ Funcionalidades del Sistema")

    while True:
        states = _get_feature_states(rag)

        # ── draw panel ────────────────────────────────────────────────────
        print()
        print(f"  {'#':<4} {'Funcionalidad':<30} {'Estado':<10} {'Parámetros'}")
        print(f"  {'─'*4} {'─'*30} {'─'*10} {'─'*30}")

        rows = [
            _row_crag(rag),
            _row_sql_router(rag),
            _row_hybrid(rag),
            _row_reranker(rag),
            _row_debug(rag),
        ]
        for i, (label, status, params) in enumerate(rows, 1):
            tag = "[✓]" if states[i - 1] else "[ ]"
            print(f"  {tag} {i:<3} {label:<30} {status:<10} {params}")

        print()
        print_info(
            "Introduce número(s) para alternar (ej: 1 3), "
            "un número seguido de parámetros (ej: 1 0.7 2), "
            "o Enter para volver"
        )

        try:
            raw = input("  ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            break

        # Parse first token as feature index, rest as optional parameters
        tokens = raw.split()
        try:
            idx = int(tokens[0])
        except ValueError:
            print_error("Introduce un número válido")
            continue

        if not 1 <= idx <= len(_FEATURE_LABELS):
            print_error(f"Selecciona entre 1 y {len(_FEATURE_LABELS)}")
            continue

        extra = tokens[1:]  # optional extra params passed inline
        _apply_toggle(rag, idx, extra)
        print_success(
            f"{_FEATURE_LABELS[idx - 1]}: "
            f"{'activado' if _get_feature_states(rag)[idx - 1] else 'desactivado'}"
        )


# ── row builders ──────────────────────────────────────────────────────────


def _row_crag(rag):
    label = "Corrective RAG"
    enabled = rag.config.corrective_rag.enabled
    status = "ON" if enabled else "off"
    params = (
        f"threshold={rag.config.corrective_rag.relevance_threshold}  "
        f"retries={rag.config.corrective_rag.max_retries}"
        if enabled
        else "—"
    )
    return label, status, params


def _row_sql_router(rag):
    label = "Router SQL/Documentos"
    enabled = rag.config.router.enabled
    status = "ON" if enabled else "off"
    params = f"default={rag.config.router.default_source}" if enabled else "—"
    return label, status, params


def _row_hybrid(rag):
    label = "Búsqueda Híbrida"
    enabled = rag.config.retrieval.use_hybrid_search
    status = "ON" if enabled else "off"
    params = (
        f"alpha={rag.config.retrieval.alpha}  top_k={rag.config.retrieval.top_k}"
        if enabled
        else f"top_k={rag.config.retrieval.top_k} (solo vectorial)"
    )
    return label, status, params


def _row_reranker(rag):
    label = "Reranker"
    enabled = rag.config.retrieval.reranker.enabled
    status = "ON" if enabled else "off"
    params = (
        f"model={rag.config.retrieval.reranker.model.split('/')[-1]}  "
        f"top_n={rag.config.retrieval.reranker.top_n}"
        if enabled
        else "—"
    )
    return label, status, params


def _row_debug(rag):
    label = "Modo Debug"
    enabled = rag.config.debug
    status = "ON" if enabled else "off"
    params = "Muestra chunks y trazas" if enabled else "—"
    return label, status, params


# ── toggle applicators ────────────────────────────────────────────────────


def _apply_toggle(rag: RAGFramework, idx: int, extra: list) -> None:  # noqa: C901
    """Apply a toggle for the selected feature index.

    When extra params are provided they are applied directly
    (e.g. ``1 0.7 2`` → enable CRAG with threshold=0.7, retries=2).
    Otherwise the feature is toggled and sub-params are prompted
    interactively only when enabling a feature that has them.
    """
    if idx == 1:  # Corrective RAG
        new_state = not rag.config.corrective_rag.enabled
        if new_state and not extra:
            # Prompt for params when enabling interactively
            threshold_str = get_input(
                "  Umbral de relevancia (relevance_threshold) [0.0-1.0]",
                str(rag.config.corrective_rag.relevance_threshold),
            )
            retries_str = get_input(
                "  Reintentos máx. (max_retries)",
                str(rag.config.corrective_rag.max_retries),
            )
            try:
                rag.config.corrective_rag.relevance_threshold = float(threshold_str)
                rag.config.corrective_rag.max_retries = int(retries_str)
            except ValueError:
                print_warning("Parámetros inválidos, usando valores actuales")
        elif new_state and extra:
            try:
                if len(extra) >= 1:
                    rag.config.corrective_rag.relevance_threshold = float(extra[0])
                if len(extra) >= 2:
                    rag.config.corrective_rag.max_retries = int(extra[1])
            except ValueError:
                print_warning("Parámetros inline inválidos, usando valores actuales")
        rag.config.corrective_rag.enabled = new_state
        # Invalidate cached query engine so CRAG takes effect
        rag._query_engine = None

    elif idx == 2:  # Router SQL/Documentos
        rag.config.router.enabled = not rag.config.router.enabled

    elif idx == 3:  # Búsqueda Híbrida
        rag.config.retrieval.use_hybrid_search = (
            not rag.config.retrieval.use_hybrid_search
        )
        rag._query_engine = None

    elif idx == 4:  # Reranker
        rag.config.retrieval.reranker.enabled = (
            not rag.config.retrieval.reranker.enabled
        )
        rag._query_engine = None

    elif idx == 5:  # Modo Debug
        rag.config.debug = not rag.config.debug
        rag._query_engine = None


def handle_toggle_debug(rag: RAGFramework) -> None:
    """Legacy shim — kept for backward compatibility; delegates to panel."""
    handle_toggle_features(rag)


def handle_save_config(rag: RAGFramework) -> None:
    """Handle saving configuration to file.

    Args:
        rag: RAGFramework instance
    """
    print_header("Guardar Configuración")

    path = get_input("Ruta del archivo YAML", "config/rag_config_custom.yaml")
    save_config_to_yaml(rag.config, path)


# ============================================================================
# Download Models Handler
# ============================================================================


def handle_download_models() -> None:
    """Interactive handler for downloading HuggingFace models."""
    from rag_framework.utils.model_downloader import (
        download_llm,
        download_embedding,
        download_reranker,
        list_popular_models,
    )
    from rag_framework.utils.constants import POPULAR_MODELS
    import os

    print_header("Descargar Modelos de HuggingFace")

    options = [
        "Listar modelos populares",
        "Descargar LLM",
        "Descargar modelo de Embeddings",
        "Descargar modelo Reranker",
        "Volver",
    ]
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")

    choice = get_choice("Selecciona opción", options, len(options))

    if choice == 1:
        list_popular_models()
        return

    if choice == 5:
        return

    type_map = {2: "llm", 3: "embedding", 4: "reranker"}
    download_map = {2: download_llm, 3: download_embedding, 4: download_reranker}
    model_type = type_map[choice]

    # Show popular shortcuts for the chosen type
    popular = POPULAR_MODELS.get(model_type, {})
    if popular:
        print(f"\nAtajos disponibles para {model_type}:")
        for name, mid in popular.items():
            print(f"  {name:20s} → {mid}")

    model_id = get_input(
        "\nModel ID o atajo (nombre corto o ID completo de HuggingFace)"
    )
    if not model_id:
        print_warning("No se proporcionó un modelo.")
        return

    # Resolve shortcut
    if model_id in popular:
        resolved = popular[model_id]
        print_info(f"Usando modelo popular: {model_id} → {resolved}")
        model_id = resolved

    output = get_input("Directorio de salida (Enter para usar el predeterminado)", "")
    token = os.getenv("HF_TOKEN")
    if not token:
        token = get_input("HuggingFace token (Enter para omitir)", "") or None

    downloader = download_map[choice]
    result = downloader(model_id, output or None, token)

    if result:
        print_success(f"Modelo descargado en: {result}")
    else:
        print_error("La descarga falló. Revisa los mensajes anteriores.")


# ============================================================================
# API Server Handler
# ============================================================================


def handle_start_api() -> None:
    """Launch the API server from within the CLI.

    The server runs until the user presses Ctrl+C, then control
    returns to the interactive menu.
    """
    from rag_framework.utils.constants import DEFAULT_API_HOST, DEFAULT_API_PORT

    print_header("Lanzar Servidor API")

    host = get_input("Host", DEFAULT_API_HOST)
    port_str = get_input("Puerto", str(DEFAULT_API_PORT))
    try:
        port = int(port_str)
    except ValueError:
        print_error(f"Puerto inválido: {port_str}")
        return

    config_path = (
        get_input("Ruta al archivo de configuración (Enter para omitir)", "") or None
    )

    print_info(f"Iniciando servidor API en http://{host}:{port} ...")
    print_info("Pulsa Ctrl+C para detener el servidor y volver al menú.\n")

    try:
        from api.server import run_server

        run_server(host, port, config_path)
    except KeyboardInterrupt:
        print("\n")
        print_info("Servidor API detenido. Volviendo al menú principal.")
