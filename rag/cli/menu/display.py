"""
Menu Display
=============

Functions for rendering menu interfaces.
"""


def display_main_menu(rag) -> None:
    """Display the main interactive menu.

    Args:
        rag: RAGFramework instance (used to read live feature states)
    """
    # Build compact feature summary for option 9
    cfg = rag.config
    enabled_count = sum(
        [
            cfg.corrective_rag.enabled,
            cfg.router.enabled,
            cfg.retrieval.use_hybrid_search,
            cfg.retrieval.reranker.enabled,
            cfg.debug,
        ]
    )
    feat_label = f"Funcionalidades [{enabled_count}/5 activas]"

    print(f"\n╔{'═' * 40}╗")
    print(f"║{'MENÚ PRINCIPAL':^40}║")
    print(f"╠{'═' * 40}╣")
    print(f"║  1. Ingestar documentos                ║")
    print(f"║  2. Modo chat interactivo              ║")
    print(f"║  3. Consulta única                     ║")
    print(f"╠{'─' * 40}╣")
    print(f"║  4. Validar modelos                    ║")
    print(f"║  5. Listar plantillas de prompts       ║")
    print(f"║  6. Mostrar configuración              ║")
    print(f"║  7. Editar configuración               ║")
    print(f"║  8. Guardar configuración              ║")
    print(f"║  9. {feat_label:<35}║")
    print(f"╠{'─' * 40}╣")
    print(f"║ 10. Descargar modelos                  ║")
    print(f"║ 11. Lanzar servidor API                ║")
    print(f"╠{'─' * 40}╣")
    print(f"║  0. Salir                              ║")
    print(f"╚{'═' * 40}╝")
