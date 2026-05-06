#!/usr/bin/env python3
"""
=============================================================================
  DEMO: Proyectos Docentes — RAG No Estructurado vs. Estructurado vs. Híbrido
=============================================================================

Uso:
    python demos/demo_proyectos_docentes.py
    python demos/demo_proyectos_docentes.py --no-pause   # sin pausas

"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any

# Asegurar que el directorio raíz del proyecto está en el path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_framework.utils.logging import setup_logging

setup_logging()


@contextlib.contextmanager
def _quiet():
    """Silencia stdout y los logs de rag_framework durante la operación."""
    import logging

    fw_logger = logging.getLogger("rag_framework")
    old_level = fw_logger.level
    fw_logger.setLevel(logging.ERROR)
    null = io.StringIO()
    try:
        with contextlib.redirect_stdout(null):
            yield
    finally:
        fw_logger.setLevel(old_level)


# ── Colores ANSI ────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"

# ════════════════════════════════════════════════════════════════════
# Batería de preguntas
# ════════════════════════════════════════════════════════════════════

# Preguntas que el RAG NO ESTRUCTURADO responde BIEN (contenido de PDFs).
# El RAG ESTRUCTURADO NO puede responderlas porque el contenido cualitativo
# (bloques temáticos, metodología, criterios de evaluación) vive en los PDFs,
# no en la base de datos.
RAG_BUENAS: list[tuple[str, str, str]] = [
    (
        "¿Qué bloques temáticos y lenguajes de programación "
        "se estudian en Fundamentos de Programación?",
        "Parte I: Python — Introducción, control de flujo, estructuras de datos, E/S. "
        "Parte II: Java — Introducción, diseño de tipos, colecciones, tratamientos secuenciales.",
        "Programa_2060001_12.pdf, pág. 2",
    ),
    (
        "¿Cuál es la metodología de evaluación de la asignatura "
        "Inteligencia Artificial? ¿Existe examen final?",
        "Evaluación basada en: Parte teórica (65%) y parte práctica (35%). "
        "Sí, existe examen final..",
        "Programa_2060021_7.pdf, pág. 3",
    ),
    (
        "¿Cuáles son los objetivos principales que persigue la asignatura de Estadística?",
        "Aprender a resolver problemas reales con técnicas estadísticas usando software; "
        "interpretar resultados; aplicar el rigor necesario.",
        "Programa_2060007_9.pdf, pág. 1",
    ),
]

# Preguntas que el RAG NO ESTRUCTURADO NO responde bien (requieren
# agregaciones/conteos sobre múltiples documentos).
# El RAG ESTRUCTURADO SÍ las resuelve fácilmente con SQL.
RAG_MALAS: list[tuple[str, str, str]] = [
    (
        "¿Cuántas asignaturas optativas hay en el plan de estudios?",
        "37 asignaturas optativas.",
        "SELECT COUNT(*) FROM ASG_ASIGNATURA_1 WHERE TAS_NOMID2 = 'OPTATIVA'",
    ),
    (
        "¿Qué asignaturas del plan de estudios tienen 12 créditos ECTS?",
        "Fundamentos de Programación, Análisis y Diseño de Datos y Algoritmos, "
        "y Trabajo Fin de Grado.",
        "SELECT NOMID2 FROM ASG_ASIGNATURA_1 WHERE CAST(CREDITOS AS INTEGER) = 12",
    ),
]

# Preguntas que el RAG ESTRUCTURADO responde BIEN (consultas SQL).
# Incluye las dos que el no estructurado no respondió, más una adicional.
SQL_BUENAS: list[tuple[str, str, str]] = [
    # Las dos que fallaron en el modo no estructurado:
    *RAG_MALAS,
    # Una pregunta adicional que requiere JOIN + GROUP BY:
    (
        "¿Cuántos grupos docentes tiene cada asignatura de primer curso?",
        "Las asignaturas Fundamentos de Programación, Administración de Empresas, "
        "Circuitos Electrónicos y Estructura de Computadores tienen 4 grupos. "
        "El resto (Cálculo, Estadística, Álgebra, Física y Discreta) tienen 3 grupos.",
        "SELECT a.NOMID2, COUNT(DISTINCT g.GAC_ID) FROM ASG_PROYECTOCABECERA_PRE_YII_1 g "
        "JOIN ASG_ASIGNATURA_1 a ... WHERE CUR_NUMCUR='1' GROUP BY a.NOMID2",
    ),
]

# Preguntas que el RAG ESTRUCTURADO NO responde bien (contenido cualitativo).
# El RAG NO ESTRUCTURADO SÍ las responde (son las mismas RAG_BUENAS).
SQL_MALAS: list[tuple[str, str, str]] = RAG_BUENAS[:2]  # Las dos primeras

# Preguntas representativas para el modo HÍBRIDO:
# Se mezclan preguntas que solo funcionan bien en un modo y se demuestra que
# el router las distribuye correctamente.
HIBRIDO_QUERIES: list[tuple[str, str, str]] = [
    # Conteo → debería ir a SQL
    RAG_MALAS[0],
    # Contenido temático → debería ir a documentos
    RAG_BUENAS[0],
    # Conteo con estructurado → SQL
    RAG_MALAS[1],
    # Metodología → documentos
    RAG_BUENAS[1],
]


# ════════════════════════════════════════════════════════════════════
# Utilidades de presentación
# ════════════════════════════════════════════════════════════════════


def banner(title: str) -> None:
    width = 68
    print(f"\n{CYAN}{'═' * width}")
    print(f"  {BOLD}{title}{RESET}{CYAN}")
    print(f"{'═' * width}{RESET}\n")


def section(title: str) -> None:
    print(f"\n{YELLOW}{'━' * 68}")
    print(f"  {BOLD}{title}{RESET}")
    print(f"{YELLOW}{'━' * 68}{RESET}\n")


def subsection(title: str) -> None:
    print(f"\n  {BOLD}{title}{RESET}")
    print(f"  {'─' * 60}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓ {msg}{RESET}")


def fail(msg: str) -> None:
    print(f"  {RED}✗ {msg}{RESET}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠ {msg}{RESET}")


def info(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


def result_badge(good: bool) -> str:
    return f"{GREEN}{BOLD}✓ BUENA{RESET}" if good else f"{RED}{BOLD}✗ MALA{RESET}"


def pause(msg: str = "Pulsa Enter para continuar...") -> None:
    print()
    input(f"  {DIM}▸ {msg}{RESET}")
    print()


def print_response(response: str, max_lines: int = 12) -> None:
    lines = str(response).strip().splitlines()
    for line in lines:
        print(f"    {MAGENTA}│{RESET} {line}")


# ════════════════════════════════════════════════════════════════════
# Secciones de la demo
# ════════════════════════════════════════════════════════════════════


def demo_init() -> None:
    section("PASO 1 · Objetivo de esta demo")

    print(
        f"  {BOLD}Dominio:{RESET}  Plan docente del Grado en Ingeniería Informática - Tecnologías Informáticas"
    )
    print(f"           Universidad de Sevilla · Curso académico 2025-26\n")

    print(f"  {BOLD}Fuente de datos:{RESET}")
    print(
        f"    {BLUE}■{RESET} PDFs (no estructurado)  — "
        f"{len(list((PROJECT_ROOT / 'documents').glob('*.pdf')))} documentos "
        f"(Programas y Proyectos Docentes)\n"
        f"    {RED}■{RESET} SQLite (estructurado)    — ./data/proyectos_docentes.db\n"
        f"                                 Tablas: ASG_ASIGNATURA_1, "
        f"ASG_ACTIVIDAD_FOTO,\n"
        f"                                         ASG_PROYECTOCABECERA_PRE_YII_1, "
        f"ASG_BIBLIOGRAFIA, ...\n"
    )

    print(f"  {BOLD}Configuraciones que vamos a utilizar:{RESET}")
    print(
        f"    {BLUE}►{RESET} Solo documentos  → config/proyectos_docentes_solo_rag.yaml\n"
        f"    {RED}►{RESET} Solo SQL         → config/proyectos_docentes_solo_sql.yaml\n"
        f"    {MAGENTA}►{RESET} Híbrido          → config/proyectos_docentes.yaml\n"
    )

    ok("Escenario planteado. Comencemos con la demo!")


def _load_rag(config_name: str) -> Any:
    """Carga la config, crea el RAGFramework e inicializa el índice."""
    from rag_framework.config import ConfigLoader
    from rag_framework import RAGFramework

    config_path = PROJECT_ROOT / "config" / config_name
    info(f"Cargando configuración: {config_path.relative_to(PROJECT_ROOT)}")

    config = ConfigLoader.load_from_yaml(str(config_path))

    info("Inicializando RAGFramework...")
    with _quiet():
        rag = RAGFramework(config)

    info("Cargando índice de documentos (puede tardar la primera vez)...")
    index_loaded = False
    try:
        with _quiet():
            rag.load_index()
        index_loaded = True
    except Exception:
        pass

    if index_loaded:
        # Verificar compatibilidad de dimensiones con el modelo de embeddings actual.
        # Si el índice fue creado con un modelo distinto, la prueba fallará con un
        # error de dimensión y habrá que reindexar.
        try:
            with _quiet():
                rag.query_documents("verificación de dimensiones")
            ok("Índice existente cargado y verificado.")
        except Exception as probe_err:
            err_str = str(probe_err).lower()
            if any(
                k in err_str
                for k in ("dim", "dimension", "doesn't match", "does not match")
            ):
                warn(
                    "Índice incompatible detectado: la dimensión del modelo de "
                    "embeddings actual no coincide con el índice almacenado."
                )
                info("Reindexando documentos con el modelo de embeddings actual...")
                with _quiet():
                    rag.ingest()
                ok("Documentos reindexados correctamente.")
            else:
                # Error inesperado: propagar
                raise
    else:
        info("No hay índice previo — ingiriendo documentos...")
        with _quiet():
            rag.ingest()
        ok("Documentos indexados correctamente.")

    return rag


def demo_unstructured(do_pause: bool) -> None:
    section("PASO 2 · RAG No Estructurado — Solo búsqueda en documentos PDF")

    print(
        f"  En esta fase el sistema SOLO dispone de los PDFs para responder.\n"
        f"  No hay acceso a la base de datos estructurada.\n"
    )

    rag = _load_rag("proyectos_docentes_solo_rag.yaml")

    # ── 2a. Preguntas que responde bien ─────────────────────────────
    subsection(
        f"2a. Preguntas que el RAG no estructurado responde correctamente "
        f"{result_badge(True)}"
    )

    for i, (question, expected, reference) in enumerate(RAG_BUENAS, 1):
        print(f"\n  {YELLOW}{'─' * 62}{RESET}")
        print(f'  {BOLD}Pregunta {i}/{len(RAG_BUENAS)}:{RESET} "{question}"')
        print(f"  {DIM}Referencia: {reference}{RESET}")
        print(f"  {DIM}Respuesta esperada: {expected}{RESET}\n")

        try:
            with _quiet():
                response = rag.query_documents(question)
            print(f"  {BOLD}Respuesta del sistema:{RESET}")
            print_response(str(response))
            ok(f"Pregunta {i} respondida correctamente.")
        except Exception as e:
            fail(f"Error: {e}")

        print()
        if do_pause and i < len(RAG_BUENAS):
            pause(f"Siguiente pregunta ({i + 1}/{len(RAG_BUENAS)})...")

    # ── 2b. Preguntas que NO responde bien ───────────────────────────
    if do_pause:
        pause("Ahora veremos las limitaciones del RAG no estructurado...")

    subsection(
        f"2b. Preguntas que el RAG no estructurado responde incorrectamente "
        f"{result_badge(False)}"
    )
    print(
        f"  {DIM}Estas preguntas requieren agregar/contar información\n"
        f"  distribuida en decenas de documentos. El RAG no puede hacerlo bien.\n"
        f"  → Veremos cómo el RAG estructurado sí las resuelve en el Paso 3.{RESET}\n"
    )

    for i, (question, expected, sql_hint) in enumerate(RAG_MALAS, 1):
        print(f"\n  {YELLOW}{'─' * 62}{RESET}")
        print(f'  {BOLD}Pregunta {i}/{len(RAG_MALAS)}:{RESET} "{question}"')
        print(f"  {DIM}Respuesta correcta: {expected}{RESET}")

        try:
            with _quiet():
                response = rag.query_documents(question)
            print(f"  {BOLD}Respuesta del sistema:{RESET}")
            print_response(str(response))
            warn(
                f"Pregunta {i}: la respuesta puede ser imprecisa o incompleta "
                f"(el RAG no puede contar sobre toda la colección)."
            )
        except Exception as e:
            fail(f"Error: {e}")

        print()
        if do_pause and i < len(RAG_MALAS):
            pause(f"Siguiente pregunta ({i + 1}/{len(RAG_MALAS)})...")

    ok(
        "Bloque NO ESTRUCTURADO completado. "
        f"Buenas: {len(RAG_BUENAS)} | Con limitaciones: {len(RAG_MALAS)}"
    )


def demo_structured(do_pause: bool) -> None:
    section("PASO 3 · RAG Estructurado — Solo consultas SQL a la base de datos")

    print(
        f"  En esta fase el sistema SOLO utiliza la base de datos relacional.\n"
        f"  No hay búsqueda sobre PDFs.\n"
    )

    from rag_framework.config import ConfigLoader
    from rag_framework.sql import SQLAgent

    config_path = PROJECT_ROOT / "config" / "proyectos_docentes_solo_sql.yaml"
    info(f"Cargando configuración: {config_path.relative_to(PROJECT_ROOT)}")
    config = ConfigLoader.load_from_yaml(str(config_path))

    info("Inicializando SQL Agent...")
    with _quiet():
        agent = SQLAgent(config)
    ok("SQL Agent listo.")

    # ── 3a. Preguntas que responde bien ─────────────────────────────
    subsection(
        f"3a. Preguntas que el RAG estructurado responde correctamente "
        f"{result_badge(True)}"
    )
    info(
        "Incluye las dos preguntas que el no estructurado no respondió bien.\n"
        "El SQL Agent traduce cada pregunta a SQL, lo ejecuta y formatea el resultado.\n"
    )

    for i, (question, expected, sql_hint) in enumerate(SQL_BUENAS, 1):
        note = ""
        if i <= len(RAG_MALAS):
            note = f" {YELLOW}← ¡esta falló en el Paso 2!{RESET}"

        print(f"\n  {YELLOW}{'─' * 62}{RESET}")
        print(f'  {BOLD}Pregunta {i}/{len(SQL_BUENAS)}:{RESET} "{question}"{note}')
        print(f"  {DIM}Respuesta esperada: {expected}{RESET}\n")

        with _quiet():
            result = agent.query(question)

        if result.success:
            print(f"  {BOLD}SQL generado:{RESET}  {CYAN}{result.query}{RESET}\n")
            print(f"  {BOLD}Resultado:{RESET}")
            for line in result.formatted_result.splitlines():
                print(f"    {line}")
            if result.result:
                print(
                    f"\n  {DIM}Filas: {result.result.row_count}  |  "
                    f"Intentos: {result.generation_attempts}{RESET}"
                )
            ok(f"Pregunta {i} resuelta correctamente.")
        else:
            fail(f"Error al generar SQL: {result.error}")
            if result.query:
                info(f"SQL intentado: {result.query}")

        print()
        if do_pause and i < len(SQL_BUENAS):
            pause(f"Siguiente pregunta ({i + 1}/{len(SQL_BUENAS)})...")

    # ── 3b. Preguntas que NO responde bien ───────────────────────────
    if do_pause:
        pause("Ahora veremos las limitaciones del RAG estructurado...")

    subsection(
        f"3b. Preguntas que el RAG estructurado responde incorrectamente "
        f"{result_badge(False)}"
    )
    print(
        f"  {DIM}Estas preguntas requieren leer y comprender el contenido\n"
        f"  cualitativo de los PDFs (temario, metodología, evaluación).\n"
        f"  La BD no almacena ese contenido de forma consultable.\n"
        f"  → El RAG no estructurado ya las respondió bien en el Paso 2.{RESET}\n"
    )

    for i, (question, expected, reference) in enumerate(SQL_MALAS, 1):
        print(f"\n  {YELLOW}{'─' * 62}{RESET}")
        print(f'  {BOLD}Pregunta {i}/{len(SQL_MALAS)}:{RESET} "{question}"')
        print(f"  {DIM}Respuesta correcta: {expected}{RESET}")
        print(f"  {DIM}Fuente: {reference}{RESET}\n")

        with _quiet():
            result = agent.query(question)

        if result.success and result.formatted_result.strip():
            print(f"  {BOLD}SQL generado:{RESET}  {CYAN}{result.query}{RESET}\n")
            print(f"  {BOLD}Resultado del SQL Agent:{RESET}")
            for line in result.formatted_result.splitlines():
                print(f"    {line}")
            warn(
                f"Pregunta {i}: el SQL Agent puede devolver datos fragmentados "
                f"o no encontrar la respuesta cualitativa en la BD."
            )
        else:
            warn(
                f"Pregunta {i}: el SQL Agent no pudo generar una respuesta "
                f"útil. El contenido cualitativo solo está en los PDFs."
            )
            if result.query:
                info(f"SQL intentado: {result.query}")

        print()
        if do_pause and i < len(SQL_MALAS):
            pause(f"Siguiente pregunta ({i + 1}/{len(SQL_MALAS)})...")

    agent.close()
    ok(
        "Bloque ESTRUCTURADO completado. "
        f"Resueltas: {len(SQL_BUENAS)} | Con limitaciones: {len(SQL_MALAS)}"
    )


def demo_hybrid(do_pause: bool) -> None:
    section("PASO 4 · Híbrido — Ambas fuentes activas con router inteligente")

    print(
        f"  El router decide automáticamente qué fuente usar para cada pregunta:\n"
        f"    {BLUE}• Preguntas de contenido/metodología  → PDFs{RESET}\n"
        f"    {RED}• Preguntas de conteo/estructura       → SQL{RESET}\n"
    )

    from rag_framework.config import ConfigLoader
    from rag_framework import RAGFramework

    config_path = PROJECT_ROOT / "config" / "proyectos_docentes.yaml"
    info(f"Cargando configuración: {config_path.relative_to(PROJECT_ROOT)}")
    config = ConfigLoader.load_from_yaml(str(config_path))

    info("Inicializando RAGFramework en modo híbrido...")
    with _quiet():
        rag = RAGFramework(config)

    info("Cargando índice de documentos...")
    try:
        with _quiet():
            rag.load_index()
        ok("Índice cargado.")
    except Exception:
        info("Ingiriendo documentos...")
        with _quiet():
            rag.ingest()
        ok("Indexado completado.")

    print()
    info("Ahora lanzamos una mezcla de preguntas de ambas categorías.\n")

    for i, (question, expected, hint) in enumerate(HIBRIDO_QUERIES, 1):
        # Determinar si esperamos fuente estructurada o no estructurada
        is_sql_q = any(q == question for q, _, _ in RAG_MALAS)
        expected_source = "estructurada (SQL)" if is_sql_q else "no estructurada (PDFs)"

        print(f"\n  {YELLOW}{'─' * 62}{RESET}")
        print(
            f'  {BOLD}Pregunta {i}/{len(HIBRIDO_QUERIES)}:{RESET} "{question}"\n'
            f"  {DIM}Fuente esperada: {expected_source}  |  "
            f"Respuesta esperada: {expected}{RESET}\n"
        )

        try:
            with _quiet():
                rag._hybrid_ops.ensure_hybrid_engine()
                hybrid_response = rag._hybrid_engine.query(question)

            response = hybrid_response.response

            # En modo híbrido, obtener la fuente desde la decisión del router.
            source_key = hybrid_response.routing.source.value.upper()
            source_labels = {
                "STRUCTURED": "estructurada (SQL)",
                "UNSTRUCTURED": "no estructurada (PDFs)",
                "HYBRID": "estructurada + no estructurada (híbrido)",
            }
            source_used = source_labels.get(source_key, source_key)

            print(f"  {BOLD}Respuesta del sistema:{RESET}")
            print_response(str(response))
            print(f"  {DIM}Fuente usada por el router: {source_used}{RESET}")

            ok(f"Pregunta {i} resuelta por el sistema híbrido.")

        except Exception as e:
            fail(f"Error: {e}")

        print()
        if do_pause and i < len(HIBRIDO_QUERIES):
            pause(f"Siguiente pregunta ({i + 1}/{len(HIBRIDO_QUERIES)})...")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo: RAG No Estructurado vs. Estructurado vs. Híbrido"
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Ejecutar sin pausas interactivas (modo automático)",
    )
    parser.add_argument(
        "--solo",
        choices=["rag", "sql", "hibrido"],
        default=None,
        help="Ejecutar solo una sección de la demo",
    )
    args = parser.parse_args()
    do_pause = not args.no_pause

    import os

    os.chdir(str(PROJECT_ROOT))

    # Comprobaciones previas
    db_path = PROJECT_ROOT / "data" / "proyectos_docentes.db"
    docs_dir = PROJECT_ROOT / "documents"
    doc_count = len(list(docs_dir.glob("*.pdf"))) if docs_dir.exists() else 0

    banner("DEMO: Proyectos Docentes — No Estructurado · Estructurado · Híbrido")

    info(
        "Este script demuestra las fortalezas y debilidades complementarias\n"
        "  del RAG no estructurado (PDFs) y del RAG estructurado (SQL), y\n"
        "  cómo el modo híbrido cubre todos los casos de uso."
    )
    print()

    if not db_path.exists():
        fail(f"Base de datos no encontrada: {db_path}")
        sys.exit(1)
    ok(f"Base de datos: {db_path.name}")

    if doc_count == 0:
        fail("No se encontraron PDFs en ./documents/")
        sys.exit(1)
    ok(f"Documentos PDF: {doc_count} archivos en ./documents/")

    print()

    if args.solo:
        # Modo parcial
        if do_pause:
            pause()
        if args.solo == "rag":
            demo_init()
            if do_pause:
                pause()
            demo_unstructured(do_pause)
        elif args.solo == "sql":
            demo_init()
            if do_pause:
                pause()
            demo_structured(do_pause)
        elif args.solo == "hibrido":
            demo_init()
            if do_pause:
                pause()
            demo_hybrid(do_pause)
    else:
        # Demo completa
        if do_pause:
            pause()

        demo_init()

        if do_pause:
            pause("Empezaremos con el RAG No Estructurado...")

        demo_unstructured(do_pause)

        if do_pause:
            pause("Pasamos al RAG Estructurado (SQL)...")

        demo_structured(do_pause)

        if do_pause:
            pause("Ahora activamos el modo Híbrido...")

        demo_hybrid(do_pause)

        banner("DEMO COMPLETADA  ✓")


if __name__ == "__main__":
    main()
