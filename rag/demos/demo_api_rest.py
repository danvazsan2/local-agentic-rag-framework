#!/usr/bin/env python3
"""
=============================================================================
  DEMO: API REST del Framework RAG
=============================================================================

Demuestra la funcionalidad completa de la API REST:

  1. Health check              GET  /health
  2. Configuración activa      GET  /config
  3. Configuraciones dispo.    GET  /configs
  4. Crear sesión              POST /sessions
  5. Ingestar documentos       POST /ingest
  6. Consultar (query)         POST /query
  7. Limpiar sesión            POST /clear

Uso:
    # Terminal 1 — arrancar el servidor:
    python -m api.server            (o  python run_rag.py api)

    # Terminal 2 — ejecutar la demo:
    python demos/demo_api_rest.py

    # (Opcional) modo automático: arranca el servidor en segundo plano:
    python demos/demo_api_rest.py --auto

Requisitos:
    - requests  (pip install requests)
    - Servidor RAG levantado en http://localhost:8765  (o usar --auto)
    - Ollama corriendo con los modelos configurados en rag_config.yaml
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import requests

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

# ── Configuración por defecto ───────────────────────────────────────
BASE_URL = "http://localhost:8765"
SESSION_ID = f"demo-{uuid.uuid4().hex[:8]}"

# Documento de prueba pequeño (se crea en memoria)
SAMPLE_DOC_NAME = "demo_document.txt"
SAMPLE_DOC_CONTENT = textwrap.dedent(
    """\
    Documento de demostración del Framework RAG
    =============================================

    Este sistema permite crear pipelines de Retrieval-Augmented Generation
    (RAG) con soporte para múltiples proveedores de LLM y embeddings.

    Características principales:
    - Ingesta y chunking automático de documentos (PDF, TXT, MD, DOCX).
    - Búsqueda híbrida: vectorial + BM25.
    - Reranking con modelos cross-encoder.
    - Soporte para consultas SQL sobre bases de datos relacionales.
    - Router inteligente que decide si la consulta necesita documentos,
      base de datos o ambos.
    - API REST para integración con interfaces web.

    La API REST expone endpoints para gestionar sesiones independientes,
    cada una con su propio índice vectorial, permitiendo que múltiples
    usuarios o conversaciones funcionen en paralelo sin interferencias.

    El router utiliza el LLM para clasificar cada consulta en tres
    categorías: STRUCTURED (SQL), UNSTRUCTURED (documentos) o HYBRID
    (ambas fuentes combinadas).
"""
)

SAMPLE_QUERY = "¿Cuáles son las características principales del sistema RAG?"


# ════════════════════════════════════════════════════════════════════
# Utilidades de presentación
# ════════════════════════════════════════════════════════════════════


def banner(title: str) -> None:
    width = 64
    print(f"\n{CYAN}{'═' * width}")
    print(f"  {BOLD}{title}{RESET}{CYAN}")
    print(f"{'═' * width}{RESET}\n")


def step(number: int, title: str, method: str, endpoint: str) -> None:
    print(f"{YELLOW}{'─' * 64}{RESET}")
    print(f"  {BOLD}Paso {number}{RESET}  │  {title}")
    print(f"  {DIM}{method} {endpoint}{RESET}")
    print(f"{YELLOW}{'─' * 64}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓ {msg}{RESET}")


def fail(msg: str) -> None:
    print(f"  {RED}✗ {msg}{RESET}")


def info(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


def show_json(data: Any, indent: int = 4) -> None:
    """Muestra un JSON con colores."""
    formatted = json.dumps(data, indent=indent, ensure_ascii=False)
    for line in formatted.splitlines():
        print(f"  {BLUE}{line}{RESET}")


def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Espera hasta que el servidor responda o se agote el timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def pause(msg: str = "Pulsa Enter para continuar...") -> None:
    """Pausa interactiva entre pasos."""
    print()
    input(f"  {DIM}▸ {msg}{RESET}")
    print()


# ════════════════════════════════════════════════════════════════════
# Pasos de la demo
# ════════════════════════════════════════════════════════════════════


def demo_health(base_url: str) -> bool:
    step(1, "Health Check", "GET", "/health")
    info("Verifica que el servidor está activo y muestra versión + sesiones.\n")

    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        data = r.json()
        show_json(data)
        ok(
            f"Servidor activo — versión {data.get('version', '?')}, "
            f"{data.get('sessions', 0)} sesiones"
        )
        return True
    except Exception as e:
        fail(f"No se pudo conectar: {e}")
        return False


def demo_config(base_url: str) -> None:
    step(2, "Configuración activa", "GET", "/config")
    info("Devuelve la configuración de LLM, embedding y retrieval cargada.\n")

    r = requests.get(f"{base_url}/config", timeout=5)
    data = r.json()
    show_json(data)

    cfg = data.get("config", {})
    llm = cfg.get("llm", {})
    ok(f"LLM: {llm.get('provider')}/{llm.get('model')}")


def demo_list_configs(base_url: str) -> None:
    step(3, "Configuraciones disponibles", "GET", "/configs")
    info("Lista los archivos YAML de configuración encontrados.\n")

    r = requests.get(f"{base_url}/configs", timeout=5)
    data = r.json()
    show_json(data)

    configs = data.get("configs", [])
    ok(f"Se encontraron {len(configs)} configuraciones")
    for cfg in configs:
        info(f"  • {cfg.get('name')}  →  {cfg.get('path')}")


def demo_create_session(base_url: str, session_id: str) -> None:
    step(4, "Crear sesión", "POST", "/sessions")
    info(f"Crea una sesión aislada con ID: {session_id}")
    info("Cada sesión tiene su propio directorio de documentos y vector store.\n")

    payload = {"session_id": session_id}
    info(f"Payload: {json.dumps(payload)}\n")

    r = requests.post(f"{base_url}/sessions", json=payload, timeout=10)
    data = r.json()
    show_json(data)

    if data.get("success"):
        ok(f"Sesión '{session_id}' creada correctamente")
    else:
        # La sesión podría no existir como endpoint separado en todas las versiones
        info("Nota: la sesión se creará automáticamente al ingestar documentos")


def demo_ingest(base_url: str, session_id: str) -> None:
    step(5, "Ingestar documentos", "POST", "/ingest")
    info("Envía archivos en base64 → se guardan, indexan y crean el vector store.")
    info(f"Documento de prueba: {SAMPLE_DOC_NAME} ({len(SAMPLE_DOC_CONTENT)} chars)\n")

    encoded = base64.b64encode(SAMPLE_DOC_CONTENT.encode("utf-8")).decode("ascii")

    payload = {
        "session_id": session_id,
        "files": [{"name": SAMPLE_DOC_NAME, "content": encoded}],
    }

    info(f"Payload: session_id={session_id}, files=[{SAMPLE_DOC_NAME}]")
    info("Enviando... (la ingesta puede tardar unos segundos)\n")

    try:
        r = requests.post(f"{base_url}/ingest", json=payload, timeout=120)
        data = r.json()
        show_json(data)

        if data.get("success"):
            ok(f"Archivos ingestados: {data.get('files_ingested', [])}")
        else:
            fail(f"Error en ingesta: {data.get('error', 'desconocido')}")
    except requests.Timeout:
        fail("Timeout en la ingesta (puede requerir más tiempo o modelos más rápidos)")


def demo_query(base_url: str, session_id: str) -> None:
    step(6, "Consultar (Query)", "POST", "/query")
    info("Realiza una consulta RAG sobre los documentos ingestados.")
    info(f'Pregunta: "{SAMPLE_QUERY}"\n')

    payload = {
        "session_id": session_id,
        "query": SAMPLE_QUERY,
    }

    info("Enviando consulta...\n")

    try:
        r = requests.post(f"{base_url}/query", json=payload, timeout=120)
        data = r.json()

        if data.get("success"):
            print(f"  {MAGENTA}{'─' * 56}{RESET}")
            print(f"  {BOLD}Respuesta del sistema:{RESET}\n")
            response_text = data.get("response", "")
            for line in response_text.splitlines():
                print(f"    {line}")
            print(f"\n  {MAGENTA}{'─' * 56}{RESET}")
            ok("Consulta ejecutada correctamente")
        else:
            fail(f"Error en consulta: {data.get('error', 'desconocido')}")
            show_json(data)
    except requests.Timeout:
        fail("Timeout en la consulta")


def demo_clear(base_url: str, session_id: str) -> None:
    step(7, "Limpiar sesión", "POST", "/clear")
    info(f"Elimina la sesión '{session_id}' y todos sus datos.\n")

    payload = {"session_id": session_id}

    r = requests.post(f"{base_url}/clear", json=payload, timeout=10)
    data = r.json()
    show_json(data)

    if data.get("success"):
        ok("Sesión eliminada correctamente")
    else:
        info(f"Nota: {data.get('message', '')}")


# ════════════════════════════════════════════════════════════════════
# Orquestación principal
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo interactiva de la API REST del Framework RAG"
    )
    parser.add_argument(
        "--url", default=BASE_URL, help=f"URL base del servidor (default: {BASE_URL})"
    )
    parser.add_argument(
        "--session", default=SESSION_ID, help="ID de sesión a usar en la demo"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Arrancar el servidor automáticamente en segundo plano",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Ejecutar sin pausas interactivas (modo continuo)",
    )
    args = parser.parse_args()

    base_url = args.url
    session_id = args.session
    server_proc: Optional[subprocess.Popen] = None

    banner("DEMO: API REST — Framework RAG")
    print(f"  {DIM}Servidor:  {base_url}{RESET}")
    print(f"  {DIM}Sesión:    {session_id}{RESET}")
    print()

    # ── Auto-start del servidor ─────────────────────────────────────
    if args.auto:
        info("Arrancando servidor en segundo plano...")
        project_root = Path(__file__).resolve().parent.parent
        server_proc = subprocess.Popen(
            [sys.executable, "-c", "from api.server import run_server; run_server()"],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        info(f"Proceso del servidor PID: {server_proc.pid}")
        info("Esperando a que el servidor esté listo...")

        if wait_for_server(base_url):
            ok("Servidor listo")
        else:
            fail("El servidor no respondió a tiempo. ¿Está Ollama ejecutándose?")
            server_proc.terminate()
            sys.exit(1)
    else:
        info("Verificando conexión con el servidor...")
        if not wait_for_server(base_url, timeout=5):
            fail(f"No se pudo conectar a {base_url}")
            print(f"\n  {YELLOW}Asegúrate de que el servidor esté arrancado:{RESET}")
            print(f"    python run_rag.py api")
            print(f"  {YELLOW}O usa --auto para arrancarlo automáticamente.{RESET}\n")
            sys.exit(1)
        ok("Servidor detectado")

    do_pause = not args.no_pause

    try:
        # ── Paso 1: Health ──────────────────────────────────────────
        if do_pause:
            pause()
        if not demo_health(base_url):
            sys.exit(1)

        # ── Paso 2: Config ─────────────────────────────────────────
        if do_pause:
            pause()
        demo_config(base_url)

        # ── Paso 3: List configs ────────────────────────────────────
        if do_pause:
            pause()
        demo_list_configs(base_url)

        # ── Paso 4: Create session ──────────────────────────────────
        if do_pause:
            pause()
        demo_create_session(base_url, session_id)

        # ── Paso 5: Ingest ──────────────────────────────────────────
        if do_pause:
            pause()
        demo_ingest(base_url, session_id)

        # ── Paso 6: Query ──────────────────────────────────────────
        if do_pause:
            pause()
        demo_query(base_url, session_id)

        # ── Paso 7: Clear ──────────────────────────────────────────
        if do_pause:
            pause()
        demo_clear(base_url, session_id)

        # ── Resumen final ───────────────────────────────────────────
        banner("DEMO COMPLETADA")
        print(f"  {GREEN}Se han demostrado los 7 endpoints de la API REST:{RESET}")
        print(f"    GET  /health    — Estado del servidor")
        print(f"    GET  /config    — Configuración activa")
        print(f"    GET  /configs   — Configuraciones disponibles")
        print(f"    POST /sessions  — Creación de sesiones aisladas")
        print(f"    POST /ingest    — Ingesta de documentos (base64)")
        print(f"    POST /query     — Consultas RAG con LLM")
        print(f"    POST /clear     — Limpieza de sesiones")
        print()
        print(f"  {DIM}Cada sesión tiene su propio vector store aislado,")
        print(f"  permitiendo múltiples usuarios/conversaciones en paralelo.{RESET}\n")

    finally:
        if server_proc:
            info("Deteniendo servidor...")
            server_proc.terminate()
            server_proc.wait(timeout=5)
            ok("Servidor detenido")


if __name__ == "__main__":
    main()
