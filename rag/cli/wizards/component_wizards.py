"""
Component Configuration Wizards
=================================

Interactive wizards for configuring individual RAG components.
"""

from typing import Tuple, Optional

from rag_framework.config.models import (
    LLMConfig,
    EmbeddingConfig,
    RerankerConfig,
    VectorStoreConfig,
    DirectoryConfig,
    SQLConfig,
    DatabaseConnectionConfig,
    LLMProvider,
    EmbeddingProvider,
    VectorStoreProvider,
)
from rag_framework.prompts import PromptTemplates

from ..ui import get_input, get_choice, confirm, print_header
from ..discovery import discover_local_models, discover_documents, discover_databases


def configure_llm() -> LLMConfig:
    """Interactive LLM configuration wizard.

    Returns:
        Configured LLMConfig instance
    """
    print_header("Configuración del LLM (Modelo de Lenguaje)")

    # Provider selection
    providers = [p.value for p in LLMProvider]
    print("Proveedores disponibles:")
    for i, provider in enumerate(providers, 1):
        print(f"  {i}. {provider}")

    provider_choice = get_choice("Selecciona proveedor", providers, 1)
    provider = providers[provider_choice - 1]

    config_data = {"provider": provider}

    if provider == "ollama":
        config_data["model"] = get_input("Modelo", "llama3-instruct-8k")
        config_data["base_url"] = get_input(
            "URL base de Ollama", "http://localhost:11434"
        )

    elif provider == "huggingface":
        # Check for local models
        local_models = discover_local_models()["llm"]

        if local_models:
            print("\nModelos locales encontrados:")
            for i, model_path in enumerate(local_models, 1):
                print(f"  {i}. {model_path}")
            print(f"  {len(local_models) + 1}. Usar modelo de HuggingFace Hub")

            model_choice = get_choice(
                "Selecciona modelo", local_models + ["HuggingFace Hub"], 1
            )

            if model_choice <= len(local_models):
                config_data["local_model_path"] = local_models[model_choice - 1]
                config_data["is_local"] = True
            else:
                config_data["hf_model_id"] = get_input(
                    "ID del modelo en HuggingFace", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                )
        else:
            config_data["hf_model_id"] = get_input(
                "ID del modelo en HuggingFace", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )

        config_data["device"] = get_input("Dispositivo (auto/cuda/cpu/mps)", "auto")

    # Advanced parameters
    if confirm("¿Configurar parámetros avanzados de generación?", default=False):
        config_data["temperature"] = float(get_input("Temperature (0.0-1.0)", "0.0"))
        config_data["max_tokens"] = int(get_input("Max tokens", "512"))
        config_data["top_p"] = float(get_input("Top P (0.0-1.0)", "0.9"))
        config_data["context_window"] = int(get_input("Context window", "8192"))

    return LLMConfig(**config_data)


def configure_embedding() -> EmbeddingConfig:
    """Interactive embedding configuration wizard.

    Returns:
        Configured EmbeddingConfig instance
    """
    print_header("Configuración de Embeddings")

    providers = [p.value for p in EmbeddingProvider]
    print("Proveedores disponibles:")
    for i, provider in enumerate(providers, 1):
        print(f"  {i}. {provider}")

    provider_choice = get_choice("Selecciona proveedor", providers, 1)
    provider = providers[provider_choice - 1]

    config_data = {"provider": provider}

    if provider == "ollama":
        config_data["model"] = get_input("Modelo", "nomic-embed-text:v1.5")
        config_data["base_url"] = get_input(
            "URL base de Ollama", "http://localhost:11434"
        )

    elif provider == "huggingface":
        recommended_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]

        print("\nModelos recomendados:")
        for i, model in enumerate(recommended_models, 1):
            print(f"  {i}. {model}")
        print(f"  {len(recommended_models) + 1}. Especificar otro")

        model_choice = get_choice("Selecciona modelo", recommended_models + ["Otro"], 1)

        if model_choice <= len(recommended_models):
            config_data["hf_model_id"] = recommended_models[model_choice - 1]
        else:
            config_data["hf_model_id"] = get_input("ID del modelo en HuggingFace")

        config_data["device"] = get_input("Dispositivo (auto/cuda/cpu/mps)", "auto")

    return EmbeddingConfig(**config_data)


def configure_reranker() -> RerankerConfig:
    """Interactive reranker configuration wizard.

    Returns:
        Configured RerankerConfig instance
    """
    print_header("Configuración del Reranker")

    enabled = confirm("¿Habilitar reranker?", default=True)

    if not enabled:
        return RerankerConfig(enabled=False)

    config_data = {"enabled": True}

    # Check for local reranker
    local_rerankers = discover_local_models()["reranker"]

    if local_rerankers:
        print("\nRerankers locales encontrados:")
        for i, path in enumerate(local_rerankers, 1):
            print(f"  {i}. {path}")
        print(f"  {len(local_rerankers) + 1}. Descargar de HuggingFace")

        choice = get_choice("Selecciona reranker", local_rerankers + ["HuggingFace"], 1)

        if choice <= len(local_rerankers):
            config_data["local_model_path"] = local_rerankers[choice - 1]
            config_data["model"] = "BAAI/bge-reranker-base"
        else:
            config_data["model"] = get_input("Modelo", "BAAI/bge-reranker-base")
    else:
        config_data["model"] = get_input("Modelo", "BAAI/bge-reranker-base")

    config_data["top_n"] = int(get_input("Top N resultados", "5"))
    config_data["device"] = get_input("Dispositivo (auto/cuda/cpu/mps)", "auto")

    return RerankerConfig(**config_data)


def configure_vector_store() -> VectorStoreConfig:
    """Interactive vector store configuration wizard.

    Returns:
        Configured VectorStoreConfig instance
    """
    print_header("Configuración del Vector Store")

    providers = [p.value for p in VectorStoreProvider]
    print("Proveedores disponibles:")
    for i, provider in enumerate(providers, 1):
        desc = {
            "lancedb": "(Recomendado - Local, rápido)",
            "chroma": "(Local/Remoto)",
            "faiss": "(Local, Meta)",
            "qdrant": "(Cloud/Local)",
            "pinecone": "(Cloud)",
        }.get(provider, "")
        print(f"  {i}. {provider} {desc}")

    provider_choice = get_choice("Selecciona proveedor", providers, 1)
    provider = providers[provider_choice - 1]

    config_data = {
        "provider": provider,
        "persist_directory": get_input("Directorio de persistencia", "./vector_store"),
        "collection_name": get_input("Nombre de colección", "documents"),
    }

    return VectorStoreConfig(**config_data)


def configure_directories() -> DirectoryConfig:
    """Interactive directory configuration wizard.

    Returns:
        Configured DirectoryConfig instance
    """
    print_header("Configuración de Directorios")

    # Show discovered documents
    doc_stats = discover_documents()
    if doc_stats:
        print("Documentos encontrados en ./documents:")
        for ext, count in doc_stats.items():
            print(f"  {ext}: {count} archivo(s)")
        print()

    return DirectoryConfig(
        documents_dir=get_input("Directorio de documentos", "./documents"),
        vector_store_dir=get_input("Directorio de vector store", "./vector_store"),
    )


def configure_prompt_template() -> Tuple[str, Optional[str]]:
    """Interactive prompt template configuration wizard.

    Returns:
        Tuple of (template_name, custom_prompt)
    """
    print_header("Configuración de Plantilla de Prompts")

    templates = PromptTemplates.list_templates()
    template_names = list(templates.keys())

    print("Plantillas disponibles:")
    for i, (name, desc) in enumerate(templates.items(), 1):
        print(f"  {i}. {name}")
        print(f"     {desc[:60]}...")

    print(f"  {len(template_names) + 1}. Plantilla personalizada")

    choice = get_choice("Selecciona plantilla", template_names + ["custom"], 1)

    if choice <= len(template_names):
        return template_names[choice - 1], None
    else:
        print("\nIntroduce tu plantilla personalizada.")
        print("Debe incluir {context_str} y {query_str}")
        print("Termina con una línea vacía:\n")

        lines = []
        while True:
            try:
                line = input()
                if line == "":
                    break
                lines.append(line)
            except EOFError:
                break

        custom_prompt = "\n".join(lines)
        return "custom", custom_prompt


def configure_sql() -> SQLConfig:
    """Interactive SQL configuration wizard.

    Returns:
        Configured SQLConfig instance
    """
    print_header("Configuración de SQL (Datos Estructurados)")

    enabled = confirm("¿Habilitar consultas SQL?", default=False)

    if not enabled:
        return SQLConfig(enabled=False)

    # Discover databases
    databases = discover_databases()

    config_data = {"enabled": True}
    connection_data = {"db_type": "sqlite"}

    if databases:
        print("\nBases de datos encontradas:")
        for i, db in enumerate(databases, 1):
            print(f"  {i}. {db}")
        print(f"  {len(databases) + 1}. Especificar otra ruta")

        choice = get_choice("Selecciona base de datos", databases + ["Otra"], 1)

        if choice <= len(databases):
            connection_data["sqlite_path"] = databases[choice - 1]
        else:
            connection_data["sqlite_path"] = get_input("Ruta a la base de datos SQLite")
    else:
        connection_data["sqlite_path"] = get_input(
            "Ruta a la base de datos SQLite", "./data/database.db"
        )

    config_data["connection"] = DatabaseConnectionConfig(**connection_data)

    return SQLConfig(**config_data)
