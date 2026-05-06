"""
System Discovery
=================

Functions to discover available models, configs, databases, and documents.
"""

from pathlib import Path
from typing import Dict, List


def discover_local_models() -> Dict[str, List[str]]:
    """Discover locally available models.

    Searches the ./models directory for LLM, embedding, and reranker models.

    Returns:
        Dictionary with keys 'llm', 'embedding', 'reranker' containing
        lists of model paths
    """
    models = {"llm": [], "embedding": [], "reranker": []}
    models_dir = Path("./models")

    if not models_dir.exists():
        return models

    # Check for LLM models
    llm_dir = models_dir / "llm"
    if llm_dir.exists():
        for item in llm_dir.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                models["llm"].append(str(item))

    # Check for embedding models
    embedding_dir = models_dir / "embedding"
    if embedding_dir.exists():
        for item in embedding_dir.iterdir():
            if item.is_dir():
                models["embedding"].append(str(item))

    # Check for reranker models
    for item in models_dir.iterdir():
        if item.is_dir() and "reranker" in item.name.lower():
            models["reranker"].append(str(item))

    return models


def discover_config_files() -> List[str]:
    """Discover existing configuration files.

    Searches for YAML configuration files in the current directory
    and the ./config directory.

    Returns:
        List of configuration file paths
    """
    config_patterns = ["*.yaml", "*.yml"]
    config_files = []

    # Check root directory
    for pattern in config_patterns:
        config_files.extend([str(p) for p in Path(".").glob(pattern)])

    # Check config directory
    config_dir = Path("./config")
    if config_dir.exists():
        for pattern in config_patterns:
            config_files.extend([str(p) for p in config_dir.glob(pattern)])

    return config_files


def discover_databases() -> List[str]:
    """Discover SQLite databases in the project.

    Searches for .db, .sqlite, and .sqlite3 files in the ./data
    directory and root directory.

    Returns:
        List of database file paths
    """
    databases = []
    data_dir = Path("./data")

    if data_dir.exists():
        databases.extend([str(p) for p in data_dir.glob("*.db")])
        databases.extend([str(p) for p in data_dir.glob("*.sqlite")])
        databases.extend([str(p) for p in data_dir.glob("*.sqlite3")])

    # Also check root
    databases.extend([str(p) for p in Path(".").glob("*.db")])

    return databases


def discover_documents() -> Dict[str, int]:
    """Discover documents in the documents directory.

    Counts documents by file extension in the ./documents directory.

    Returns:
        Dictionary mapping file extensions to document counts
    """
    docs_dir = Path("./documents")
    doc_stats = {}

    if not docs_dir.exists():
        return doc_stats

    extensions = [".txt", ".pdf", ".md", ".docx", ".html"]
    for ext in extensions:
        count = len(list(docs_dir.glob(f"*{ext}")))
        if count > 0:
            doc_stats[ext] = count

    return doc_stats
