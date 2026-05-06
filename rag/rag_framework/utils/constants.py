"""
Centralized constants for the RAG Framework.

All default values and configuration constants should be defined here
to ensure consistency across the entire system.
"""

# =============================================================================
# FRAMEWORK VERSION
# =============================================================================
FRAMEWORK_VERSION = "1.1.0"
FRAMEWORK_NAME = "RAG Framework"

# =============================================================================
# DEFAULT MODEL CONFIGURATION
# =============================================================================

# LLM defaults (Ollama)
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_LLM_MODEL = "llama3-instruct-8k"
DEFAULT_LLM_CONTEXT_WINDOW = 8192
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_TOP_P = 0.9
DEFAULT_LLM_TOP_K = 40
DEFAULT_LLM_MAX_TOKENS = 512
DEFAULT_LLM_REPEAT_PENALTY = 1.15
DEFAULT_LLM_TIMEOUT = 120.0

# Embedding defaults (Ollama)
DEFAULT_EMBEDDING_PROVIDER = "ollama"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:v1.5"

# Reranker defaults (HuggingFace)
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANKER_TOP_N = 3

# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# =============================================================================
# VECTOR STORE CONFIGURATION
# =============================================================================
DEFAULT_VECTOR_STORE_PROVIDER = "lancedb"
DEFAULT_VECTOR_STORE_DIR = "./vector_store"
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_LANCE_MODE = "overwrite"

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================
DEFAULT_USE_HYBRID_SEARCH = True
DEFAULT_TOP_K = 5
DEFAULT_HYBRID_ALPHA = 0.5  # 0=only BM25, 1=only vector
DEFAULT_RRF_K = 60  # Reciprocal Rank Fusion constant

# =============================================================================
# API SERVER CONFIGURATION
# =============================================================================
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8765
DEFAULT_SESSIONS_DIR = "./sessions"

# =============================================================================
# DIRECTORIES
# =============================================================================
DEFAULT_DOCUMENTS_DIR = "./documents"
DEFAULT_MODELS_DIR = "./models"

# =============================================================================
# PROMPT CONFIGURATION
# =============================================================================
DEFAULT_PROMPT_TEMPLATE = "default"

# =============================================================================
# SUPPORTED FILE EXTENSIONS
# =============================================================================
SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".html",
    ".md",
    ".txt",
    ".xlsx",
    ".csv",
    ".json",
}

# Extensions that require Docling
DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".md", ".xlsx", ".csv"}

# Extensions for simple reader
SIMPLE_READER_EXTENSIONS = {".txt", ".json"}

# =============================================================================
# LLM STOP SEQUENCES (Spanish context)
# =============================================================================
DEFAULT_STOP_SEQUENCES = [
    "\n\nPregunta:",
    "\n\nUsuario:",
    "\n\nContexto:",
    "\n\nQuestion:",
    "\n\nUser:",
]

# =============================================================================
# POPULAR MODELS (for `run_rag.py download`)
# =============================================================================
POPULAR_MODELS = {
    "llm": {
        "llama3-instruct-8k": "meta-llama/Llama-3-8B-Instruct",
        "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
        "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
        "phi-2": "microsoft/phi-2",
        "tiny-llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    },
    "embedding": {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "bge-large-en": "BAAI/bge-large-en-v1.5",
        "multilingual-MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
    "reranker": {
        "bge-reranker-base": "BAAI/bge-reranker-base",
        "bge-reranker-large": "BAAI/bge-reranker-large",
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    },
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
VERBOSE_LOGGERS = [
    "httpx",
    "httpcore",
    "urllib3",
    "llama_index",
    "transformers",
    "sentence_transformers",
    "chromadb",
    "lancedb",
    "bm25s",
]
