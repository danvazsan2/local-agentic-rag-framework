"""
Model Downloader
=================

Functions to download and cache HuggingFace models (LLMs, embeddings,
rerankers) for offline use in the RAG system.
"""

from pathlib import Path
from typing import Optional

from rag_framework.utils.constants import POPULAR_MODELS


def _get_models_dir() -> Path:
    """Get the default models directory (project root / models)."""
    return Path(__file__).resolve().parent.parent.parent / "models"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_llm(
    model_id: str,
    output_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[str]:
    """Download a LLM from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g. ``meta-llama/Llama-2-7b-chat-hf``).
        output_dir: Custom output directory. Defaults to ``models/llm/<name>``.
        token: HuggingFace token for gated / private models.

    Returns:
        Path to the downloaded model, or *None* on failure.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if output_dir:
        local_dir = Path(output_dir)
    else:
        model_name = model_id.replace("/", "--")
        local_dir = _get_models_dir() / "llm" / model_name

    print(f"Downloading LLM: {model_id}")
    print(f"Saving to: {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        tokenizer.save_pretrained(str(local_dir))
        print("   Tokenizer saved")

        print("   Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, torch_dtype="auto", low_cpu_mem_usage=True
        )
        model.save_pretrained(str(local_dir))
        print("   Model saved")

        print(f"\nLLM downloaded successfully to: {local_dir}")
        print(f"\nTo use this model, update your rag_config.yaml:")
        print(f"   llm:")
        print(f"     provider: huggingface")
        print(f'     local_model_path: "{local_dir}"')
        return str(local_dir)

    except Exception as e:
        print(f"\nError downloading model: {e}")
        print("\nCommon issues:")
        print("   - Make sure you have internet connection")
        print("   - For gated models (like Llama-2), you need a HuggingFace token")
        print("   - Run: pip install transformers torch accelerate")
        return None


def download_embedding(
    model_id: str,
    output_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[str]:
    """Download an embedding model from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g. ``sentence-transformers/all-MiniLM-L6-v2``).
        output_dir: Custom output directory. Defaults to ``models/embedding/<name>``.
        token: HuggingFace token for gated / private models.

    Returns:
        Path to the downloaded model, or *None* on failure.
    """
    from sentence_transformers import SentenceTransformer

    if output_dir:
        local_dir = Path(output_dir)
    else:
        model_name = model_id.replace("/", "--")
        local_dir = _get_models_dir() / "embedding" / model_name

    print(f"Downloading Embedding Model: {model_id}")
    print(f"Saving to: {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("   Downloading model...")
        model = SentenceTransformer(model_id, use_auth_token=token)
        model.save(str(local_dir))
        print("   Model saved")

        print(f"\nEmbedding model downloaded successfully to: {local_dir}")
        print(f"\nTo use this model, update your rag_config.yaml:")
        print(f"   embedding:")
        print(f"     provider: huggingface")
        print(f'     local_model_path: "{local_dir}"')
        return str(local_dir)

    except Exception as e:
        print(f"\nError downloading model: {e}")
        print("\nMake sure you have internet connection and run:")
        print("   pip install sentence-transformers")
        return None


def download_reranker(
    model_id: str,
    output_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[str]:
    """Download a reranker model from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g. ``BAAI/bge-reranker-base``).
        output_dir: Custom output directory. Defaults to ``models/reranker/<name>``.
        token: HuggingFace token for gated / private models.

    Returns:
        Path to the downloaded model, or *None* on failure.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if output_dir:
        local_dir = Path(output_dir)
    else:
        model_name = model_id.replace("/", "--")
        local_dir = _get_models_dir() / "reranker" / model_name

    print(f"Downloading Reranker Model: {model_id}")
    print(f"Saving to: {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        tokenizer.save_pretrained(str(local_dir))
        print("   Tokenizer saved")

        print("   Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, token=token
        )
        model.save_pretrained(str(local_dir))
        print("   Model saved")

        print(f"\nReranker model downloaded successfully to: {local_dir}")
        print(f"\nTo use this model, update your rag_config.yaml:")
        print(f"   retrieval:")
        print(f"     reranker:")
        print(f"       enabled: true")
        print(f'       local_model_path: "{local_dir}"')
        return str(local_dir)

    except Exception as e:
        print(f"\nError downloading model: {e}")
        print("\nMake sure you have internet connection and run:")
        print("   pip install transformers torch")
        return None


# ---------------------------------------------------------------------------
# Listing helper
# ---------------------------------------------------------------------------


def list_popular_models() -> None:
    """Print a table of popular models for each category."""
    print("\n" + "=" * 60)
    print("POPULAR MODELS")
    print("=" * 60)

    print("\nLLMs (Large Language Models):")
    for name, mid in POPULAR_MODELS["llm"].items():
        print(f"   {name:20s} → {mid}")

    print("\nEmbedding Models:")
    for name, mid in POPULAR_MODELS["embedding"].items():
        print(f"   {name:20s} → {mid}")

    print("\nReranker Models:")
    for name, mid in POPULAR_MODELS["reranker"].items():
        print(f"   {name:20s} → {mid}")

    print("\n" + "=" * 60)
    print("\nUsage examples:")
    print("   python run_rag.py download llm tiny-llama")
    print("   python run_rag.py download embedding all-MiniLM-L6-v2")
    print("   python run_rag.py download reranker bge-reranker-base")
    print("\n   Or use full HuggingFace model IDs:")
    print("   python run_rag.py download llm meta-llama/Llama-2-7b-chat-hf")


# ---------------------------------------------------------------------------
# Dispatcher (called from run_rag.py)
# ---------------------------------------------------------------------------


def run_download(args) -> None:
    """Execute a download action based on parsed *argparse* args.

    Expected attributes on *args*:
        list_popular (bool), model_type (str | None),
        model_id (str | None), output (str | None), token (str | None).
    """
    import os

    if args.list_popular:
        list_popular_models()
        return

    if not args.model_type or not args.model_id:
        print("Error: both model_type and model_id are required.")
        print("Run  python run_rag.py download --help  for details.")
        raise SystemExit(1)

    token = args.token or os.getenv("HF_TOKEN")

    # Resolve shortcut names
    model_id = args.model_id
    if args.model_id in POPULAR_MODELS.get(args.model_type, {}):
        model_id = POPULAR_MODELS[args.model_type][args.model_id]
        print(f"Using popular model: {args.model_id} → {model_id}")

    downloaders = {
        "llm": download_llm,
        "embedding": download_embedding,
        "reranker": download_reranker,
    }
    downloaders[args.model_type](model_id, args.output, token)
