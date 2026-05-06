#!/usr/bin/env python3
"""
RAG Framework Usage Examples
============================

This file demonstrates various ways to use the RAG Framework.

Run examples:
    python examples/usage_examples.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_basic_usage():
    """Basic usage with default configuration."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage (Default Configuration)")
    print("=" * 60)

    from rag_framework import RAGFramework

    # Create framework with default config (uses rag_config.yaml)
    rag = RAGFramework()

    # Ingest documents from default directory
    rag.ingest()

    # Query
    response = rag.query("¿Cuál es el contenido principal de los documentos?")
    print(f"Response: {response}")


def example_custom_yaml():
    """Usage with custom YAML configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Custom YAML Configuration")
    print("=" * 60)

    from rag_framework import RAGFramework

    # Load from custom config file
    rag = RAGFramework.from_yaml("config/huggingface.yaml")

    # Ingest and query
    rag.ingest()
    response = rag.query("What is this about?")
    print(f"Response: {response}")


def example_programmatic_config():
    """Usage with programmatic configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Programmatic Configuration")
    print("=" * 60)

    from rag_framework import (
        RAGFramework,
        RAGConfig,
        LLMConfig,
        EmbeddingConfig,
        VectorStoreConfig,
        RetrievalConfig,
    )
    from rag_framework.config.models import (
        ChunkingConfig,
        DirectoryConfig,
        RerankerConfig,
    )

    # Create custom configuration
    config = RAGConfig(
        llm=LLMConfig(
            provider="ollama",
            model="llama3-instruct-8k",
            temperature=0.0,
            max_tokens=1024,
        ),
        embedding=EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text:v1.5",
        ),
        vector_store=VectorStoreConfig(
            provider="lancedb",
            persist_directory="./my_vectors",
            collection_name="my_docs",
        ),
        chunking=ChunkingConfig(
            chunk_size=1024,
            chunk_overlap=100,
        ),
        retrieval=RetrievalConfig(
            use_hybrid_search=True,
            top_k=5,
            reranker=RerankerConfig(
                enabled=True,
                model="BAAI/bge-reranker-base",
                top_n=3,
            ),
        ),
        directories=DirectoryConfig(
            documents_dir="./my_documents",
            vector_store_dir="./my_vectors",
        ),
        prompt_template="default",
        debug=False,
    )

    rag = RAGFramework(config)
    print("Configuration created successfully!")


def example_huggingface_models():
    """Usage with HuggingFace models (for offline/local use)."""
    print("\n" + "=" * 60)
    print("Example 4: HuggingFace Models (Local)")
    print("=" * 60)

    from rag_framework import (
        RAGFramework,
        RAGConfig,
        LLMConfig,
        EmbeddingConfig,
    )

    # Configuration for local HuggingFace models
    config = RAGConfig(
        llm=LLMConfig(
            provider="huggingface",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            # Or use a locally downloaded model:
            # local_model_path="./models/llm/TinyLlama--TinyLlama-1.1B-Chat-v1.0",
            device="auto",  # Will use CUDA if available
        ),
        embedding=EmbeddingConfig(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            # Or use a locally downloaded model:
            # local_model_path="./models/embedding/sentence-transformers--all-MiniLM-L6-v2",
            device="auto",
        ),
    )

    print("HuggingFace configuration created!")
    print("To download models locally, run:")
    print("  python run_rag.py download llm tiny-llama")
    print("  python run_rag.py download embedding all-MiniLM-L6-v2")


def example_prompt_templates():
    """Working with different prompt templates."""
    print("\n" + "=" * 60)
    print("Example 5: Prompt Templates")
    print("=" * 60)

    from rag_framework import RAGFramework, PromptTemplates

    # List available templates
    print("\nAvailable templates:")
    for name, desc in PromptTemplates.list_templates().items():
        print(f"  - {name}: {desc[:50]}...")

    # Create framework and change template
    rag = RAGFramework()

    # Change to conversational Spanish
    rag.set_prompt_template("conversational_es")

    # Or set a custom template
    custom_prompt = """
    Contexto: {context_str}
    
    Pregunta: {query_str}
    
    Por favor responde de forma breve y directa:
    """
    rag.set_custom_prompt(custom_prompt)

    print("\nCustom template set!")


def example_debug_mode():
    """Using debug mode to see retrieved chunks."""
    print("\n" + "=" * 60)
    print("Example 6: Debug Mode")
    print("=" * 60)

    from rag_framework import RAGFramework

    # Create with default config
    rag = RAGFramework()

    # Enable debug mode to see retrieved chunks
    rag.config.debug = True

    print("Debug mode enabled!")
    print("Now when you query, you'll see the retrieved chunks before the response.")


def example_validate_models():
    """Validate model availability."""
    print("\n" + "=" * 60)
    print("Example 7: Validate Models")
    print("=" * 60)

    from rag_framework import RAGFramework

    rag = RAGFramework()

    # Check if configured models are available
    is_valid = rag.validate_models()

    if is_valid:
        print("All models are available and ready!")
    else:
        print("Some models may not be available.")
        print("For Ollama models, run: ollama pull <model_name>")


def main():
    """Run all examples (demonstration only)."""
    print("=" * 60)
    print("RAG FRAMEWORK - USAGE EXAMPLES")
    print("=" * 60)
    print("\nThese examples demonstrate how to use the RAG Framework.")
    print("Some examples require Ollama to be running with the appropriate models.")
    print("\nTo run a specific example, uncomment it in the main() function.\n")

    # Uncomment the examples you want to run:

    # example_basic_usage()
    # example_custom_yaml()
    example_programmatic_config()
    example_huggingface_models()
    example_prompt_templates()
    example_debug_mode()
    # example_validate_models()  # Requires Ollama running


if __name__ == "__main__":
    main()
