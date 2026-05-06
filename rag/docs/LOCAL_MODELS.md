# Local Models Guide

This guide explains how to download and use HuggingFace models locally in the RAG framework.

## Table of Contents

1. [Overview](#overview)
2. [Downloading Models](#downloading-models)
3. [Configuration](#configuration)
4. [Supported Model Types](#supported-model-types)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)

## Overview

The RAG framework supports both remote and local models from HuggingFace. Local models offer several advantages:

- **Offline usage**: No internet connection required
- **Privacy**: Models run entirely on your machine
- **Speed**: No network latency
- **Cost**: No API costs
- **Reproducibility**: Fixed model versions

## Downloading Models

### Using the CLI Tool

The framework includes a command-line tool to download models easily:

```bash
# List all popular models
python run_rag.py download --list-popular

# Download using shortcuts
python run_rag.py download llm tiny-llama
python run_rag.py download embedding all-MiniLM-L6-v2
python run_rag.py download reranker bge-reranker-base

# Download using full HuggingFace model IDs
python run_rag.py download llm TinyLlama/TinyLlama-1.1B-Chat-v1.0
python run_rag.py download embedding sentence-transformers/all-MiniLM-L6-v2
python run_rag.py download reranker BAAI/bge-reranker-base
```

### Advanced Options

```bash
# Custom output directory
python run_rag.py download llm tiny-llama --output ./custom/path

# Private/gated models (requires HuggingFace token)
python run_rag.py download llm meta-llama/Llama-2-7b-chat-hf --token YOUR_HF_TOKEN

# Or set token as environment variable
export HF_TOKEN=your_token_here
python run_rag.py download llm meta-llama/Llama-2-7b-chat-hf
```

### Download Locations

By default, models are downloaded to:
- LLMs: `./models/llm/<model_name>`
- Embeddings: `./models/embedding/<model_name>`
- Rerankers: `./models/reranker/<model_name>`

## Configuration

### Using rag_config.yaml

After downloading models, update your configuration:

```yaml
# Local LLM
llm:
  provider: huggingface
  local_model_path: "./models/llm/TinyLlama--TinyLlama-1.1B-Chat-v1.0"
  device: auto

# Local Embedding
embedding:
  provider: huggingface
  local_model_path: "./models/embedding/sentence-transformers--all-MiniLM-L6-v2"
  device: auto

# Local Reranker
retrieval:
  reranker:
    enabled: true
    local_model_path: "./models/reranker/BAAI--bge-reranker-base"
    device: auto
```

### Device Selection

The `device` parameter determines where the model runs:

- `auto`: Automatically selects best available (CUDA > MPS > CPU)
- `cuda`: Force GPU (NVIDIA)
- `mps`: Force GPU (Apple Silicon)
- `cpu`: Force CPU

### Example Configuration File

See `config/local_models.yaml` for a complete example configuration using only local models.

## Supported Model Types

### 1. LLMs (Large Language Models)

**Popular Options:**

| Shortcut | Model ID | Size | Description |
|----------|----------|------|-------------|
| tiny-llama | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | ~1.1B | Small, fast, good for testing |
| phi-2 | microsoft/phi-2 | ~2.7B | Efficient small model |
| mistral-7b-instruct | mistralai/Mistral-7B-Instruct-v0.2 | ~7B | High quality medium model |
| llama-2-7b-chat | meta-llama/Llama-2-7b-chat-hf | ~7B | Meta's LLama 2 (requires token) |

**Download Examples:**
```bash
python run_rag.py download llm tiny-llama
python run_rag.py download llm microsoft/phi-2
python run_rag.py download llm meta-llama/Llama-2-7b-chat-hf --token YOUR_TOKEN
```

### 2. Embedding Models

**Popular Options:**

| Shortcut | Model ID | Dimensions | Description |
|----------|----------|------------|-------------|
| all-MiniLM-L6-v2 | sentence-transformers/all-MiniLM-L6-v2 | 384 | Small, fast, good quality |
| all-mpnet-base-v2 | sentence-transformers/all-mpnet-base-v2 | 768 | Medium, better quality |
| bge-large-en | BAAI/bge-large-en-v1.5 | 1024 | Large, best quality |
| multilingual-MiniLM | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | Multilingual support |

**Download Examples:**
```bash
python run_rag.py download embedding all-MiniLM-L6-v2
python run_rag.py download embedding BAAI/bge-large-en-v1.5
```

### 3. Reranker Models

**Popular Options:**

| Shortcut | Model ID | Description |
|----------|----------|-------------|
| bge-reranker-base | BAAI/bge-reranker-base | Base model, good balance |
| bge-reranker-large | BAAI/bge-reranker-large | Larger, better quality |
| bge-reranker-v2-m3 | BAAI/bge-reranker-v2-m3 | Latest version, multilingual |

**Download Examples:**
```bash
python run_rag.py download reranker bge-reranker-base
python run_rag.py download reranker BAAI/bge-reranker-large
```

## Examples

### Complete Workflow

1. **Download all models:**
```bash
# Small, fast setup (recommended for testing)
python run_rag.py download llm tiny-llama
python run_rag.py download embedding all-MiniLM-L6-v2
python run_rag.py download reranker bge-reranker-base
```

2. **Update configuration:**
```yaml
llm:
  provider: huggingface
  local_model_path: "./models/llm/TinyLlama--TinyLlama-1.1B-Chat-v1.0"
  device: auto

embedding:
  provider: huggingface
  local_model_path: "./models/embedding/sentence-transformers--all-MiniLM-L6-v2"
  device: auto

retrieval:
  reranker:
    enabled: true
    local_model_path: "./models/reranker/BAAI--bge-reranker-base"
    device: auto
```

3. **Run the RAG system:**
```bash
python run_rag.py --config config/local_models.yaml chat
```

### Mixed Configuration

You can also mix local and remote models:

```yaml
# Local LLM (for offline)
llm:
  provider: huggingface
  local_model_path: "./models/llm/TinyLlama--TinyLlama-1.1B-Chat-v1.0"
  device: auto

# Ollama embedding (if Ollama is running)
embedding:
  provider: ollama
  model: "nomic-embed-text:v1.5"

# Remote reranker (will download on first use)
retrieval:
  reranker:
    enabled: true
    model: "BAAI/bge-reranker-base"
    device: auto
```

## Troubleshooting

### Model Not Found Error

**Problem:** "Local model not found at: ..."

**Solution:**
1. Verify the model path is correct
2. Make sure you downloaded the model first
3. Check that the directory contains model files (`.bin`, `.safetensors`, `config.json`)

### CUDA Out of Memory

**Problem:** "CUDA out of memory"

**Solutions:**
1. Use a smaller model (e.g., TinyLlama instead of Llama-2-7B)
2. Force CPU: `device: cpu`
3. Reduce batch sizes in configuration

### Slow Performance on CPU

**Problem:** Model inference is very slow

**Solutions:**
1. Use GPU if available: `device: cuda` or `device: mps`
2. Choose smaller models (all-MiniLM-L6-v2 instead of bge-large)
3. Reduce `context_window` and `max_tokens` in LLM config

### Download Fails

**Problem:** Download interrupted or fails

**Solutions:**
1. Check internet connection
2. For gated models (Llama-2), ensure you have accepted the license and provided a valid token
3. Try downloading again - HuggingFace has resume capability
4. Check disk space

### Import Errors

**Problem:** "No module named 'transformers'" or similar

**Solution:**
```bash
pip install transformers torch sentence-transformers
```

## Hardware Requirements

### Minimum Requirements

- **CPU**: Any modern CPU
- **RAM**: 8GB minimum
- **Disk**: ~5-20GB per model

### Recommended Requirements

- **GPU**: NVIDIA GPU with 6GB+ VRAM or Apple Silicon
- **RAM**: 16GB+
- **Disk**: SSD recommended

### Model Size Guidelines

| Model Size | VRAM Needed | RAM (CPU) |
|------------|-------------|-----------|
| 1B params | 2-4 GB | 4-8 GB |
| 3B params | 6-8 GB | 12-16 GB |
| 7B params | 14-16 GB | 28-32 GB |

## Additional Resources

- [HuggingFace Model Hub](https://huggingface.co/models)
- [Sentence Transformers](https://www.sbert.net/)
- [BGE Models](https://github.com/FlagOpen/FlagEmbedding)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
