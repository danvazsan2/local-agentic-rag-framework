"""
Download BAAI/bge-reranker-v2-m3 from HuggingFace to the local models directory.
Run once before using config/rtx4060_candidates.yaml.

Usage:
    conda activate sprint5
    python download_reranker_v2_m3.py
"""

from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "BAAI/bge-reranker-v2-m3"
LOCAL_DIR = Path("./models/bge-reranker-v2-m3")

print(f"Downloading {MODEL_ID} → {LOCAL_DIR.resolve()}")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(LOCAL_DIR),
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)
print(f"\nDone. Model saved to: {LOCAL_DIR.resolve()}")
print("To use the local copy, set in your YAML:")
print(f"  reranker:")
print(f"    provider: huggingface")
print(f"    model: BAAI/bge-reranker-v2-m3")
print(f"    local_model_path: {LOCAL_DIR}")
