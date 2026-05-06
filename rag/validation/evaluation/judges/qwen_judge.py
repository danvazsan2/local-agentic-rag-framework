"""
Qwen judge implementation — placeholder for Phase 4.

Uses qwen3:8b via Ollama with temperature=0 and seed=42 for determinism.
Caches responses by hash(prompt + context + response) to avoid re-evaluation.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

from validation.evaluation.judges.base import BaseJudge


class QwenJudge(BaseJudge):
    """LLM judge using Ollama models (default: qwen3:8b)."""

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        cache_dir: Optional[str] = None,
        prompt_strategy: str = "simple",  # "simple" or "decomposed"
        base_url: str = "http://localhost:11434",
    ):
        cache_path = cache_dir or str(
            Path(__file__).parent / ".cache"
        )
        super().__init__(model_name=model_name, cache_dir=cache_path)
        self.prompt_strategy = prompt_strategy
        self.base_url = base_url
        self._cache = self._load_cache()

    def _cache_key(self, *parts: str) -> str:
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict:
        cache_path = Path(self.cache_dir) / f"judge_cache_{self.model_name.replace(':', '_')}.json"
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        filepath = cache_path / f"judge_cache_{self.model_name.replace(':', '_')}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def score_faithfulness(self, context: str, response: str) -> float:
        """Score faithfulness — placeholder, returns 0.0 until Phase 4."""
        key = self._cache_key("faithfulness", context, response)
        if key in self._cache:
            return self._cache[key]
        # TODO Phase 4: implement actual LLM scoring
        score = 0.0
        self._cache[key] = score
        return score

    def score_relevancy(self, question: str, response: str) -> float:
        """Score relevancy — placeholder, returns 0.0 until Phase 4."""
        key = self._cache_key("relevancy", question, response)
        if key in self._cache:
            return self._cache[key]
        # TODO Phase 4: implement actual LLM scoring
        score = 0.0
        self._cache[key] = score
        return score

    def get_prompt_version(self) -> str:
        return f"{self.prompt_strategy}_v0"
