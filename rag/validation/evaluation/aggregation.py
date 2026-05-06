"""
Offline aggregation of events.jsonl into metrics.json and errors.json.

Reads raw event logs from a run directory and computes all metrics
using the individual metric modules.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_events(run_dir: Path) -> List[dict]:
    """Load all events from a run directory."""
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_dataset(dataset_path: Optional[Path] = None) -> List[dict]:
    """Load the evaluation dataset."""
    if dataset_path is None:
        dataset_path = Path(__file__).parent.parent / "dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def build_dataset_index(dataset: List[dict]) -> Dict[str, dict]:
    """Build {query_id: entry} index."""
    return {q["id"]: q for q in dataset}


def save_metrics(run_dir: Path, metrics: dict):
    """Save aggregated metrics to metrics.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)


def save_errors(run_dir: Path, errors: dict):
    """Save error analysis to errors.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False, default=str)
