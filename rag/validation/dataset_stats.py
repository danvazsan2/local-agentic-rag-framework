"""
Dataset statistics for validation/dataset.json.
Used in TFG chapter 7.2.
"""

import json
import re
from collections import Counter
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "dataset.json"
CORPUS_DIR = Path(__file__).parent.parent / "documents"


def load_dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def approx_token_count(text: str) -> int:
    return max(1, len(text.split()))


def corpus_vocabulary() -> set:
    """Build vocabulary from all PDF filenames (proxy for corpus token overlap)."""
    vocab = set()
    if not CORPUS_DIR.exists():
        return vocab
    for pdf in CORPUS_DIR.glob("*.pdf"):
        for token in re.split(r"[_\-. ]+", pdf.stem.lower()):
            if token:
                vocab.add(token)
    return vocab


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def main():
    data = load_dataset()
    total = len(data)

    print_section("DATASET OVERVIEW")
    print(f"Total queries: {total}")

    # Distribution by partition
    print_section("BY PARTITION")
    for partition, count in Counter(q["partition"] for q in data).items():
        pct = count / total * 100
        print(f"  {partition:<20} {count:>3}  ({pct:.1f}%)")

    # Distribution by type
    print_section("BY TYPE")
    for qtype, count in Counter(q["type"] for q in data).most_common():
        pct = count / total * 100
        print(f"  {qtype:<20} {count:>3}  ({pct:.1f}%)")

    # Distribution by difficulty
    print_section("BY DIFFICULTY")
    for diff in ["easy", "medium", "hard"]:
        count = sum(1 for q in data if q["difficulty"] == diff)
        pct = count / total * 100
        print(f"  {diff:<20} {count:>3}  ({pct:.1f}%)")

    # Cross-table: partition × type
    print_section("PARTITION × TYPE")
    partitions = ["well_formed", "adversarial"]
    types = ["rag", "sql", "hybrid", "negative", "out_of_domain"]
    header = f"{'':18}" + "".join(f"{t:>15}" for t in types) + f"{'TOTAL':>8}"
    print(header)
    for part in partitions:
        row = f"  {part:<16}"
        row_total = 0
        for t in types:
            n = sum(1 for q in data if q["partition"] == part and q["type"] == t)
            row += f"{n:>15}"
            row_total += n
        row += f"{row_total:>8}"
        print(row)

    # Abstention
    print_section("ABSTENTION")
    abstentions = sum(1 for q in data if q.get("expected_abstention"))
    print(f"  expected_abstention=true:  {abstentions}")
    print(f"  expected_abstention=false: {total - abstentions}")

    # Query length statistics
    print_section("QUERY LENGTH")
    chars = [len(q["query"]) for q in data]
    tokens = [approx_token_count(q["query"]) for q in data]

    def stats(values):
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std = variance ** 0.5
        return mean, std, min(values), max(values)

    c_mean, c_std, c_min, c_max = stats(chars)
    t_mean, t_std, t_min, t_max = stats(tokens)
    print(f"  Characters — mean={c_mean:.1f}  std={c_std:.1f}  min={c_min}  max={c_max}")
    print(f"  Tokens     — mean={t_mean:.1f}  std={t_std:.1f}  min={t_min}  max={t_max}")

    # By partition
    for part in partitions:
        subset = [q for q in data if q["partition"] == part]
        c = [len(q["query"]) for q in subset]
        t = [approx_token_count(q["query"]) for q in subset]
        cm, cs, _, _ = stats(c)
        tm, ts, _, _ = stats(t)
        print(f"  [{part}] chars mean={cm:.1f} std={cs:.1f} | tokens mean={tm:.1f} std={ts:.1f}")

    # Subject distribution (for well_formed)
    print_section("SUBJECT DISTRIBUTION (well_formed RAG + hybrid)")
    subject_counts = Counter(
        q["expected_subject"]
        for q in data
        if q["partition"] == "well_formed" and q.get("expected_subject")
    )
    for subj, count in subject_counts.most_common():
        print(f"  {count:>2}×  {subj}")
    print(f"  ({len(subject_counts)} distinct subjects covered)")

    # Vocabulary overlap with corpus
    print_section("VOCABULARY OVERLAP WITH CORPUS")
    corpus_vocab = corpus_vocabulary()
    if corpus_vocab:
        all_query_tokens = set()
        for q in data:
            for token in re.split(r"\W+", q["query"].lower()):
                if token and len(token) > 2:
                    all_query_tokens.add(token)
        overlap = all_query_tokens & corpus_vocab
        pct_overlap = len(overlap) / len(all_query_tokens) * 100 if all_query_tokens else 0
        print(f"  Unique query tokens (>2 chars): {len(all_query_tokens)}")
        print(f"  Overlap with corpus filenames:  {len(overlap)}  ({pct_overlap:.1f}%)")
        print("  Note: corpus vocab is built from filenames only (not PDF content)")
    else:
        print(f"  Corpus dir not found at {CORPUS_DIR}; skipping overlap analysis.")
        all_query_tokens = set()
        for q in data:
            for token in re.split(r"\W+", q["query"].lower()):
                if token and len(token) > 2:
                    all_query_tokens.add(token)
        print(f"  Unique query tokens (>2 chars): {len(all_query_tokens)}")

    # Expected behaviors breakdown
    print_section("EXPECTED BEHAVIORS")
    routing_dist = Counter(
        q["expected_behaviors"]["routing"]
        for q in data
        if "expected_behaviors" in q
    )
    print("  Routing distribution:")
    for route, count in routing_dist.most_common():
        print(f"    {route:<20} {count:>3}")
    crag_rewrite = sum(
        1 for q in data
        if q.get("expected_behaviors", {}).get("crag_should_rewrite", False)
    )
    meta_filter = sum(
        1 for q in data
        if q.get("expected_behaviors", {}).get("requires_metadata_filter", False)
    )
    print(f"  crag_should_rewrite=true:    {crag_rewrite}")
    print(f"  requires_metadata_filter=true: {meta_filter}")

    print()


if __name__ == "__main__":
    main()
