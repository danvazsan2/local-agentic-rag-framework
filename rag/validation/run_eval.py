"""
RAG Framework Evaluation Script.

Single entry point for all evaluation suites:

  fast (default)  — retrieval IR metrics + reranking impact + router accuracy
                    (no LLM synthesis required, runs in minutes)
  latency         — end-to-end query timing by type (requires live LLM, slow)
  sql             — NL2SQL robustness: success rate, first-attempt rate
  ablation        — run retrieval metrics across C1-C7 configurations
  all             — all suites

Usage (run from the rag/ directory):

    python validation/run_eval.py
    python validation/run_eval.py --suite fast
    python validation/run_eval.py --suite ablation --run-id ablation_v1
    python validation/run_eval.py --pipeline C6 --suite retrieval
    python validation/run_eval.py --pipeline all --suite retrieval --run-id ablation_v1
    python validation/run_eval.py --suite latency --run-id eval_v1 --latency-repeats 3
    python validation/run_eval.py --suite all --run-id eval_v1 --save-json results/eval.json
    python validation/run_eval.py --partition adversarial --suite retrieval

Suites:
    fast       retrieval quality (IR metrics) + reranking impact + router accuracy
    retrieval  IR metrics only (HR@k, MRR, P@k, NDCG@k) with/without reranker
    router     query router accuracy: precision, recall, F1 per class
    latency    end-to-end p50/p95 latency by query type (uses LLM) — writes events.jsonl
    sql        NL2SQL success rate, first-attempt rate, timing (uses LLM) — writes events.jsonl
    ablation   retrieval metrics for all C1-C7 configurations
    all        all suites above

Dataset: validation/dataset.json (80 queries: 35 RAG, 18 SQL, 14 hybrid, 9 negative, 4 OOD)
Events:  validation/runs/<run_id>/events.jsonl  (one JSON per line per query)
"""

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — works when invoked as: python validation/run_eval.py
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]  # rag/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATASET_DEFAULT = Path(__file__).parent / "dataset.json"
_CONFIG_DEFAULT = _ROOT / "config" / "proyectos_docentes.yaml"
_RUNS_DIR = Path(__file__).parent / "runs"
_K_VALUES_DEFAULT = [3, 5, 10]
_MAX_RETRIEVAL_RETRIES = 2
_RETRIEVAL_RETRY_SLEEP_S = 1.0

logger = logging.getLogger(__name__)

# Maps dataset query types to router source labels
_LABEL_TO_SOURCE = {
    "rag": "unstructured",
    "sql": "structured",
    "hybrid": "hybrid",
    "negative": "unstructured",  # negative queries have no DB answer
}

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _hit_at_k(files: List[str], pattern: str, k: int) -> bool:
    """True if any retrieved file at rank ≤ k matches the expected pattern."""
    r = re.compile(pattern, re.IGNORECASE)
    return any(r.search(f) for f in files[:k])


def _rr_at_k(files: List[str], pattern: str, k: int) -> float:
    """Reciprocal rank of the first relevant file within top-k (0 if none)."""
    r = re.compile(pattern, re.IGNORECASE)
    for i, f in enumerate(files[:k], 1):
        if r.search(f):
            return 1.0 / i
    return 0.0


def _precision_at_k(files: List[str], pattern: str, k: int) -> float:
    """Fraction of top-k retrieved files that match the expected pattern."""
    subset = files[:k]
    if not subset:
        return 0.0
    r = re.compile(pattern, re.IGNORECASE)
    return sum(1 for f in subset if r.search(f)) / len(subset)


def _ndcg_at_k(files: List[str], pattern: str, k: int) -> float:
    """NDCG@k with binary relevance (1 if file matches pattern, 0 otherwise)."""
    r = re.compile(pattern, re.IGNORECASE)
    subset = files[:k]
    dcg = sum(
        (1.0 if r.search(f) else 0.0) / math.log2(i + 1)
        for i, f in enumerate(subset, 1)
    )
    if dcg == 0:
        return 0.0
    n_rel = sum(1 for f in subset if r.search(f))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


# ---------------------------------------------------------------------------
# Dataset and framework loading
# ---------------------------------------------------------------------------


def load_dataset(path: Optional[str] = None) -> List[dict]:
    p = Path(path) if path else _DATASET_DEFAULT
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_framework(config_path: str):
    """Load and index the RAGFramework from a YAML config, suppressing logs."""
    import logging
    logging.basicConfig(level=logging.WARNING)
    from rag_framework import RAGFramework
    rag = RAGFramework.from_yaml(str(config_path))
    rag.load_index()
    return rag


# ---------------------------------------------------------------------------
# Retrieval helpers (shared between retrieval and reranking suites)
# ---------------------------------------------------------------------------


def _retrieve(retriever, postprocessors, preprocessor, query_str: str) -> List[str]:
    """Run one query through retrieval (+ optional postprocessors) and return filenames."""
    metadata_filters = None
    prefilter_result = None

    if preprocessor is not None:
        prefilter_result = preprocessor.analyse(query_str)
        metadata_filters = prefilter_result.metadata_filters

    try:
        if metadata_filters is not None:
            nodes = retriever.retrieve(query_str, filters=metadata_filters)
        else:
            nodes = retriever.retrieve(query_str)
    except Exception as exc:
        # If hybrid vector retrieval fails (e.g. transient Ollama NaN), keep
        # the evaluation running by falling back to BM25-only retrieval.
        if not hasattr(retriever, "bm25_retriever"):
            raise

        logger.warning(
            "Hybrid retrieval failed for query '%s' (%s). Falling back to BM25-only.",
            query_str[:80],
            exc,
        )
        nodes = retriever.bm25_retriever.retrieve(query_str)
        if metadata_filters is not None and hasattr(retriever, "_manual_metadata_filter"):
            nodes = retriever._manual_metadata_filter(nodes, metadata_filters)

    for pp in postprocessors:
        nodes = pp.postprocess_nodes(nodes, query_str=query_str)

    if (
        prefilter_result is not None
        and prefilter_result.boost_field_value
        and preprocessor is not None
    ):
        cfg = preprocessor.config
        nodes = preprocessor.apply_boost(
            nodes,
            boost_field=cfg.boost_field,
            boost_value=prefilter_result.boost_field_value,
            factor=cfg.boost_factor,
        )

    return [n.node.metadata.get("file_name", "") for n in nodes]


def _retrieve_with_retries(
    retriever,
    postprocessors,
    preprocessor,
    query_str: str,
    max_retries: int = _MAX_RETRIEVAL_RETRIES,
) -> List[str]:
    """Retry retrieval a few times for transient provider failures."""
    last_error = None
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            return _retrieve(retriever, postprocessors, preprocessor, query_str)
        except Exception as exc:
            last_error = exc
            if attempt >= total_attempts:
                break
            logger.warning(
                "Retrieval failed for query '%s' (%d/%d): %s. Retrying...",
                query_str[:80],
                attempt,
                total_attempts,
                exc,
            )
            time.sleep(_RETRIEVAL_RETRY_SLEEP_S)

    raise last_error


# ---------------------------------------------------------------------------
# Suite: Retrieval quality (IR metrics)
# ---------------------------------------------------------------------------


def run_retrieval_suite(rag, queries: List[dict], k_values: List[int], full_pipeline: bool):
    """Compute HR@k, MRR@k, P@k, NDCG@k.

    When full_pipeline=False, only the hybrid retriever is used (no reranker).
    When full_pipeline=True, the query preprocessor and reranker are also applied.
    """
    rag_queries = [q for q in queries if q.get("expected_source_pattern")]
    if not rag_queries:
        return {"error": "No RAG queries with expected_source_pattern in dataset."}

    rag._query_ops._ensure_query_engine()
    qe = rag._query_engine
    retriever = qe.retriever
    postprocessors = qe.node_postprocessors if full_pipeline else []
    preprocessor = qe.query_preprocessor if full_pipeline else None

    # Ensure retriever returns at least max(k) candidates
    max_k = max(k_values)
    if hasattr(retriever, "config"):
        retriever.config.top_k = max(max_k, retriever.config.top_k)

    per_query = []
    errors = []
    t_start = time.time()

    for item in rag_queries:
        t0 = time.time()
        try:
            files = _retrieve_with_retries(
                retriever, postprocessors, preprocessor, item["query"]
            )
            error_msg = None
        except Exception as exc:
            files = []
            error_msg = str(exc)
            errors.append({"id": item["id"], "error": error_msg})
            logger.error(
                "Query %s failed after retries in retrieval suite: %s",
                item["id"],
                exc,
            )

        elapsed = time.time() - t0
        pat = item["expected_source_pattern"]

        per_query.append({
            "id": item["id"],
            "query": item["query"],
            "pattern": pat,
            "files": files,
            "error": error_msg,
            "elapsed_s": elapsed,
            "hit_at_k": {k: _hit_at_k(files, pat, k) for k in k_values},
            "rr_at_k": {k: _rr_at_k(files, pat, k) for k in k_values},
            "precision_at_k": {k: _precision_at_k(files, pat, k) for k in k_values},
            "ndcg_at_k": {k: _ndcg_at_k(files, pat, k) for k in k_values},
        })

    n = len(per_query)
    total_time = time.time() - t_start
    aggregates = {
        "n_queries": n,
        "total_time_s": total_time,
        "avg_time_s": total_time / max(n, 1),
    }
    for k in k_values:
        aggregates[f"hr@{k}"] = sum(r["hit_at_k"][k] for r in per_query) / n
        aggregates[f"mrr@{k}"] = sum(r["rr_at_k"][k] for r in per_query) / n
        aggregates[f"p@{k}"] = sum(r["precision_at_k"][k] for r in per_query) / n
        aggregates[f"ndcg@{k}"] = sum(r["ndcg_at_k"][k] for r in per_query) / n

    return {
        "per_query": per_query,
        "aggregates": aggregates,
        "k_values": k_values,
        "full_pipeline": full_pipeline,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Suite: Router accuracy
# ---------------------------------------------------------------------------


def run_router_suite(rag, queries: List[dict]) -> dict:
    """Route all queries and compare predicted source against ground-truth type."""
    rag._query_ops._ensure_query_engine()
    rag._hybrid_ops.ensure_hybrid_engine()
    router = rag._hybrid_engine.router

    per_query = []
    for item in queries:
        truth = _LABEL_TO_SOURCE.get(item["type"], "unstructured")
        routing = router.route(item["query"])
        pred = routing.source.value
        conf = routing.confidence
        per_query.append({
            "id": item["id"],
            "type": item["type"],
            "truth": truth,
            "predicted": pred,
            "confidence": conf,
            "correct": truth == pred,
        })

    n = len(per_query)
    correct = sum(1 for r in per_query if r["correct"])

    labels = sorted(set(_LABEL_TO_SOURCE.values()))
    matrix: Dict[str, Counter] = defaultdict(Counter)
    for r in per_query:
        matrix[r["truth"]][r["predicted"]] += 1

    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    confs = [r["confidence"] for r in per_query]
    return {
        "per_query": per_query,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "n_queries": n,
        "per_class": per_class,
        "confusion_matrix": {k: dict(v) for k, v in matrix.items()},
        "confidence": {
            "min": min(confs),
            "max": max(confs),
            "mean": sum(confs) / len(confs),
        },
    }


# ---------------------------------------------------------------------------
# Suite: End-to-end latency
# ---------------------------------------------------------------------------


def run_latency_suite(rag, queries: List[dict], run_dir: Optional[Path] = None) -> dict:
    """Run all queries through HybridQueryEngine and record per-query timing."""
    from rag_framework.core.instrumentation import QueryTrace, write_event

    rag._query_ops._ensure_query_engine()
    rag._hybrid_ops.ensure_hybrid_engine()
    hybrid_engine = rag._hybrid_engine

    run_id = run_dir.name if run_dir else "adhoc"

    per_query = []
    for item in queries:
        trace = QueryTrace(
            query_id=item["id"],
            run_id=run_id,
            query=item["query"],
            configuration="full_pipeline",
        )
        t0 = time.time()
        try:
            response = hybrid_engine.query(item["query"], trace=trace)
            success = True
            elapsed_s = (
                getattr(response, "total_time_ms", (time.time() - t0) * 1000) / 1000
            )
            trace.phases.setdefault("total_ms", round(elapsed_s * 1000, 2))
            if not trace.response:
                trace.response = getattr(response, "response", "")
        except Exception as exc:
            success = False
            elapsed_s = time.time() - t0
            trace.error = str(exc)

        if run_dir is not None:
            write_event(run_dir, trace)

        per_query.append({
            "id": item["id"],
            "type": item["type"],
            "partition": item.get("partition", ""),
            "query": item["query"][:60],
            "success": success,
            "elapsed_s": elapsed_s,
        })

    by_type: Dict[str, List[float]] = defaultdict(list)
    for r in per_query:
        if r["success"]:
            by_type[r["type"]].append(r["elapsed_s"])

    all_times = [r["elapsed_s"] for r in per_query if r["success"]]
    profile = {}
    for qtype, times in sorted(by_type.items()):
        profile[qtype] = {
            "count": len(times),
            "p50": _percentile(times, 50),
            "p95": _percentile(times, 95),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }

    overall = None
    if all_times:
        overall = {
            "count": len(all_times),
            "p50": _percentile(all_times, 50),
            "p95": _percentile(all_times, 95),
            "mean": sum(all_times) / len(all_times),
            "min": min(all_times),
            "max": max(all_times),
        }

    return {"per_query": per_query, "by_type": profile, "overall": overall}


# ---------------------------------------------------------------------------
# Suite: SQL robustness
# ---------------------------------------------------------------------------


def run_sql_suite(rag, queries: List[dict], run_dir: Optional[Path] = None) -> dict:
    """Run SQL/hybrid queries through the SQLAgent and collect success metrics."""
    from rag_framework.core.instrumentation import QueryTrace, write_event

    sql_queries = [q for q in queries if q["type"] in ("sql", "hybrid")]
    if not sql_queries:
        return {"error": "No SQL/hybrid queries found in dataset."}

    rag._query_ops._ensure_query_engine()
    rag._hybrid_ops.ensure_hybrid_engine()
    sql_agent = rag._hybrid_engine.sql_agent

    run_id = run_dir.name if run_dir else "adhoc"

    per_query = []
    for item in sql_queries:
        trace = QueryTrace(
            query_id=item["id"],
            run_id=run_id,
            query=item["query"],
            configuration="sql_agent_only",
        )
        result = sql_agent.query(item["query"], trace=trace)

        if run_dir is not None:
            write_event(run_dir, trace)

        per_query.append({
            "id": item["id"],
            "type": item["type"],
            "partition": item.get("partition", ""),
            "query": item["query"][:60],
            "success": result.success,
            "attempts": result.generation_attempts,
            "relaxed": result.query_relaxed,
            "time_ms": result.total_time_ms,
            "error": result.error,
        })

    n = len(per_query)
    successes = sum(1 for r in per_query if r["success"])
    first_attempt = sum(1 for r in per_query if r["success"] and r["attempts"] == 1)
    relaxed = sum(1 for r in per_query if r["relaxed"])
    times = [r["time_ms"] for r in per_query if r["success"]]

    return {
        "per_query": per_query,
        "n_queries": n,
        "success_rate": successes / n if n else 0.0,
        "first_attempt_rate": first_attempt / n if n else 0.0,
        "relaxation_rate": relaxed / n if n else 0.0,
        "latency_ms": {
            "p50": _percentile(times, 50),
            "p95": _percentile(times, 95),
            "mean": sum(times) / len(times) if times else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Console printing
# ---------------------------------------------------------------------------


def _hr(char="=", w=65):
    return char * w


def print_retrieval(result: dict, label: str = ""):
    agg = result["aggregates"]
    k_values = result["k_values"]
    mode = "Full pipeline (preprocessor + reranker)" if result["full_pipeline"] else "Retrieval-only"

    print(f"\n{_hr()}")
    print(f"RETRIEVAL QUALITY — {mode} {label}")
    print(_hr())
    print(f"  Queries: {agg['n_queries']}  |  Avg time/query: {agg['avg_time_s']:.2f}s")
    print()
    print(f"  {'k':>3}  {'HR@k':>7}  {'MRR@k':>7}  {'P@k':>7}  {'NDCG@k':>7}")
    print(f"  {'-' * 35}")
    for k in k_values:
        print(
            f"  {k:>3}  {agg[f'hr@{k}']:>7.3f}  {agg[f'mrr@{k}']:>7.3f}"
            f"  {agg[f'p@{k}']:>7.3f}  {agg[f'ndcg@{k}']:>7.3f}"
        )

    errors = result.get("errors") or []
    if errors:
        print(f"\n  Retrieval errors handled: {len(errors)} (counted as misses).")

    # Show misses at the smallest k
    k0 = k_values[0]
    misses = [r for r in result["per_query"] if not r["hit_at_k"][k0]]
    if misses:
        print(f"\n  Misses at k={k0} ({len(misses)}/{agg['n_queries']}):")
        for r in misses[:6]:
            top = ", ".join(r["files"][:2]) or "(none)"
            print(f"    {r['id']}: {r['query'][:55]}")
            print(f"      Pattern: {r['pattern']}  |  Top-2: {top}")


def print_reranking_delta(base: dict, full: dict, k_values: List[int]):
    k = k_values[-1]  # report delta at the highest k
    base_t = base["aggregates"]["avg_time_s"]
    full_t = full["aggregates"]["avg_time_s"]

    print(f"\n{_hr()}")
    print("RERANKING IMPACT (retrieval-only vs full pipeline)")
    print(_hr())
    print(f"  {'Metric':<14}  {'Retrieval-only':>14}  {'Full pipeline':>14}  {'Δ':>7}  {'Δ%':>7}")
    print(f"  {'-' * 60}")
    for name, bk, fk in [
        (f"HR@{k}", f"hr@{k}", f"hr@{k}"),
        (f"MRR@{k}", f"mrr@{k}", f"mrr@{k}"),
        (f"P@{k}", f"p@{k}", f"p@{k}"),
        (f"NDCG@{k}", f"ndcg@{k}", f"ndcg@{k}"),
    ]:
        bv = base["aggregates"][bk]
        fv = full["aggregates"][fk]
        d = fv - bv
        pct = (d / bv * 100) if bv > 0 else 0.0
        sign = "+" if d >= 0 else ""
        print(
            f"  {name:<14}  {bv:>14.3f}  {fv:>14.3f}  {sign}{d:>6.3f}  {sign}{pct:>5.1f}%"
        )
    print(f"\n  Avg time: retrieval-only = {base_t:.2f}s  |  full pipeline = {full_t:.2f}s")


def print_router(result: dict):
    print(f"\n{_hr()}")
    print("ROUTER ACCURACY")
    print(_hr())
    print(f"  Overall: {result['accuracy']:.2%} ({result['correct']}/{result['n_queries']})")
    print()
    print(f"  {'Class':<14}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Support':>10}")
    print(f"  {'-' * 58}")
    for label, m in sorted(result["per_class"].items()):
        print(
            f"  {label:<14}  {m['precision']:>10.2f}  {m['recall']:>10.2f}"
            f"  {m['f1']:>10.2f}  {m['support']:>10}"
        )
    c = result["confidence"]
    print(f"\n  Confidence: min={c['min']:.2f}  max={c['max']:.2f}  mean={c['mean']:.2f}")

    # Confusion matrix
    labels = sorted(result["confusion_matrix"])
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = f"  {'':14}" + "".join(f"{l:>14}" for l in labels)
    print(header)
    for tl in labels:
        row_data = result["confusion_matrix"].get(tl, {})
        row = f"  {tl:<14}" + "".join(f"{row_data.get(pl, 0):>14}" for pl in labels)
        print(row)


def print_latency(result: dict):
    print(f"\n{_hr()}")
    print("END-TO-END LATENCY PROFILE")
    print(_hr())
    print(f"  {'Type':<12}  {'Count':>6}  {'p50 (s)':>8}  {'p95 (s)':>8}  {'Mean (s)':>9}")
    print(f"  {'-' * 50}")
    for qtype, p in sorted(result["by_type"].items()):
        print(
            f"  {qtype:<12}  {p['count']:>6}  {p['p50']:>8.2f}"
            f"  {p['p95']:>8.2f}  {p['mean']:>9.2f}"
        )
    if result["overall"]:
        o = result["overall"]
        n_failed = sum(1 for r in result["per_query"] if not r["success"])
        print(f"  {'-' * 50}")
        print(
            f"  {'OVERALL':<12}  {o['count']:>6}  {o['p50']:>8.2f}"
            f"  {o['p95']:>8.2f}  {o['mean']:>9.2f}"
        )
        if n_failed:
            print(f"\n  Failed queries: {n_failed}")


def print_sql(result: dict):
    print(f"\n{_hr()}")
    print("SQL ROBUSTNESS")
    print(_hr())
    n = result["n_queries"]
    lat = result["latency_ms"]
    print(f"  Queries evaluated     : {n}")
    print(f"  Success rate          : {result['success_rate']:.0%}")
    print(f"  First-attempt success : {result['first_attempt_rate']:.0%}")
    print(f"  Required relaxation   : {result['relaxation_rate']:.0%}")
    print(f"  Latency p50           : {lat['p50']:.0f} ms")
    print(f"  Latency p95           : {lat['p95']:.0f} ms")

    failed = [r for r in result["per_query"] if not r["success"]]
    if failed:
        print(f"\n  Failed queries ({len(failed)}):")
        for r in failed:
            print(f"    {r['id']}: {r['query']}")
            if r["error"]:
                print(f"      Error: {r['error'][:80]}")


# ---------------------------------------------------------------------------
# Markdown summary (copy-paste for TFG)
# ---------------------------------------------------------------------------


def print_markdown(results: dict, k_values: List[int]):
    print(f"\n{_hr('=', 70)}")
    print("MARKDOWN SUMMARY — copy-paste into TFG report")
    print(_hr("=", 70))

    if "retrieval_base" in results and "retrieval_full" in results:
        full_agg = results["retrieval_full"]["aggregates"]
        base_agg = results["retrieval_base"]["aggregates"]
        k_hi = k_values[-1]

        print("\n#### Retrieval Quality (full pipeline)\n")
        print("| k | HR@k | MRR@k | P@k | NDCG@k |")
        print("|---|------|-------|-----|--------|")
        for k in k_values:
            print(
                f"| {k} | {full_agg[f'hr@{k}']:.3f} | {full_agg[f'mrr@{k}']:.3f}"
                f" | {full_agg[f'p@{k}']:.3f} | {full_agg[f'ndcg@{k}']:.3f} |"
            )

        print("\n#### Reranking Impact\n")
        print("| Metric | Retrieval-Only | + Reranker | Δ | Δ% |")
        print("|--------|---------------|-----------|---|-----|")
        for name, key in [
            (f"HR@{k_hi}", f"hr@{k_hi}"),
            (f"MRR@{k_hi}", f"mrr@{k_hi}"),
            (f"P@{k_hi}", f"p@{k_hi}"),
        ]:
            bv = base_agg[key]
            fv = full_agg[key]
            d = fv - bv
            pct = (d / bv * 100) if bv > 0 else 0.0
            sign = "+" if d >= 0 else ""
            print(f"| {name} | {bv:.3f} | {fv:.3f} | {sign}{d:.3f} | {sign}{pct:.1f}% |")

        avg_base = base_agg["avg_time_s"]
        avg_full = full_agg["avg_time_s"]
        print(f"\nAverage time/query: retrieval-only = {avg_base:.2f} s, full pipeline = {avg_full:.2f} s")

    if "router" in results:
        r = results["router"]
        print("\n#### Router Accuracy\n")
        print("| Class | Precision | Recall | F1 | Support |")
        print("|-------|-----------|--------|-----|---------|")
        for label, m in sorted(r["per_class"].items()):
            print(
                f"| {label} | {m['precision']:.2f} | {m['recall']:.2f}"
                f" | {m['f1']:.2f} | {m['support']} |"
            )
        print(f"\nOverall accuracy: **{r['accuracy']:.0%}** ({r['correct']}/{r['n_queries']})")

    if "latency" in results:
        lat = results["latency"]
        print("\n#### End-to-End Latency\n")
        print("| Query Type | Count | p50 (s) | p95 (s) | Mean (s) |")
        print("|------------|-------|---------|---------|----------|")
        for qtype, p in sorted(lat["by_type"].items()):
            print(f"| {qtype} | {p['count']} | {p['p50']:.2f} | {p['p95']:.2f} | {p['mean']:.2f} |")
        if lat["overall"]:
            o = lat["overall"]
            print(f"| **Overall** | {o['count']} | {o['p50']:.2f} | {o['p95']:.2f} | {o['mean']:.2f} |")

    if "sql" in results:
        s = results["sql"]
        lat = s["latency_ms"]
        print("\n#### NL2SQL Robustness\n")
        print("| Metric | Value |")
        print("|--------|-------|")
        print(f"| Success rate | {s['success_rate']:.0%} ({int(s['success_rate'] * s['n_queries'])}/{s['n_queries']}) |")
        print(f"| First-attempt success | {s['first_attempt_rate']:.0%} |")
        print(f"| Required relaxation | {s['relaxation_rate']:.0%} |")
        print(f"| Latency p50 | {lat['p50']:.0f} ms |")
        print(f"| Latency p95 | {lat['p95']:.0f} ms |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="RAG Framework Evaluation — unified runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=str(_CONFIG_DEFAULT),
        help="Path to YAML config (default: config/proyectos_docentes.yaml)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to eval dataset JSON (default: validation/dataset.json)",
    )
    parser.add_argument(
        "--suite",
        choices=["fast", "retrieval", "router", "latency", "sql", "ablation", "all"],
        default="fast",
        help=(
            "Suite to run: "
            "fast = retrieval + reranking + router (no LLM, default); "
            "latency = e2e timing; sql = NL2SQL robustness; "
            "ablation = retrieval across C1-C7; all = everything"
        ),
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        metavar="CONFIG_ID",
        help=(
            "Pipeline configuration for ablation: C1..C7, C6_crag, or 'all'. "
            "Used with --suite retrieval or --suite ablation."
        ),
    )
    parser.add_argument(
        "--partition",
        choices=["well_formed", "adversarial", "all"],
        default="all",
        help="Restrict evaluation to a specific partition (default: all).",
    )
    parser.add_argument(
        "--latency-repeats",
        type=int,
        default=1,
        metavar="N",
        help="Number of repetitions for latency variance analysis (default: 1).",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=_K_VALUES_DEFAULT,
        metavar="K",
        help="k values for IR metrics (default: 3 5 10)",
    )
    parser.add_argument(
        "--save-json",
        metavar="PATH",
        default=None,
        help="Save all raw results to a JSON file",
    )
    parser.add_argument(
        "--run-id",
        metavar="ID",
        default=None,
        help=(
            "Run identifier for events.jsonl output "
            "(default: eval_<timestamp>). "
            "Writes to validation/runs/<run_id>/events.jsonl."
        ),
    )
    args = parser.parse_args()

    # Determine run_dir for event logging
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = _RUNS_DIR / run_id

    k_values = args.k_values
    queries = load_dataset(args.dataset)

    # Partition filter
    if args.partition and args.partition != "all":
        queries = [q for q in queries if q.get("partition") == args.partition]
        print(f"Partition filter: {args.partition} → {len(queries)} queries")

    type_counts = Counter(q["type"] for q in queries)
    print(f"Dataset: {len(queries)} queries  "
          + "  ".join(f"{t}={c}" for t, c in sorted(type_counts.items())))

    # Determine which suites to run
    run_ir = args.suite in ("fast", "retrieval", "all")
    run_router = args.suite in ("fast", "router", "all")
    run_latency = args.suite in ("latency", "all")
    run_sql = args.suite in ("sql", "all")
    run_ablation = args.suite in ("ablation", "all")

    print(f"\nLoading framework from {args.config} ...")
    rag = load_framework(args.config)

    all_results = {}

    # ── Ablation suite ──
    if run_ablation or args.pipeline:
        _run_ablation(rag, queries, k_values, args.pipeline, run_dir, all_results)

    if run_ir and not args.pipeline:
        print("\n>>> Retrieval-only pass ...")
        base = run_retrieval_suite(rag, queries, k_values, full_pipeline=False)
        all_results["retrieval_base"] = base
        print_retrieval(base)

        print("\n>>> Full pipeline pass (preprocessor + reranker) ...")
        full = run_retrieval_suite(rag, queries, k_values, full_pipeline=True)
        all_results["retrieval_full"] = full
        print_retrieval(full)

        print_reranking_delta(base, full, k_values)

    if run_router:
        print("\n>>> Router accuracy ...")
        router_result = run_router_suite(rag, queries)
        all_results["router"] = router_result
        print_router(router_result)

    if run_latency:
        n_q = len(queries)
        n_repeats = args.latency_repeats
        print(f"\n>>> End-to-end latency ({n_q} queries × {n_repeats} repeats, uses LLM) ...")
        print(f"    Events → {run_dir}/events.jsonl")
        for iteration in range(1, n_repeats + 1):
            if n_repeats > 1:
                print(f"\n  --- Iteration {iteration}/{n_repeats} ---")
            latency_result = run_latency_suite(
                rag, queries, run_dir=run_dir,
            )
            all_results[f"latency_iter{iteration}"] = latency_result
        # Store last iteration as "latency" for backward compat
        all_results["latency"] = latency_result
        print_latency(latency_result)

    if run_sql:
        n_sql = sum(1 for q in queries if q["type"] in ("sql", "hybrid"))
        print(f"\n>>> SQL robustness ({n_sql} queries, uses LLM) ...")
        print(f"    Events → {run_dir}/events.jsonl")
        sql_result = run_sql_suite(rag, queries, run_dir=run_dir)
        all_results["sql"] = sql_result
        print_sql(sql_result)

    # Always print markdown summary
    print_markdown(all_results, k_values)

    # Save metrics.json to run dir
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nMetrics saved to {metrics_path}")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nRaw results saved to {out}")


def _run_ablation(rag, queries, k_values, pipeline_arg, run_dir, all_results):
    """Run ablation study across pipeline configurations."""
    from validation.evaluation.configurations import (
        get_config, get_all_configs, describe_configs,
    )
    from validation.evaluation.runner import run_config_retrieval
    from validation.evaluation.metrics.retrieval import format_table

    print(f"\n{'=' * 65}")
    print("ABLATION STUDY — Retrieval metrics across configurations")
    print(f"{'=' * 65}")
    print(describe_configs())
    print()

    # Determine which configs to run
    if pipeline_arg and pipeline_arg.lower() != "all":
        configs = [get_config(pipeline_arg)]
    else:
        configs = get_all_configs(include_crag=False)

    ablation_results = {}

    for cfg in configs:
        print(f"\n>>> [{cfg.id}] {cfg.name} ...")
        result = run_config_retrieval(rag, queries, cfg, k_values)
        ablation_results[cfg.id] = result

        # Print summary
        agg = result["aggregated"]
        print(f"    Time: {agg['total_time_s']:.1f}s  |  Queries: {agg['n_queries']}")
        print(format_table(agg["overall"], k_values, label=f"{cfg.id} overall"))
        if agg["well_formed"]["n_queries"] > 0:
            print(format_table(agg["well_formed"], k_values, label=f"{cfg.id} well_formed"))
        if agg["adversarial"]["n_queries"] > 0:
            print(format_table(agg["adversarial"], k_values, label=f"{cfg.id} adversarial"))

    all_results["ablation"] = ablation_results

    # Print comparison table
    _print_ablation_comparison(ablation_results, k_values)


def _print_ablation_comparison(ablation_results, k_values):
    """Print a comparison table across all configurations."""
    k = k_values[-1]  # use highest k

    print(f"\n{'=' * 75}")
    print(f"ABLATION COMPARISON (k={k})")
    print(f"{'=' * 75}")

    header = f"  {'Config':<22}  {'HR@k':>7}  {'MRR@k':>7}  {'P@k':>7}  {'NDCG@k':>7}"
    for partition in ["overall", "well_formed", "adversarial"]:
        print(f"\n  [{partition}]")
        print(header)
        print(f"  {'-' * 58}")
        for config_id, result in sorted(ablation_results.items()):
            agg = result["aggregated"].get(partition, {})
            if agg.get("n_queries", 0) == 0:
                continue
            name = result.get("config_name", config_id)
            label = f"{config_id} ({name})"
            print(
                f"  {label:<22}  "
                f"{agg.get(f'hr@{k}', 0):>7.3f}  "
                f"{agg.get(f'mrr@{k}', 0):>7.3f}  "
                f"{agg.get(f'p@{k}', 0):>7.3f}  "
                f"{agg.get(f'ndcg@{k}', 0):>7.3f}"
            )


if __name__ == "__main__":
    main()
