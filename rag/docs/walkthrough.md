# CV Metric Instrumentation — Final Deliverable

## Phase 1 — Metric Opportunity Table (Ranked by CV Impact × Implementability)

| # | Metric | CV Signal | Script | Effort | Expected Range |
|---|--------|-----------|--------|--------|----------------|
| 1 | **Reranking MRR Δ** | Proves reranker adds real quality, not complexity | [eval_reranking_delta.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_reranking_delta.py) | Low | MRR +0.10–0.25 |
| 2 | **Router accuracy** | 3-layer routing correctly classifies intent | [eval_router_accuracy.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_router_accuracy.py) | Low | 85–96% |
| 3 | **HR@5, MRR, NDCG@5** | Canonical IR metrics at multiple k | [eval_ir_metrics.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_ir_metrics.py) | Low | HR@5 0.85–1.0, MRR 0.80–0.95 |
| 4 | **SQL first-attempt success** | NL2SQL robustness in a fragile pipeline | [eval_sql_robustness.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_sql_robustness.py) | Med | 60–80% first-attempt |
| 5 | **CRAG rewrite rate & filtering** | Corrective loop is functional, not decorative | [eval_crag_behavior.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_crag_behavior.py) | Med | 15–40% rewrite, 20–40% filtered |
| 6 | **Latency p50/p95** | Production-viable response times | [eval_latency_profile.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_latency_profile.py) | Med | RAG p50 ~2s, SQL p50 ~1s |
| 7 | **Test coverage %** | Engineering discipline signal | [eval_test_coverage.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_test_coverage.py) | Low | 70–90% |
| 8 | **Preprocessing rank shift** | Metadata matcher is not decorative | [eval_preprocessing_impact.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_preprocessing_impact.py) | Med | Avg +1–5 positions |

---

## Phase 2 — Scripts Created

All scripts are under [scripts/metrics/](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/).

### How to run

```bash
# Activate your environment first, e.g.: conda activate sprint5

# Scripts that need the full framework (index + models):
python -m scripts.metrics.eval_reranking_delta --config config/proyectos_docentes.yaml
python -m scripts.metrics.eval_ir_metrics --config config/proyectos_docentes.yaml --full-pipeline
python -m scripts.metrics.eval_preprocessing_impact --config config/proyectos_docentes.yaml

# Scripts that also need LLM + DB:
python -m scripts.metrics.eval_router_accuracy --config config/proyectos_docentes.yaml
python -m scripts.metrics.eval_sql_robustness --config config/proyectos_docentes.yaml
python -m scripts.metrics.eval_crag_behavior --config config/proyectos_docentes.yaml
python -m scripts.metrics.eval_latency_profile --config config/proyectos_docentes.yaml

# Dry-run (no infra needed — simulated placeholders):
python -m scripts.metrics.eval_router_accuracy --dry-run
python -m scripts.metrics.eval_sql_robustness --dry-run
python -m scripts.metrics.eval_crag_behavior --dry-run
python -m scripts.metrics.eval_latency_profile --dry-run

# Test coverage (only needs pytest + test dependencies):
python -m scripts.metrics.eval_test_coverage
```

Every script prints a clean `### Markdown` section at the end that can be copy-pasted directly into a report.

---

## Phase 3 — Rewritten CV Entry

> All `[PLACEHOLDER]` values should be replaced with real numbers after running the corresponding script.

---

**RAG System for Academic Query Answering** *(TFG — Double Degree in Mathematics & Computer Science)*

- **Improved retrieval quality by [PLACEHOLDER — run [eval_reranking_delta.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_reranking_delta.py)]% MRR** (Hit Rate@5: [PLACEHOLDER — run [eval_ir_metrics.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_ir_metrics.py)]) by engineering a hybrid dense+sparse retrieval pipeline with weighted Reciprocal Rank Fusion and cross-encoder reranking (25→5 candidates)

- **Achieved [PLACEHOLDER — run [eval_router_accuracy.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_router_accuracy.py)]% routing accuracy** across 25 labeled queries by designing a 3-layer query router (manual override → keyword rules → LLM classification) with confidence gating, eliminating unnecessary LLM and retrieval calls

- **Reduced hallucination-prone context by [PLACEHOLDER — run [eval_crag_behavior.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_crag_behavior.py)]%** by implementing Corrective RAG with per-document relevance grading, automatic query rewriting (triggered in [PLACEHOLDER]% of queries), and node-level deduplication

- **Achieved [PLACEHOLDER — run [eval_sql_robustness.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_sql_robustness.py)]% NL2SQL success rate** with SELECT-only enforcement, SQL injection detection (12 pattern classes), schema-grounded query generation, and automatic query relaxation on zero-result responses

- Built a metadata-aware query preprocessor combining fuzzy matching, token-overlap scoring, and Roman↔Arabic numeral expansion that shifted target documents an average of [PLACEHOLDER — run [eval_preprocessing_impact.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_preprocessing_impact.py)] positions higher in retrieval rank

- **Validated with [PLACEHOLDER — run [eval_test_coverage.py](file:///c:/Users/Dani/UNIVERSIDAD/TFG/custom_rag/scripts/metrics/eval_test_coverage.py)]% line coverage** across 190+ automated test cases spanning routing, CRAG edge cases, SQL security, and regression scenarios, with an IR evaluation harness computing Hit Rate@k, MRR, Precision@k, and NDCG@k

---

### Design Notes

| Bullet | Format | Rationale |
|--------|--------|-----------|
| 1 (Retrieval) | **XYZ** — metric-led | MRR delta is the single most defensible IR stat |
| 2 (Router) | **XYZ** — metric-led | Accuracy % on labeled data is directly verifiable |
| 3 (CRAG) | **XYZ** — metric-led | Filtering % + rewrite % show functional loop |
| 4 (NL2SQL) | **XYZ** — metric-led | Success rate in a fragile pipeline is impressive |
| 5 (Preprocessor) | Technical statement | Rank shift is meaningful but less recognizable to non-IR audiences |
| 6 (Testing) | **XYZ** — metric-led | Coverage % + test count + metric names show rigor |

Total: **6 bullets** (4 XYZ + 2 technical). Fits within 8–10 CV lines.
