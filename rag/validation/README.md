# Evaluación — Estructura del directorio

Este directorio contiene el pipeline de evaluación del TFG RAG.

## Ficheros

| Fichero | Descripción |
|---------|-------------|
| `dataset.json` | Dataset principal — 80 consultas supervisadas (v2) |
| `dataset_legacy.json` | Dataset original de 38 consultas (solo trazabilidad, no usar) |
| `dataset_stats.py` | Estadísticas descriptivas del dataset (sección 7.2 del TFG) |
| `corpus_inventory.md` | Inventario del corpus: asignaturas, BD, clusters semánticos |
| `run_eval.py` | Pipeline de evaluación — genera métricas + `runs/<run_id>/` |
| `inspect_run.py` | Inspección de ejecuciones anteriores por run_id |
| `runs/` | Directorio de resultados (generado en tiempo de ejecución) |

## Dataset — v2 (80 consultas)

### Particiones

| Partición | Consultas | Propósito |
|-----------|-----------|-----------|
| `well_formed` | 48 | Régimen bien formado; mide el techo del sistema |
| `adversarial` | 32 | Régimen adversarial; revela límites de cada componente |

### Tipos de consulta

| Tipo | Descripción |
|------|-------------|
| `rag` | Consulta sobre documentos (PDFs de programas y proyectos docentes) |
| `sql` | Consulta sobre la base de datos estructurada |
| `hybrid` | Requiere combinar evidencia de BD y documentos |
| `negative` | El sistema debe abstenerse (dato no disponible en corpus o asignatura/profesor inexistentes) |
| `out_of_domain` | Completamente fuera del dominio universitario |

### Campos de cada entrada

```json
{
  "id": "r01",
  "partition": "well_formed",
  "type": "rag",
  "query": "¿Cómo se evalúa Fundamentos de Programación?",
  "expected_source_pattern": ".*2060001.*",
  "expected_keywords": ["evaluación", "examen", "prácticas", "nota"],
  "expected_subject": "Fundamentos de Programación",
  "expected_abstention": false,
  "difficulty": "easy",
  "rationale": "...",
  "expected_behaviors": {
    "routing": "unstructured",
    "crag_should_rewrite": false,
    "requires_metadata_filter": true
  }
}
```

### Prefijos de IDs

- `r01–r35`: consultas RAG (25 well_formed + 10 adversarial)
- `s01–s18`: consultas SQL (10 well_formed + 8 adversarial)
- `h01–h14`: consultas híbridas (8 well_formed + 6 adversarial)
- `n01–n09`: consultas negativas (5 well_formed + 4 adversarial)
- `a01–a04`: consultas out-of-domain (todas adversarial)

## Uso

### Ver estadísticas del dataset

```bash
python validation/dataset_stats.py
```

### Ejecutar evaluación completa

```bash
python validation/run_eval.py --run-id eval_v1
```

Genera `validation/runs/eval_v1/events.jsonl` con un JSON por consulta.

### Inspeccionar una ejecución

```bash
# Ver todas las consultas de una ejecución
python validation/inspect_run.py eval_v1

# Filtrar por tipo
python validation/inspect_run.py eval_v1 --type sql

# Ver una consulta específica
python validation/inspect_run.py eval_v1 --query-id s01

# Percentiles de latencia
python validation/inspect_run.py eval_v1 --percentile 95
```

## Estructura de `runs/`

```
runs/
  eval_v1/
    events.jsonl          # Un JSON por línea, una línea por consulta
    summary.json          # Métricas agregadas (generado por run_eval.py)
    report.md             # Informe en markdown (generado por run_eval.py)
```

### Formato de evento en `events.jsonl`

```json
{
  "query_id": "r01",
  "timestamp": "2026-04-24T10:15:32.123Z",
  "run_id": "eval_v1",
  "configuration": "full_pipeline",
  "query": "...",
  "route": {"predicted": "unstructured", "confidence": 0.97, "method": "llm"},
  "phases": {
    "embedding_ms": 62.3,
    "bm25_ms": 18.1,
    "vector_ms": 31.4,
    "rrf_ms": 2.8,
    "preprocessor_ms": 3.5,
    "reranker_ms": 924.1,
    "synthesis_ms": 32145.7,
    "synthesis_tokens_in": 2341,
    "synthesis_tokens_out": 487,
    "total_ms": 33187.9
  },
  "crag": null,
  "sql": null,
  "retrieved_docs": [...],
  "response": "...",
  "error": null
}
```

## Nota sobre dataset_legacy.json

El fichero `dataset_legacy.json` contiene las 38 consultas originales generadas automáticamente con LLMs en la primera versión. **No debe usarse para evaluación** porque tiene sesgo de co-generación (vocabulario alineado con el corpus que infla artificialmente las métricas de retrieval). Está archivado únicamente para trazabilidad histórica.
