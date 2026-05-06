"""
Query Router Module.

Three-layer routing of queries to appropriate data sources:
  1. **Keyword rules** — fast, no LLM call, fires only on high confidence.
  2. **LLM classification** — when rules are inconclusive.
  3. **Post-execution fallback** — handled by the caller (HybridQueryEngine)
     when the primary source returns empty results.

All behaviour is driven by ``RouterConfig`` in the YAML configuration,
keeping the framework domain-agnostic.
"""

import re
import time
import unicodedata
from typing import Optional, Any, Dict
from dataclasses import dataclass
import logging

from rag_framework.config.models import RAGConfig, RouterConfig
from rag_framework.config.models import SourceType as SourceTypeEnum
from rag_framework.exceptions import RoutingError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lower-case and strip diacritics for keyword matching."""
    text = text.lower()
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


_OVERRIDE_RE = re.compile(
    r"(?i)\b(NO\s+ESTRUCTURADO|ESTRUCTURADO|HIBRIDO|H[ÍI]BRIDO)\b"
)


@dataclass
class RoutingResult:
    """Result of query routing decision."""

    source: "SourceTypeEnum"
    confidence: float  # 0-1 confidence in the decision
    method: str  # "rules", "llm", "default", "manual_override"
    reasoning: str  # Human-readable explanation
    clean_query: Optional[str] = (
        None  # Query with override tag stripped (None = unchanged)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "source": self.source.value,
            "confidence": self.confidence,
            "method": self.method,
            "reasoning": self.reasoning,
        }


class QueryRouter:
    """
    Three-layer query router that directs queries to the appropriate source.

    Layer 1 – **Keyword rules**: counts keyword hits from the YAML lists
    for structured vs unstructured.  If one side clearly dominates
    (confidence ≥ ``keyword_confidence_threshold``), the decision is
    taken immediately without calling the LLM.

    Layer 2 – **LLM classification**: when rules are inconclusive the
    LLM is asked to classify the query.  The prompt now requests both
    the source label and a confidence score.

    Layer 3 – **Post-execution fallback**: not handled here; see
    ``HybridQueryEngine`` which checks for empty results and retries
    with an alternative source when ``fallback_on_empty`` is enabled.
    """

    # LLM prompt for query classification
    CLASSIFICATION_PROMPT = """Eres un clasificador de consultas para un sistema RAG híbrido.
Tu ÚNICA tarea es decidir qué fuente de datos debe usarse para responder la consulta del usuario.

## Fuentes disponibles

- **STRUCTURED**: Base de datos SQL. Contiene datos cuantitativos organizados en tablas (registros, cifras, relaciones entre entidades). Útil para conteos, listados, agregaciones, filtros, rankings y cualquier pregunta que se responda con una consulta SQL. IMPORTANTE: la base de datos puede contener solo identificadores (IDs, códigos, DNIs) y NO nombres o descripciones textuales completas.
- **UNSTRUCTURED**: Documentos de texto (normativas, políticas, manuales, guías, programas docentes, etc.). Contiene información cualitativa: explicaciones, definiciones, procedimientos, criterios, derechos, plazos, requisitos, nombres de personas, descripciones detalladas y cualquier conocimiento descriptivo o normativo.
- **HYBRID**: Se necesitan AMBAS fuentes. La respuesta requiere datos concretos de la base de datos Y además contexto, explicaciones o normativa de los documentos.

## Esquema de la base de datos

{schema_summary}

## Reglas de clasificación (seguir EN ORDEN)

1. Identifica el TEMA de la consulta.
2. ¿El tema está representado en las tablas del esquema?
   - NO → **UNSTRUCTURED** (no importa cómo esté formulada la pregunta).
   - SÍ → continúa al paso 3.
3. ¿La consulta pide DATOS CONCRETOS que se pueden extraer con SQL (conteos, listados, filtros, rankings, agregaciones)?
   - SÍ → **STRUCTURED**.
   - NO (pide explicaciones, normativa, criterios, definiciones, procedimientos, nombres de personas, detalles cualitativos) → **UNSTRUCTURED**.
4. ¿La información de la base de datos es SUFICIENTE o faltan datos cualitativos que solo aparecen en los documentos?
   - Si la BD solo tiene IDs/códigos pero el usuario necesita nombres, descripciones o contexto → **HYBRID** o **UNSTRUCTURED**.
5. ¿La consulta necesita datos SQL Y además explicaciones o contexto documental?
   - SÍ → **HYBRID**.

## Ejemplos

- "¿Cuántas asignaturas optativas hay?" → STRUCTURED 0.95
- "¿Qué se estudia en Inteligencia Artificial?" → UNSTRUCTURED 0.95
- "¿Cuál es la metodología de evaluación de Fundamentos de Programación?" → UNSTRUCTURED 0.90
- "¿Cuántos créditos tiene IA y qué temas cubre?" → HYBRID 0.85
- "¿Qué departamentos tienen más asignaturas?" → STRUCTURED 0.90
- "¿Cuáles son los objetivos de aprendizaje de Estadística?" → UNSTRUCTURED 0.95
- "¿Qué profesores imparten clase en el grupo 1?" → HYBRID 0.80
- "¿Cuántas asignaturas hay por curso?" → STRUCTURED 0.95
- "¿Cuál es el temario de Redes de Computadores?" → UNSTRUCTURED 0.90
- "¿Cuántos créditos tiene Fundamentos de Programación y cómo se evalúa?" → HYBRID 0.85

## Consulta del usuario

{query}

Responde con el formato EXACTO: FUENTE CONFIANZA (ejemplo: STRUCTURED 0.90)"""

    def __init__(
        self,
        config: RAGConfig,
        llm: Optional[Any] = None,
        schema_info: Optional[Dict] = None,
    ):
        """
        Initialize the query router.

        Args:
            config: RAG configuration including router settings
            llm: LLM instance for classification (optional, lazy-loaded)
            schema_info: Database schema information for context
        """
        self.config = config
        self.router_config = config.router
        self._llm = llm
        self._schema_info = schema_info or {}

        logger.info("QueryRouter initialized - using LLM-based routing")

    @property
    def llm(self) -> Optional[Any]:
        """Lazy-load LLM if not provided."""
        if self._llm is None:
            try:
                from rag_framework.providers.llm import LLMFactory

                self._llm = LLMFactory.get_llm(self.config.llm)
            except Exception as e:
                logger.warning(f"Could not load LLM for routing: {e}")
        return self._llm

    def update_schema_info(self, schema_info: Dict) -> None:
        """Update the database schema information."""
        self._schema_info = schema_info
        logger.debug(f"Schema info updated: {list(schema_info.keys())}")

    # =========================================================================
    # Main entry point
    # =========================================================================

    def route(self, query: str, trace=None) -> RoutingResult:
        """
        Route a query through the 3-layer strategy.

        Layer 1: keyword rules (fast, no LLM).
        Layer 2: LLM classification (if rules inconclusive).
        Fallback: default source (if both layers fail).
        """
        # --- Layer 0: manual override for testing ---
        override = self._check_manual_override(query)
        if override is not None:
            if trace is not None:
                trace.route = override.to_dict()
            return override

        if not self.router_config.enabled:
            result = self._create_default_result(query, "Router disabled")
            if trace is not None:
                trace.route = result.to_dict()
            return result

        if not self.config.sql.enabled:
            result = RoutingResult(
                source=SourceTypeEnum.UNSTRUCTURED,
                confidence=1.0,
                method="default",
                reasoning="SQL not enabled, using document retrieval",
            )
            if trace is not None:
                trace.route = result.to_dict()
            return result

        # --- Layer 1: keyword rules ---
        if self.router_config.use_keyword_routing:
            t0 = time.perf_counter()
            result = self._route_by_rules(query)
            if trace is not None:
                trace.phases["router_keyword_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            if result is not None:
                if trace is not None:
                    trace.route = result.to_dict()
                return result

        # --- Layer 2: LLM classification ---
        t0 = time.perf_counter()
        result = self._route_by_llm(query)
        if trace is not None:
            trace.phases["router_llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if result is not None:
            if trace is not None:
                trace.route = result.to_dict()
            return result

        # --- Fallback: default source ---
        logger.warning("All routing layers failed, using default source")
        fallback = self._create_default_result(query, "LLM unavailable, using default")
        if trace is not None:
            trace.route = fallback.to_dict()
        return fallback

    # =========================================================================
    # Layer 0: manual override (testing helper)
    # =========================================================================

    def _check_manual_override(self, query: str) -> Optional[RoutingResult]:
        """Detect ESTRUCTURADO / NO ESTRUCTURADO tags in the query.

        If present the tag is stripped and the cleaned query is stored in
        ``RoutingResult.clean_query`` so downstream code uses it instead
        of the original.  This is intended purely for developer testing.
        """
        match = _OVERRIDE_RE.search(query)
        if match is None:
            return None

        tag = _normalise(
            match.group(1)
        )  # "estructurado", "no estructurado" or "hibrido"
        cleaned = _OVERRIDE_RE.sub("", query).strip()
        # Collapse any double spaces left behind
        cleaned = re.sub(r"\s{2,}", " ", cleaned)

        if "hibrido" in tag:
            source = SourceTypeEnum.HYBRID
        elif "no" in tag:
            source = SourceTypeEnum.UNSTRUCTURED
        else:
            source = SourceTypeEnum.STRUCTURED

        logger.info(
            "Manual override detected → %s (cleaned query: %r)",
            source.value,
            cleaned,
        )

        return RoutingResult(
            source=source,
            confidence=1.0,
            method="manual_override",
            reasoning=f"Manual override tag '{match.group(1)}' detected",
            clean_query=cleaned,
        )

    # =========================================================================
    # Layer 1: keyword-based routing
    # =========================================================================

    def _route_by_rules(self, query: str) -> Optional[RoutingResult]:
        """Fast keyword-based routing.

        Counts hits from ``structured_keywords`` and ``unstructured_keywords``
        in the normalised query.  Only returns a decision when one side
        clearly dominates (confidence ≥ ``keyword_confidence_threshold``).
        Returns ``None`` when the signal is too weak — causing Layer 2
        (LLM) to take over.
        """
        struct_kws = self.router_config.structured_keywords
        unstruct_kws = self.router_config.unstructured_keywords

        if not struct_kws and not unstruct_kws:
            return None

        norm_query = _normalise(query)

        struct_hits = sum(1 for kw in struct_kws if _normalise(kw) in norm_query)
        unstruct_hits = sum(1 for kw in unstruct_kws if _normalise(kw) in norm_query)

        total_hits = struct_hits + unstruct_hits
        if total_hits == 0:
            return None

        struct_ratio = struct_hits / total_hits
        unstruct_ratio = unstruct_hits / total_hits
        threshold = self.router_config.keyword_confidence_threshold

        # Both sides match → ambiguous → let LLM decide
        if struct_hits > 0 and unstruct_hits > 0:
            # Only decide if one side is overwhelmingly dominant
            if struct_ratio >= threshold:
                source = SourceTypeEnum.STRUCTURED
            elif unstruct_ratio >= threshold:
                source = SourceTypeEnum.UNSTRUCTURED
            else:
                logger.debug(
                    "Keyword routing ambiguous (struct=%.2f, unstruct=%.2f), "
                    "deferring to LLM",
                    struct_ratio,
                    unstruct_ratio,
                )
                return None
        elif struct_hits > 0:
            source = SourceTypeEnum.STRUCTURED
            struct_ratio = 1.0
        else:
            source = SourceTypeEnum.UNSTRUCTURED
            unstruct_ratio = 1.0

        confidence = max(struct_ratio, unstruct_ratio)

        # Only accept if confidence meets threshold
        if confidence < threshold:
            logger.debug(
                "Keyword routing confidence %.2f < threshold %.2f, deferring to LLM",
                confidence,
                threshold,
            )
            return None

        logger.info(
            "Query routed to %s by keyword rules (confidence: %.2f, "
            "struct_hits=%d, unstruct_hits=%d)",
            source.value,
            confidence,
            struct_hits,
            unstruct_hits,
        )

        return RoutingResult(
            source=source,
            confidence=confidence,
            method="rules",
            reasoning=(
                f"Keyword routing: {struct_hits} structured hits, "
                f"{unstruct_hits} unstructured hits"
            ),
        )

    # =========================================================================
    # Layer 2: LLM-based classification
    # =========================================================================

    def _route_by_llm(self, query: str) -> Optional[RoutingResult]:
        """Use the LLM to classify the query.

        The prompt requests the format ``FUENTE CONFIANZA``
        (e.g.  ``STRUCTURED 0.90``).  If the confidence value cannot
        be parsed, 0.80 is used as a safe default.
        """
        if not self.llm:
            logger.warning("LLM not available for routing")
            return None

        try:
            schema_summary = self._build_schema_summary()
            prompt = self.CLASSIFICATION_PROMPT.format(
                schema_summary=schema_summary,
                query=query,
            )

            response = self.llm.complete(prompt)
            response_text = str(response).strip()

            logger.debug(f"LLM routing response: {response_text}")

            source, confidence = self._parse_llm_response(response_text)
            if source is None:
                return None

            # If confidence is below threshold, nudge towards HYBRID
            if (
                confidence < self.router_config.confidence_threshold
                and source != SourceTypeEnum.HYBRID
            ):
                logger.info(
                    "LLM confidence %.2f < threshold %.2f — upgrading to HYBRID",
                    confidence,
                    self.router_config.confidence_threshold,
                )
                source = SourceTypeEnum.HYBRID

            return RoutingResult(
                source=source,
                confidence=confidence,
                method="llm",
                reasoning=f"LLM classified query as {source.value}",
            )

        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            return None

    def _parse_llm_response(self, text: str) -> tuple:
        """Parse LLM response into (SourceTypeEnum, confidence).

        Expected format: ``STRUCTURED 0.90`` or just ``STRUCTURED``.
        Returns (None, 0.0) if unparseable.
        """
        upper = text.upper().strip()

        # Try to extract confidence number
        confidence = 0.80  # safe default
        match = re.search(r"(0\.\d+)", text)
        if match:
            confidence = float(match.group(1))

        # Determine source (check HYBRID and UNSTRUCTURED before STRUCTURED
        # to avoid substring false-positive)
        if "HYBRID" in upper:
            return SourceTypeEnum.HYBRID, confidence
        elif "UNSTRUCTURED" in upper:
            return SourceTypeEnum.UNSTRUCTURED, confidence
        elif "STRUCTURED" in upper:
            return SourceTypeEnum.STRUCTURED, confidence

        logger.warning(f"Could not parse LLM routing response: {text}")
        return None, 0.0

    # =========================================================================
    # Schema summary builder
    # =========================================================================

    def _build_schema_summary(self) -> str:
        """Build a detailed schema summary for LLM context."""
        if not self._schema_info:
            return "No hay esquema de base de datos disponible."

        table_details = self._schema_info.get("table_details", [])
        if not table_details:
            tables = self._schema_info.get("tables", [])
            if not tables:
                return "No hay tablas configuradas en la base de datos."
            return "Tablas disponibles: " + ", ".join(tables)

        lines = []
        for table in table_details:
            name = table.get("name", "unknown")
            desc = table.get("description", "")
            row_count = table.get("row_count")
            columns = table.get("columns", [])

            header = f"TABLE {name}"
            if desc:
                header += f"  -- {desc}"
            if row_count is not None:
                header += f"  (~{row_count} filas)"
            lines.append(header)

            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("name", "?")
                    col_type = col.get("type", "")
                    parts = [f"  {col_name}"]
                    if col_type:
                        parts[0] += f" ({col_type})"
                    if col.get("primary_key"):
                        parts.append("PK")
                    fk = col.get("foreign_key")
                    if fk:
                        parts.append(f"FK→{fk}")
                    lines.append(" ".join(parts))
                else:
                    lines.append(f"  {col}")

            lines.append("")  # blank line between tables

        return "\n".join(lines)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _create_default_result(self, query: str, reason: str) -> RoutingResult:
        """Create a default routing result."""
        default = SourceTypeEnum(self.router_config.default_source)
        return RoutingResult(
            source=default,
            confidence=1.0,
            method="default",
            reasoning=reason,
        )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def is_structured_query(self, query: str) -> bool:
        """Quick check if query needs structured data."""
        result = self.route(query)
        return result.source in (SourceTypeEnum.STRUCTURED, SourceTypeEnum.HYBRID)

    def is_unstructured_query(self, query: str) -> bool:
        """Quick check if query needs unstructured data."""
        result = self.route(query)
        return result.source in (SourceTypeEnum.UNSTRUCTURED, SourceTypeEnum.HYBRID)

    def is_hybrid_query(self, query: str) -> bool:
        """Quick check if query needs both sources."""
        result = self.route(query)
        return result.source == SourceTypeEnum.HYBRID
