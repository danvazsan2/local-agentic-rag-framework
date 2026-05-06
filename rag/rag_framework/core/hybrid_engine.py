"""
Hybrid Query Engine Module.

Orchestrates queries that require multiple data sources (SQL + Documents),
combining results and synthesizing coherent responses.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag_framework.config.models import RAGConfig, SourceType
from rag_framework.routing import QueryRouter, RoutingResult
from rag_framework.sql import SQLAgent, SQLQueryResult
from rag_framework.exceptions import RAGFrameworkError

logger = logging.getLogger(__name__)


@dataclass
class SourceResult:
    """Result from a single data source."""

    source: SourceType
    success: bool
    content: str = ""  # Formatted content for synthesis
    raw_data: Any = None  # Original result object
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class HybridQueryResponse:
    """
    Response from hybrid query execution.

    Contains results from all queried sources, routing decision,
    and final synthesized response.
    """

    # Final response
    response: str

    # Routing information
    routing: RoutingResult

    # Individual source results
    source_results: Dict[SourceType, SourceResult] = field(default_factory=dict)

    # Metadata
    total_time_ms: float = 0.0
    sources_used: List[str] = field(default_factory=list)

    def get_sources_summary(self) -> str:
        """Get human-readable summary of sources used."""
        if not self.sources_used:
            return "No sources queried"

        parts = []
        for source in self.sources_used:
            result = self.source_results.get(SourceType(source))
            if result and result.success:
                parts.append(f"[OK] {source}")
            elif result:
                parts.append(f"[FAIL] {source}")

        return "Sources: " + ", ".join(parts)


class HybridQueryEngine:
    """
    Orchestrates hybrid queries across multiple data sources.

    This engine:
    1. Uses the router to determine which sources to query
    2. Executes queries against appropriate sources (possibly in parallel)
    3. Combines context from multiple sources
    4. Synthesizes a coherent response using the LLM

    It integrates with the existing RAG pipeline for document queries
    and the new SQL agent for database queries.
    """

    # Prompt template for hybrid synthesis
    # CHANGED: Enriched with source priority rules and citation instructions
    HYBRID_SYNTHESIS_PROMPT = """Eres un asistente experto que integra información de MÚLTIPLES FUENTES para responder preguntas del usuario.

INSTRUCCIONES DE SÍNTESIS:
1. **Prioridad**: Los DATOS CUANTITATIVOS de la base de datos (números, conteos, listados) son la fuente de verdad para cifras exactas.
2. **Contexto**: Los DOCUMENTOS proporcionan explicaciones, normativas, procedimientos y contexto cualitativo.
3. **Integración**: Combina ambas fuentes de manera coherente. Si hay cifras de la BD, úsalas; si hay explicaciones en documentos, inclúyelas.
4. **Citación**: Indica la fuente cuando menciones información específica (ej: "Según la base de datos, hay X registros" o "El documento indica que...").
5. **Coherencia**: Si ambas fuentes contienen información complementaria, intégralas en una respuesta unificada.
6. **Conflictos**: Si hay contradicciones entre fuentes, prioriza los datos de la base de datos para cifras y los documentos para normativas/procedimientos.

{context}

PREGUNTA DEL USUARIO: {question}

RESPUESTA INTEGRADA:"""

    # Prompt for SQL-only synthesis
    SQL_SYNTHESIS_PROMPT = """Basándote en los siguientes resultados de la consulta de base de datos, responde a la pregunta del usuario.

{context}

PREGUNTA DEL USUARIO: {question}

Proporciona una respuesta clara y concisa basada en los datos. Incluye números y detalles relevantes.

RESPUESTA:"""

    def __init__(
        self,
        config: RAGConfig,
        rag_query_engine: Any = None,  # Existing RAG query engine
        llm: Any = None,
    ):
        """
        Initialize hybrid query engine.

        Args:
            config: RAG configuration
            rag_query_engine: Existing document RAG query engine
            llm: LLM for synthesis
        """
        self.config = config
        self._rag_engine = rag_query_engine
        self._llm = llm

        # Initialize components (lazy loaded)
        self._router: Optional[QueryRouter] = None
        self._sql_agent: Optional[SQLAgent] = None

        # Parallel execution settings
        self._use_parallel = True
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info("HybridQueryEngine initialized")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def llm(self):
        """Lazy-load LLM."""
        if self._llm is None:
            from rag_framework.providers.llm import LLMFactory

            self._llm = LLMFactory.get_llm(self.config.llm)
        return self._llm

    @property
    def router(self) -> QueryRouter:
        """Lazy-load router."""
        if self._router is None:
            self._router = QueryRouter(
                config=self.config,
                llm=self.llm if self.config.router.use_llm_fallback else None,
            )

            # Update router with SQL schema if available
            if self.config.sql.enabled and self._sql_agent:
                schema_info = self._sql_agent.get_schema_for_router()
                self._router.update_schema_info(schema_info)

        return self._router

    @property
    def sql_agent(self) -> Optional[SQLAgent]:
        """Lazy-load SQL agent."""
        if self._sql_agent is None and self.config.sql.enabled:
            self._sql_agent = SQLAgent(self.config, llm=self.llm)

            # Update router with schema info
            if self._router:
                schema_info = self._sql_agent.get_schema_for_router()
                self._router.update_schema_info(schema_info)

        return self._sql_agent

    def set_rag_engine(self, engine: Any) -> None:
        """Set the RAG query engine for document queries."""
        self._rag_engine = engine

    # =========================================================================
    # Main Query Interface
    # =========================================================================

    def query(self, question: str, trace=None) -> HybridQueryResponse:
        """
        Execute a query using intelligent routing and source orchestration.

        This is the main entry point for hybrid queries. It:
        1. Routes the query to appropriate source(s)
        2. Executes queries against selected sources
        3. Synthesizes results into a coherent response

        Args:
            question: User's natural language question
            trace: Optional QueryTrace for per-phase instrumentation

        Returns:
            HybridQueryResponse with full results and metadata
        """
        start_time = datetime.now()

        # Step 1: Route the query
        routing = self.router.route(question, trace=trace)

        # If the router stripped an override tag, use the cleaned query
        if routing.clean_query is not None:
            question = routing.clean_query

        logger.info(
            f"Query routed to {routing.source.value} "
            f"(confidence: {routing.confidence:.2f}, method: {routing.method})"
        )

        # Step 2: Execute based on routing decision
        if routing.source == SourceType.STRUCTURED:
            return self._execute_structured_query(question, routing, start_time, trace=trace)

        elif routing.source == SourceType.UNSTRUCTURED:
            return self._execute_unstructured_query(question, routing, start_time, trace=trace)

        else:  # HYBRID
            return self._execute_hybrid_query(question, routing, start_time, trace=trace)

    # =========================================================================
    # Source-Specific Execution
    # =========================================================================

    def _execute_structured_query(
        self,
        question: str,
        routing: RoutingResult,
        start_time: datetime,
        trace=None,
    ) -> HybridQueryResponse:
        """Execute SQL-only query with optional fallback on empty results."""
        source_results = {}

        # Execute SQL query
        sql_result = self._query_sql(question, trace=trace)
        source_results[SourceType.STRUCTURED] = sql_result

        if sql_result.success:
            # Check for 0-row results + fallback configuration
            empty_result = self._is_empty_sql_result(sql_result)
            fallback_cfg = self.config.router

            if empty_result and fallback_cfg.fallback_on_empty:
                logger.info(
                    "SQL returned 0 rows — applying fallback strategy: %s",
                    fallback_cfg.fallback_strategy,
                )
                return self._apply_fallback(
                    question, routing, source_results, start_time, trace=trace
                )

            # Normal path: synthesize from SQL
            context = sql_result.content
            response = self._synthesize_sql_response(question, context)
        else:
            # Fall back to unstructured if SQL fails
            logger.warning(
                f"SQL query failed: {sql_result.error}, falling back to documents"
            )
            rag_result = self._query_documents(question)
            source_results[SourceType.UNSTRUCTURED] = rag_result

            if rag_result.success:
                response = rag_result.content  # RAG already synthesizes
            else:
                response = f"Unable to answer: SQL error ({sql_result.error})"

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return HybridQueryResponse(
            response=response,
            routing=routing,
            source_results=source_results,
            total_time_ms=total_time,
            sources_used=[
                s.value for s in source_results.keys() if source_results[s].success
            ],
        )

    def _is_empty_sql_result(self, sql_result: "SourceResult") -> bool:
        """Check whether an SQL SourceResult succeeded but returned 0 rows."""
        raw = getattr(sql_result, "raw_data", None)
        if raw is None:
            return False
        query_result = getattr(raw, "result", None)
        if query_result is None:
            return False
        return getattr(query_result, "row_count", -1) == 0

    def _apply_fallback(
        self,
        question: str,
        routing: RoutingResult,
        source_results: dict,
        start_time: datetime,
        trace=None,
    ) -> HybridQueryResponse:
        """Apply the configured fallback strategy when SQL returns 0 rows."""
        strategy = self.config.router.fallback_strategy

        if strategy == "try_hybrid":
            # Execute both SQL (already done) + documents
            rag_result = self._query_documents(question, trace=trace)
            source_results[SourceType.UNSTRUCTURED] = rag_result

            sql_result = source_results.get(SourceType.STRUCTURED)
            if rag_result.success:
                response = self._synthesize_hybrid_response(
                    question, sql_result, rag_result
                )
            else:
                response = self._synthesize_sql_response(
                    question, sql_result.content if sql_result else ""
                )
        else:
            # Default: "try_unstructured" — query documents only
            rag_result = self._query_documents(question, trace=trace)
            source_results[SourceType.UNSTRUCTURED] = rag_result

            if rag_result.success:
                response = rag_result.content
            else:
                # Last resort: return the (empty) SQL result as-is
                sql_result = source_results.get(SourceType.STRUCTURED)
                context = sql_result.content if sql_result else ""
                response = self._synthesize_sql_response(question, context)

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return HybridQueryResponse(
            response=response,
            routing=routing,
            source_results=source_results,
            total_time_ms=total_time,
            sources_used=[
                s.value for s in source_results.keys() if source_results[s].success
            ],
        )

    def _execute_unstructured_query(
        self,
        question: str,
        routing: RoutingResult,
        start_time: datetime,
        trace=None,
    ) -> HybridQueryResponse:
        """Execute document-only query."""
        source_results = {}

        # Execute RAG query
        rag_result = self._query_documents(question, trace=trace)
        source_results[SourceType.UNSTRUCTURED] = rag_result

        if rag_result.success:
            response = rag_result.content  # RAG already synthesizes
        else:
            response = f"Unable to retrieve documents: {rag_result.error}"

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return HybridQueryResponse(
            response=response,
            routing=routing,
            source_results=source_results,
            total_time_ms=total_time,
            sources_used=[SourceType.UNSTRUCTURED.value] if rag_result.success else [],
        )

    def _execute_hybrid_query(
        self,
        question: str,
        routing: RoutingResult,
        start_time: datetime,
        trace=None,
    ) -> HybridQueryResponse:
        """Execute query against both sources and combine results."""
        source_results = {}

        if self._use_parallel:
            # Execute in parallel (trace passed to both; shared dict is thread-safe for
            # disjoint keys since SQL writes trace.sql and RAG writes trace.phases/crag)
            sql_future = self._executor.submit(self._query_sql, question, trace)
            rag_future = self._executor.submit(self._query_documents, question, trace)

            sql_result = sql_future.result()
            rag_result = rag_future.result()
        else:
            # Sequential execution
            sql_result = self._query_sql(question, trace=trace)
            rag_result = self._query_documents(question, trace=trace)

        source_results[SourceType.STRUCTURED] = sql_result
        source_results[SourceType.UNSTRUCTURED] = rag_result

        # Combine and synthesize
        response = self._synthesize_hybrid_response(
            question,
            sql_result,
            rag_result,
        )

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        sources_used = []
        if sql_result.success:
            sources_used.append(SourceType.STRUCTURED.value)
        if rag_result.success:
            sources_used.append(SourceType.UNSTRUCTURED.value)

        return HybridQueryResponse(
            response=response,
            routing=routing,
            source_results=source_results,
            total_time_ms=total_time,
            sources_used=sources_used,
        )

    # =========================================================================
    # Individual Source Queries
    # =========================================================================

    def _query_sql(self, question: str, trace=None) -> SourceResult:
        """Query the SQL database."""
        start_time = datetime.now()

        if not self.sql_agent:
            return SourceResult(
                source=SourceType.STRUCTURED,
                success=False,
                error="SQL agent not available",
            )

        try:
            result = self.sql_agent.query(question, trace=trace)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            if result.success:
                return SourceResult(
                    source=SourceType.STRUCTURED,
                    success=True,
                    content=result.get_context_for_llm(),
                    raw_data=result,
                    execution_time_ms=execution_time,
                )
            else:
                return SourceResult(
                    source=SourceType.STRUCTURED,
                    success=False,
                    error=result.error,
                    raw_data=result,
                    execution_time_ms=execution_time,
                )

        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            return SourceResult(
                source=SourceType.STRUCTURED,
                success=False,
                error=str(e),
            )

    def _query_documents(self, question: str, trace=None) -> SourceResult:
        """Query the document RAG system."""
        start_time = datetime.now()

        if not self._rag_engine:
            return SourceResult(
                source=SourceType.UNSTRUCTURED,
                success=False,
                error="RAG engine not available",
            )

        try:
            if self.config.debug:
                print("\n[Debug] Iniciando pipeline RAG (documentos)...")
            response = self._rag_engine.query(question, trace=trace)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            if self.config.debug:
                print(f"[Debug] Pipeline RAG completado en {execution_time:.0f}ms")

            return SourceResult(
                source=SourceType.UNSTRUCTURED,
                success=True,
                content=str(response),
                raw_data=response,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"Document query failed: {e}")
            return SourceResult(
                source=SourceType.UNSTRUCTURED,
                success=False,
                error=str(e),
            )

    # =========================================================================
    # Response Synthesis
    # =========================================================================

    def _synthesize_sql_response(self, question: str, context: str) -> str:
        """Synthesize response from SQL results only."""
        prompt = self.SQL_SYNTHESIS_PROMPT.format(
            context=context,
            question=question,
        )

        try:
            response = self.llm.complete(prompt)
            return str(response).strip()
        except Exception as e:
            logger.error(f"SQL synthesis failed: {e}")
            return f"Data retrieved but synthesis failed: {context}"

    def _synthesize_hybrid_response(
        self,
        question: str,
        sql_result: SourceResult,
        rag_result: SourceResult,
    ) -> str:
        """Synthesize response combining both sources."""
        # Build combined context
        context_parts = []

        if sql_result.success:
            context_parts.append(f"=== DATABASE RESULTS ===\n{sql_result.content}")
        elif sql_result.error:
            context_parts.append(
                f"=== DATABASE ===\n(Query failed: {sql_result.error})"
            )

        if rag_result.success:
            context_parts.append(f"=== DOCUMENT CONTEXT ===\n{rag_result.content}")
        elif rag_result.error:
            context_parts.append(
                f"=== DOCUMENTS ===\n(Retrieval failed: {rag_result.error})"
            )

        if not context_parts:
            return "Unable to retrieve information from any source."

        # Truncate context if too long (preserve token budget)
        combined_context = "\n\n".join(context_parts)
        max_context_chars = 6000  # Rough estimate for ~2000 tokens
        if len(combined_context) > max_context_chars:
            combined_context = self._truncate_context(
                combined_context, max_context_chars
            )

        prompt = self.HYBRID_SYNTHESIS_PROMPT.format(
            context=combined_context,
            question=question,
        )

        try:
            response = self.llm.complete(prompt)
            return str(response).strip()
        except Exception as e:
            logger.error(f"Hybrid synthesis failed: {e}")
            # Return raw context as fallback
            return f"Synthesis failed. Raw results:\n{combined_context}"

    def _truncate_context(self, context: str, max_chars: int) -> str:
        """Truncate context while preserving structure and key information.

        Prioritizes SQL results over documents when space is limited,
        and truncates documents at paragraph boundaries to preserve
        semantic coherence.
        """
        if len(context) <= max_chars:
            return context

        # CHANGED: Semantic-aware truncation with source prioritization
        # Separate SQL and document sections
        sql_section = None
        doc_section = None
        other_parts = []

        for section in context.split("==="):
            section_stripped = section.strip()
            if not section_stripped:
                continue
            if "DATABASE" in section or "BASE DE DATOS" in section or "SQL" in section:
                sql_section = section
            elif "DOCUMENT" in section or "DOCUMENTOS" in section:
                doc_section = section
            else:
                other_parts.append(section)

        result = []
        current_len = 0

        # 1. ALWAYS include SQL results if available (priority: 60% of budget)
        if sql_section:
            sql_budget = int(max_chars * 0.6)
            if len(sql_section) <= sql_budget:
                result.append(sql_section)
                current_len += len(sql_section)
            else:
                # Truncate SQL preserving first rows
                result.append(sql_section[:sql_budget] + "\n... [truncated]")
                current_len += sql_budget

        # 2. Add document context if space remains
        if doc_section:
            remaining = max_chars - current_len - 20  # margin for separators
            if remaining > 200:
                if len(doc_section) <= remaining:
                    result.append(doc_section)
                else:
                    # Truncate by complete paragraphs
                    paragraphs = doc_section.split("\n\n")
                    truncated_doc = ""
                    for para in paragraphs:
                        if len(truncated_doc) + len(para) + 2 <= remaining:
                            truncated_doc += para + "\n\n"
                        else:
                            break
                    result.append(truncated_doc.rstrip() + "\n... [truncated]")

        # 3. Add any other parts if there's still space
        for part in other_parts:
            remaining = max_chars - current_len - 20
            if remaining > 100 and len(part) <= remaining:
                result.append(part)
                current_len += len(part)

        return "===".join(result) if result else context[:max_chars]

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Clean up resources."""
        if self._sql_agent:
            self._sql_agent.close()
        if self._executor:
            self._executor.shutdown(wait=False)
        logger.info("HybridQueryEngine resources released")
