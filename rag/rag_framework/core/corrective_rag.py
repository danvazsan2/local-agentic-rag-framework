"""
Corrective RAG (CRAG) Module.

Implements the Corrective Retrieval-Augmented Generation pattern,
which adds a document grading step between retrieval and synthesis.

The CRAG pipeline:
1. Retrieve documents (standard retrieval + optional reranking)
2. Grade each document for relevance using the LLM
3. If relevant documents are insufficient:
   a. Rewrite the query for better retrieval
   b. Re-retrieve with the rewritten query
   c. Optionally perform web search as fallback
4. Pass only relevant documents to synthesis

This module provides:
- RelevanceGrader: Uses the LLM to grade document relevance
- QueryRewriter: Uses the LLM to rewrite ambiguous/ineffective queries
- CorrectiveRAGEngine: Orchestrates the full CRAG pipeline

Reference:
    Yan et al., "Corrective Retrieval Augmented Generation" (2024)
"""

import logging
import time
from enum import Enum
from typing import List, Optional, Any, Tuple
from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore

from rag_framework.config.models import RAGConfig

logger = logging.getLogger(__name__)


# =========================================================================
# Data Models
# =========================================================================


class RelevanceGrade(str, Enum):
    """Relevance grade for a retrieved document."""

    RELEVANT = "RELEVANT"
    IRRELEVANT = "IRRELEVANT"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class GradedDocument:
    """A retrieved document with its relevance grade."""

    node: NodeWithScore
    grade: RelevanceGrade
    reasoning: str = ""


@dataclass
class CorrectiveRAGResult:
    """Result of the Corrective RAG grading and filtering pipeline."""

    # Final filtered nodes passed to synthesis
    filtered_nodes: List[NodeWithScore]

    # Grading details
    graded_documents: List[GradedDocument]
    total_retrieved: int
    relevant_count: int
    irrelevant_count: int
    ambiguous_count: int

    # Whether a query rewrite was performed
    query_rewritten: bool = False
    rewritten_query: Optional[str] = None

    # Whether web search was used
    web_search_used: bool = False

    @property
    def relevance_ratio(self) -> float:
        """Fraction of documents graded as relevant."""
        if self.total_retrieved == 0:
            return 0.0
        return self.relevant_count / self.total_retrieved


# =========================================================================
# Prompts
# =========================================================================

RELEVANCE_GRADING_PROMPT = """Eres un evaluador experto de relevancia documental. Tu tarea es determinar si un documento recuperado es relevante para responder la pregunta del usuario.

### INSTRUCCIONES:
- Evalúa si el documento contiene información que ayude a responder la pregunta.
- No es necesario que el documento responda COMPLETAMENTE la pregunta, basta con que aporte información útil y relacionada.
- Responde con UNA SOLA palabra: RELEVANT, IRRELEVANT o AMBIGUOUS.
  - RELEVANT: El documento contiene información directamente útil para la pregunta.
  - IRRELEVANT: El documento NO contiene información relevante para la pregunta.
  - AMBIGUOUS: El documento puede tener relación tangencial pero no es claramente útil.

### PREGUNTA DEL USUARIO:
{query}

### DOCUMENTO A EVALUAR:
{document}

### VEREDICTO (una sola palabra: RELEVANT, IRRELEVANT o AMBIGUOUS):"""

QUERY_REWRITE_PROMPT = """Eres un experto en reformulación de consultas para sistemas de búsqueda. La consulta original del usuario no obtuvo suficientes documentos relevantes.

Tu tarea es reescribir la consulta para mejorar la recuperación de documentos relevantes.

### INSTRUCCIONES:
- Mantén la intención original de la pregunta.
- Usa sinónimos o términos alternativos que puedan aparecer en los documentos.
- Haz la consulta más específica si es demasiado vaga, o más general si es demasiado restrictiva.
- Responde ÚNICAMENTE con la consulta reformulada, sin explicaciones.

### CONSULTA ORIGINAL:
{query}

### CONSULTA REFORMULADA:"""


# =========================================================================
# RelevanceGrader
# =========================================================================


class RelevanceGrader:
    """
    Grades retrieved documents for relevance using the LLM.

    The grader sends each document to the LLM with a grading prompt
    and parses the response to determine relevance.
    """

    def __init__(self, llm: Any, debug: bool = False):
        """
        Initialize the relevance grader.

        Args:
            llm: LLM instance for grading
            debug: Enable debug logging
        """
        self.llm = llm
        self.debug = debug

    def grade_document(self, query: str, node: NodeWithScore) -> GradedDocument:
        """
        Grade a single document for relevance to the query.

        Args:
            query: The user's query
            node: The retrieved document node

        Returns:
            GradedDocument with relevance grade
        """
        # Extract text content from the node
        text = (
            node.node.get_content()
            if hasattr(node.node, "get_content")
            else str(node.node)
        )

        # Truncate very long documents to avoid exceeding context window
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        prompt = RELEVANCE_GRADING_PROMPT.format(query=query, document=text)

        try:
            response = self.llm.complete(prompt)
            response_text = str(response).strip().upper()

            grade = self._parse_grade(response_text)

            if self.debug:
                filename = (
                    node.node.metadata.get("file_name", "Unknown")
                    if hasattr(node.node, "metadata")
                    else "Unknown"
                )
                logger.debug(f"[CRAG] Graded document ({filename}): {grade.value}")

            return GradedDocument(
                node=node,
                grade=grade,
                reasoning=response_text[:100],
            )

        except Exception as e:
            logger.warning(f"[CRAG] Failed to grade document: {e}")
            # On failure, give benefit of the doubt
            return GradedDocument(
                node=node,
                grade=RelevanceGrade.AMBIGUOUS,
                reasoning=f"Grading failed: {e}",
            )

    def grade_documents(
        self,
        query: str,
        nodes: List[NodeWithScore],
        chunk_timings: Optional[List[float]] = None,
    ) -> List[GradedDocument]:
        """
        Grade all retrieved documents for relevance.

        Args:
            query: The user's query
            nodes: List of retrieved document nodes
            chunk_timings: Optional list to append per-chunk grading ms into

        Returns:
            List of GradedDocument with relevance grades
        """
        logger.info(f"[CRAG] Evaluando relevancia de {len(nodes)} documentos...")
        graded = []
        for i, node in enumerate(nodes):
            t0 = time.time()
            graded_doc = self.grade_document(query, node)
            elapsed = time.time() - t0
            graded.append(graded_doc)

            if chunk_timings is not None:
                chunk_timings.append(round(elapsed * 1000, 2))

            if self.debug:
                filename = (
                    node.node.metadata.get("file_name", "Unknown")
                    if hasattr(node.node, "metadata")
                    else "Unknown"
                )
                print(
                    f"    Doc {i + 1}/{len(nodes)}: {graded_doc.grade.value} "
                    f"({elapsed:.2f}s) — {filename}"
                )

        # Log summary
        relevant = sum(1 for g in graded if g.grade == RelevanceGrade.RELEVANT)
        irrelevant = sum(1 for g in graded if g.grade == RelevanceGrade.IRRELEVANT)
        ambiguous = sum(1 for g in graded if g.grade == RelevanceGrade.AMBIGUOUS)

        logger.info(
            f"[CRAG] Evaluación completada: "
            f"{relevant} relevantes, {irrelevant} irrelevantes, "
            f"{ambiguous} ambiguos de {len(nodes)} total"
        )

        return graded

    def _parse_grade(self, response_text: str) -> RelevanceGrade:
        """
        Parse the LLM response to extract a relevance grade.

        Uses tolerant parsing to handle varied LLM outputs.

        Args:
            response_text: Raw LLM response (already uppercased)

        Returns:
            RelevanceGrade enum value
        """
        import re

        # Direct match
        if "IRRELEVANT" in response_text or "NO RELEVANTE" in response_text:
            return RelevanceGrade.IRRELEVANT
        if "RELEVANT" in response_text or "RELEVANTE" in response_text:
            return RelevanceGrade.RELEVANT
        if "AMBIGUOUS" in response_text or "AMBIGUO" in response_text:
            return RelevanceGrade.AMBIGUOUS

        # Fuzzy matching
        if re.search(r"NO\s*(ES\s*)?RELEV|IRRELEV|SIN\s*RELEV", response_text):
            return RelevanceGrade.IRRELEVANT
        if re.search(r"RELEV|ÚTIL|PERTINENTE|APLICABLE", response_text):
            return RelevanceGrade.RELEVANT

        # Default to ambiguous when unclear
        return RelevanceGrade.AMBIGUOUS


# =========================================================================
# QueryRewriter
# =========================================================================


class QueryRewriter:
    """
    Rewrites queries to improve retrieval when initial results are poor.

    Uses the LLM to reformulate the query with alternative terms
    that may better match the document corpus.
    """

    def __init__(self, llm: Any, debug: bool = False):
        """
        Initialize the query rewriter.

        Args:
            llm: LLM instance for query rewriting
            debug: Enable debug logging
        """
        self.llm = llm
        self.debug = debug

    def rewrite(self, query: str) -> str:
        """
        Rewrite a query to improve retrieval.

        Args:
            query: The original user query

        Returns:
            Rewritten query string
        """
        prompt = QUERY_REWRITE_PROMPT.format(query=query)

        try:
            response = self.llm.complete(prompt)
            rewritten = str(response).strip()

            # Clean up: remove quotes if the LLM wrapped the query
            rewritten = rewritten.strip('"').strip("'").strip()

            # Validate: don't accept empty or extremely short rewrites
            if len(rewritten) < 5:
                logger.warning(
                    f"[CRAG] Query rewrite too short ({len(rewritten)} chars), "
                    f"keeping original"
                )
                return query

            logger.info(f"[CRAG] Consulta reescrita: '{rewritten}'")

            if self.debug:
                logger.debug(f"[CRAG] Original: '{query}'")
                logger.debug(f"[CRAG] Reescrita: '{rewritten}'")

            return rewritten

        except Exception as e:
            logger.warning(f"[CRAG] Query rewrite failed: {e}")
            return query


# =========================================================================
# CorrectiveRAGEngine
# =========================================================================


class CorrectiveRAGEngine:
    """
    Orchestrates the Corrective RAG pipeline.

    After standard retrieval and optional reranking, this engine:
    1. Grades each document for relevance
    2. Filters out irrelevant documents
    3. If too few relevant docs remain, rewrites the query and retries
    4. Returns only relevant documents for synthesis

    This engine is designed to be used within DocumentQueryEngine
    when corrective_rag is enabled in the configuration.
    """

    def __init__(
        self,
        config: RAGConfig,
        llm: Any,
        retriever: Any = None,
        node_postprocessors: Optional[List[Any]] = None,
    ):
        """
        Initialize the Corrective RAG engine.

        Args:
            config: RAG configuration
            llm: LLM instance (shared with query engine)
            retriever: Retriever for re-retrieval on query rewrite
            node_postprocessors: Postprocessors (rerankers) for re-retrieval
        """
        self.config = config
        self.crag_config = config.corrective_rag
        self.llm = llm
        self.retriever = retriever
        self.node_postprocessors = node_postprocessors or []

        self.grader = RelevanceGrader(llm=llm, debug=config.debug)
        self.rewriter = QueryRewriter(llm=llm, debug=config.debug)

        logger.info(
            f"[CRAG] CorrectiveRAGEngine inicializado "
            f"(umbral={self.crag_config.relevance_threshold}, "
            f"max_retries={self.crag_config.max_retries})"
        )

    def process(
        self,
        query: str,
        nodes: List[NodeWithScore],
        trace=None,
    ) -> CorrectiveRAGResult:
        """
        Execute the corrective RAG pipeline on retrieved documents.

        Args:
            query: The user's query
            nodes: Initially retrieved (and optionally reranked) nodes
            trace: Optional QueryTrace for per-phase timing

        Returns:
            CorrectiveRAGResult with filtered nodes and grading metadata
        """
        logger.info(
            f"[CRAG] Iniciando evaluación correctiva de {len(nodes)} documentos"
        )

        # Step 1: Grade documents
        chunk_timings: List[float] = []
        graded_documents = self.grader.grade_documents(query, nodes, chunk_timings=chunk_timings)

        # Step 2: Filter based on grades
        relevant_nodes, stats = self._filter_documents(graded_documents)

        # Step 3: Check if we have enough relevant documents
        relevance_ratio = stats["relevant"] / max(len(nodes), 1)

        if (
            relevance_ratio < self.crag_config.relevance_threshold
            and self.crag_config.max_retries > 0
        ):
            logger.info(
                f"[CRAG] Ratio de relevancia ({relevance_ratio:.2f}) bajo el umbral "
                f"({self.crag_config.relevance_threshold}). Reescribiendo consulta..."
            )

            # Try query rewrite + re-retrieval
            t_rewrite = time.time()
            rewritten_query = self.rewriter.rewrite(query)
            rewrite_ms = round((time.time() - t_rewrite) * 1000, 2)

            if rewritten_query != query and self.retriever is not None:
                t_re_retrieval = time.time()
                retry_nodes = self._re_retrieve(rewritten_query)
                re_retrieval_ms = round((time.time() - t_re_retrieval) * 1000, 2)

                if retry_nodes:
                    # Grade the new documents
                    retry_timings: List[float] = []
                    retry_graded = self.grader.grade_documents(
                        rewritten_query, retry_nodes, chunk_timings=retry_timings
                    )
                    chunk_timings.extend(retry_timings)

                    # Merge new relevant documents with existing ones
                    retry_relevant, retry_stats = self._filter_documents(retry_graded)

                    # Combine, deduplicating by node_id
                    existing_ids = {n.node.node_id for n in relevant_nodes}
                    for node in retry_relevant:
                        if node.node.node_id not in existing_ids:
                            relevant_nodes.append(node)
                            existing_ids.add(node.node.node_id)

                    # Update graded_documents list
                    graded_documents.extend(retry_graded)
                    stats["relevant"] += retry_stats["relevant"]
                    stats["irrelevant"] += retry_stats["irrelevant"]
                    stats["ambiguous"] += retry_stats["ambiguous"]

                    logger.info(
                        f"[CRAG] Tras reescritura: {len(relevant_nodes)} "
                        f"documentos relevantes totales"
                    )

                    if trace is not None:
                        trace.crag = {
                            "relevance_ratio": round(relevance_ratio, 3),
                            "rewrite_triggered": True,
                            "rewrite_ms": rewrite_ms,
                            "re_retrieval_ms": re_retrieval_ms,
                            "grading_per_chunk_ms": chunk_timings,
                            "relevant_count": stats["relevant"],
                            "irrelevant_count": stats["irrelevant"],
                            "ambiguous_count": stats["ambiguous"],
                            "rewritten_query": rewritten_query,
                        }

                    return CorrectiveRAGResult(
                        filtered_nodes=relevant_nodes,
                        graded_documents=graded_documents,
                        total_retrieved=len(nodes) + len(retry_nodes),
                        relevant_count=stats["relevant"],
                        irrelevant_count=stats["irrelevant"],
                        ambiguous_count=stats["ambiguous"],
                        query_rewritten=True,
                        rewritten_query=rewritten_query,
                    )

        # If no rewrite needed or rewrite not possible
        logger.info(
            f"[CRAG] Resultado: {len(relevant_nodes)} documentos relevantes "
            f"de {len(nodes)} recuperados"
        )

        if trace is not None:
            trace.crag = {
                "relevance_ratio": round(relevance_ratio, 3),
                "rewrite_triggered": False,
                "rewrite_ms": None,
                "re_retrieval_ms": None,
                "grading_per_chunk_ms": chunk_timings,
                "relevant_count": stats["relevant"],
                "irrelevant_count": stats["irrelevant"],
                "ambiguous_count": stats["ambiguous"],
                "rewritten_query": None,
            }

        return CorrectiveRAGResult(
            filtered_nodes=relevant_nodes,
            graded_documents=graded_documents,
            total_retrieved=len(nodes),
            relevant_count=stats["relevant"],
            irrelevant_count=stats["irrelevant"],
            ambiguous_count=stats["ambiguous"],
        )

    def _filter_documents(
        self,
        graded_documents: List[GradedDocument],
    ) -> Tuple[List[NodeWithScore], dict]:
        """
        Filter documents based on their relevance grade.

        RELEVANT documents are always kept.
        AMBIGUOUS documents are kept (benefit of the doubt).
        IRRELEVANT documents are discarded.

        Args:
            graded_documents: List of graded documents

        Returns:
            Tuple of (filtered nodes, stats dict)
        """
        relevant_nodes = []
        stats = {"relevant": 0, "irrelevant": 0, "ambiguous": 0}

        for graded_doc in graded_documents:
            if graded_doc.grade == RelevanceGrade.RELEVANT:
                relevant_nodes.append(graded_doc.node)
                stats["relevant"] += 1
            elif graded_doc.grade == RelevanceGrade.AMBIGUOUS:
                # Keep ambiguous documents with lower priority
                relevant_nodes.append(graded_doc.node)
                stats["ambiguous"] += 1
            else:
                stats["irrelevant"] += 1

        return relevant_nodes, stats

    def _re_retrieve(self, query: str) -> List[NodeWithScore]:
        """
        Perform re-retrieval with a rewritten query.

        Args:
            query: The rewritten query

        Returns:
            List of newly retrieved nodes
        """
        try:
            nodes = self.retriever.retrieve(query)

            # Apply postprocessors (reranking)
            for postprocessor in self.node_postprocessors:
                nodes = postprocessor.postprocess_nodes(nodes, query_str=query)

            logger.info(f"[CRAG] Re-retrieval obtuvo {len(nodes)} documentos")
            return nodes

        except Exception as e:
            logger.warning(f"[CRAG] Re-retrieval failed: {e}")
            return []
