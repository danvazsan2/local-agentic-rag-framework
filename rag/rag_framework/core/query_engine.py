"""
Query engine module with support for custom prompts and LLMs.

Provides DocumentQueryEngine for document-only RAG queries and
QueryEngineManager for query engine creation and configuration.
"""

import logging
import time
from typing import List, Optional, Any

from rag_framework.core.instrumentation import QueryTrace, phase_timer

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate as LlamaIndexPrompt
from llama_index.core.schema import TextNode

from rag_framework.config.models import RAGConfig
from rag_framework.providers.llm import LLMFactory
from rag_framework.prompts.templates import PromptTemplates
from rag_framework.core.retrieval import RetrieverManager, HybridRetriever

logger = logging.getLogger(__name__)


class DocumentQueryEngine:
    """Query engine for document retrieval with optional reranking.

    This engine handles queries against the document vector store,
    supporting hybrid retrieval (dense + sparse), reranking, and
    optional Corrective RAG (document grading + query rewriting).
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        response_synthesizer: Any,
        node_postprocessors: List[Any] = None,
        debug: bool = False,
        corrective_rag_engine: Optional[Any] = None,
        config: Optional[Any] = None,
        query_preprocessor: Optional[Any] = None,
    ):
        self.retriever = retriever
        self.response_synthesizer = response_synthesizer
        self.node_postprocessors = node_postprocessors or []
        self._config = config
        self._debug_fallback = debug
        self.corrective_rag_engine = corrective_rag_engine
        self.query_preprocessor = query_preprocessor

    @property
    def debug(self) -> bool:
        """Read debug flag dynamically from config so runtime toggles take effect."""
        if self._config is not None:
            return self._config.debug
        return self._debug_fallback

    def _debug_print_nodes(self, header: str, nodes) -> None:
        """Print node details when debug mode is active."""
        print(f"\n  [{header}]")
        for i, node in enumerate(nodes):
            score = node.score if hasattr(node, "score") else 0
            filename = (
                node.node.metadata.get("file_name", "Unknown")
                if hasattr(node.node, "metadata")
                else "Unknown"
            )
            text_preview = (
                node.node.get_content()[:80] + "..."
                if hasattr(node.node, "get_content")
                else ""
            )
            print(f"    Chunk {i+1}/{len(nodes)}: score={score:.4f}, file={filename}")
            if text_preview:
                print(f"      {text_preview}")

    def query(self, query_str: str, trace: Optional[QueryTrace] = None) -> Any:
        """Execute a RAG query."""
        t_total = time.time()

        # 0. Query preprocessing (metadata pre-filter)
        prefilter_result = None
        metadata_filters = None
        if self.query_preprocessor is not None:
            with phase_timer(trace, "preprocessor_ms"):
                prefilter_result = self.query_preprocessor.analyse(query_str)
            metadata_filters = prefilter_result.metadata_filters
            if self.debug and prefilter_result.matched_value:
                print(
                    f"\n[Debug] Pre-filtro metadata: {prefilter_result.matched_field}"
                    f"='{prefilter_result.matched_value}' "
                    f"(score={prefilter_result.similarity:.2f})"
                )

        # 1. Retrieve
        if self.debug:
            print("\n[Debug] Paso 1/4: Recuperando documentos (retrieval)...")
        t0 = time.time()
        if isinstance(self.retriever, HybridRetriever) and metadata_filters is not None:
            nodes = self.retriever.retrieve(query_str, filters=metadata_filters, trace=trace)
        else:
            nodes = self.retriever.retrieve(query_str, trace=trace)
        t_retrieval = time.time() - t0

        if trace is not None:
            trace.phases["retrieval_ms"] = round(t_retrieval * 1000, 2)

        if self.debug:
            print(
                f"  Retrieval completado en {t_retrieval:.2f}s — {len(nodes)} chunks recuperados"
            )
            self._debug_print_nodes("Chunks antes de reranking", nodes[:5])

        # 2. Postprocess (rerank)
        if self.debug and self.node_postprocessors:
            print(
                f"\n[Debug] Paso 2/4: Reranking ({len(self.node_postprocessors)} postprocessor(s))..."
            )
        t0 = time.time()
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_str=query_str)
        t_rerank = time.time() - t0

        if trace is not None:
            trace.phases["reranker_ms"] = round(t_rerank * 1000, 2)

        if self.debug:
            print(
                f"  Reranking completado en {t_rerank:.2f}s — {len(nodes)} chunks tras filtrado"
            )
            self._debug_print_nodes("Chunks después de reranking", nodes)

        # 2.5 Apply metadata boost (if applicable)
        if (
            prefilter_result is not None
            and prefilter_result.boost_field_value
            and self.query_preprocessor is not None
        ):
            filter_cfg = self.query_preprocessor.config
            nodes = self.query_preprocessor.apply_boost(
                nodes,
                boost_field=filter_cfg.boost_field,
                boost_value=prefilter_result.boost_field_value,
                factor=filter_cfg.boost_factor,
            )
            if self.debug:
                print(
                    f"\n[Debug] Boost aplicado: {filter_cfg.boost_field}"
                    f"='{prefilter_result.boost_field_value}' (x{filter_cfg.boost_factor})"
                )

        # Record retrieved docs for the trace before CRAG filtering
        if trace is not None:
            trace.retrieved_docs = [
                n.node.metadata.get("file_name", str(n.node.node_id)) for n in nodes
            ]

        # 3. Corrective RAG (grade + filter + optional rewrite)
        if self.corrective_rag_engine is not None:
            if self.debug:
                print(
                    f"\n[Debug] Paso 3/4: Corrective RAG (evaluando {len(nodes)} chunks)..."
                )
            t0 = time.time()
            crag_result = self.corrective_rag_engine.process(query_str, nodes, trace=trace)
            nodes = crag_result.filtered_nodes
            t_crag = time.time() - t0

            if trace is not None:
                trace.phases["crag_ms"] = round(t_crag * 1000, 2)

            if self.debug:
                print(
                    f"  CRAG completado en {t_crag:.2f}s — "
                    f"{crag_result.relevant_count} relevantes, "
                    f"{crag_result.irrelevant_count} irrelevantes, "
                    f"{crag_result.ambiguous_count} ambiguos"
                )
                if crag_result.query_rewritten:
                    print(f"  Query reescrita: '{crag_result.rewritten_query}'")
                self._debug_print_nodes("Chunks tras CRAG", nodes)
        elif self.debug:
            print("\n[Debug] Paso 3/4: Corrective RAG — deshabilitado, saltando")

        # 4. Synthesize response
        if self.debug:
            print(
                f"\n[Debug] Paso 4/4: Sintetizando respuesta con LLM ({len(nodes)} chunks de contexto)..."
            )
        t0 = time.time()
        response = self.response_synthesizer.synthesize(
            query_str,
            nodes=nodes,
        )
        t_synth = time.time() - t0

        if trace is not None:
            trace.phases["synthesis_ms"] = round(t_synth * 1000, 2)
            trace.phases["total_ms"] = round((time.time() - t_total) * 1000, 2)
            trace.response = str(response)

        if self.debug:
            t_total_elapsed = time.time() - t_total
            print(f"  Síntesis completada en {t_synth:.2f}s")
            print(f"\n[Debug] Pipeline RAG completado en {t_total_elapsed:.2f}s total")
            print(
                f"  Retrieval: {t_retrieval:.2f}s | Reranking: {t_rerank:.2f}s | Síntesis: {t_synth:.2f}s"
            )

        return response


# Backwards compatibility alias (deprecated - will be removed in v2.0)
HybridQueryEngine = DocumentQueryEngine


class QueryEngineManager:
    """Manages query engine creation and configuration."""

    def __init__(self, config: RAGConfig):
        """Initialize query engine manager.

        Args:
            config: RAG configuration
        """
        self.config = config
        self._llm = None

    @property
    def llm(self):
        """Lazily load the LLM."""
        if self._llm is None:
            self._llm = LLMFactory.get_llm(self.config.llm)
        return self._llm

    def get_prompt_template(self) -> str:
        """Get the configured prompt template."""
        if self.config.prompt_template == "custom" and self.config.custom_prompt:
            return self.config.custom_prompt

        try:
            return PromptTemplates.get_template_string(self.config.prompt_template)
        except KeyError:
            logger.warning(
                f"Template '{self.config.prompt_template}' not found, using default"
            )
            return PromptTemplates.get_template_string("default")

    def create_query_engine(
        self,
        index: VectorStoreIndex,
        nodes: Optional[List[TextNode]] = None,
        streaming: bool = False,
    ) -> Any:
        """Create a query engine with configured LLM and prompt.

        Args:
            index: Vector store index
            nodes: Original nodes (for hybrid search)
            streaming: Enable streaming responses

        Returns:
            Query engine instance
        """
        logger.info("Configuring query engine")

        # Create retriever
        retriever_manager = RetrieverManager(self.config)
        retriever, postprocessors = retriever_manager.create_retriever(
            index=index,
            nodes=nodes,
        )

        # Get prompt template
        prompt_template = self.get_prompt_template()
        qa_prompt = LlamaIndexPrompt(prompt_template)

        logger.info(f"Using prompt template: {self.config.prompt_template}")

        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            streaming=streaming,
            response_mode="compact",
            text_qa_template=qa_prompt,
        )

        # Create Corrective RAG engine if enabled
        corrective_rag_engine = None
        if self.config.corrective_rag.enabled:
            from rag_framework.core.corrective_rag import CorrectiveRAGEngine

            corrective_rag_engine = CorrectiveRAGEngine(
                config=self.config,
                llm=self.llm,
                retriever=retriever,
                node_postprocessors=postprocessors,
            )
            logger.info("Corrective RAG enabled")

        # Create query preprocessor for metadata-based filtering
        query_preprocessor = None
        meta_cfg = self.config.metadata
        if meta_cfg.enabled and meta_cfg.filtering.enabled and nodes:
            from rag_framework.core.query_preprocessor import QueryPreprocessor
            from rag_framework.core.metadata_extractor import MetadataExtractor

            extractor = MetadataExtractor(meta_cfg)
            known_values = extractor.get_all_values(
                meta_cfg.filtering.match_field, nodes
            )
            if known_values:
                query_preprocessor = QueryPreprocessor(meta_cfg.filtering, known_values)
                logger.info(
                    "Query preprocessor enabled: %d known values for '%s'",
                    len(known_values),
                    meta_cfg.filtering.match_field,
                )

        # Create query engine
        if isinstance(retriever, HybridRetriever):
            query_engine = DocumentQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=postprocessors,
                corrective_rag_engine=corrective_rag_engine,
                config=self.config,
                query_preprocessor=query_preprocessor,
            )
        else:
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=postprocessors,
            )

        logger.info("Query engine ready")

        return query_engine
