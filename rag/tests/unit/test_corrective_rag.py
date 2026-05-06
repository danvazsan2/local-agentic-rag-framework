"""Unit tests for Corrective RAG components.

Tests RelevanceGrader, QueryRewriter, CorrectiveRAGEngine, and CorrectiveRAGConfig
independently using mocked LLM responses.
"""

from unittest.mock import MagicMock
import pytest

from rag_framework.config.corrective_rag_config import CorrectiveRAGConfig
from rag_framework.core.corrective_rag import (
    RelevanceGrader,
    RelevanceGrade,
    GradedDocument,
    QueryRewriter,
    CorrectiveRAGEngine,
    CorrectiveRAGResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(node_id: str, text: str):
    mock_node = MagicMock()
    mock_node.node_id = node_id
    mock_node.get_content.return_value = text
    mock_node.metadata = {"file_name": "doc.pdf"}
    mock_nws = MagicMock()
    mock_nws.node = mock_node
    mock_nws.score = 0.9
    return mock_nws


def _make_llm(responses: list[str]):
    llm = MagicMock()
    llm.complete.side_effect = [MagicMock(__str__=lambda s, r=r: r) for r in responses]
    return llm


# ---------------------------------------------------------------------------
# CorrectiveRAGConfig
# ---------------------------------------------------------------------------


class TestCorrectiveRAGConfig:
    def test_defaults(self):
        cfg = CorrectiveRAGConfig()
        assert cfg.enabled is False
        assert cfg.relevance_threshold == 0.5
        assert cfg.max_retries == 1

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="relevance_threshold"):
            CorrectiveRAGConfig(relevance_threshold=1.5)
        with pytest.raises(ValueError, match="relevance_threshold"):
            CorrectiveRAGConfig(relevance_threshold=-0.1)

    def test_negative_retries_raises(self):
        with pytest.raises(ValueError, match="max_retries"):
            CorrectiveRAGConfig(max_retries=-1)


# ---------------------------------------------------------------------------
# RelevanceGrader
# ---------------------------------------------------------------------------


class TestRelevanceGrader:
    def test_grade_relevant(self):
        grader = RelevanceGrader(llm=_make_llm(["RELEVANT"]))
        result = grader.grade_document("¿Cuál es el horario?", _make_node("n1", "horario"))
        assert result.grade == RelevanceGrade.RELEVANT
        assert isinstance(result, GradedDocument)

    def test_grade_irrelevant(self):
        grader = RelevanceGrader(llm=_make_llm(["IRRELEVANT"]))
        result = grader.grade_document("¿Cuál es el horario?", _make_node("n2", "biología"))
        assert result.grade == RelevanceGrade.IRRELEVANT

    def test_grade_ambiguous(self):
        grader = RelevanceGrader(llm=_make_llm(["AMBIGUOUS"]))
        result = grader.grade_document("¿Cuál es el horario?", _make_node("n3", "contenido tangencial"))
        assert result.grade == RelevanceGrade.AMBIGUOUS

    def test_grade_documents_batch(self):
        grader = RelevanceGrader(llm=_make_llm(["RELEVANT", "IRRELEVANT", "RELEVANT"]))
        nodes = [_make_node(f"n{i}", f"text {i}") for i in range(3)]
        results = grader.grade_documents("some question", nodes)
        assert len(results) == 3
        assert results[0].grade == RelevanceGrade.RELEVANT
        assert results[1].grade == RelevanceGrade.IRRELEVANT

    def test_llm_failure_returns_ambiguous(self):
        llm = MagicMock()
        llm.complete.side_effect = Exception("LLM connection lost")
        grader = RelevanceGrader(llm=llm)
        result = grader.grade_document("question", _make_node("n6", "some text"))
        assert result.grade == RelevanceGrade.AMBIGUOUS


# ---------------------------------------------------------------------------
# QueryRewriter
# ---------------------------------------------------------------------------


class TestQueryRewriter:
    def test_rewrite_returns_expanded_query(self):
        rewriter = QueryRewriter(llm=_make_llm(["horario apertura servicio atención"]))
        result = rewriter.rewrite("¿Cuál es el horario?")
        assert result == "horario apertura servicio atención"

    def test_rewrite_strips_surrounding_quotes(self):
        rewriter = QueryRewriter(llm=_make_llm(['"reformulated query here"']))
        assert rewriter.rewrite("original query") == "reformulated query here"

    def test_rewrite_too_short_falls_back_to_original(self):
        rewriter = QueryRewriter(llm=_make_llm(["hi"]))
        assert rewriter.rewrite("original question here") == "original question here"

    def test_rewrite_failure_returns_original(self):
        llm = MagicMock()
        llm.complete.side_effect = Exception("LLM error")
        assert QueryRewriter(llm=llm).rewrite("original query") == "original query"


# ---------------------------------------------------------------------------
# CorrectiveRAGEngine
# ---------------------------------------------------------------------------


def _engine_config(enabled=True, threshold=0.5, max_retries=1):
    from rag_framework.config.models import RAGConfig
    return RAGConfig(
        corrective_rag=CorrectiveRAGConfig(
            enabled=enabled,
            relevance_threshold=threshold,
            max_retries=max_retries,
        )
    )


class TestCorrectiveRAGEngine:
    def test_all_relevant_no_rewrite_triggered(self):
        config = _engine_config(threshold=0.5)
        nodes = [_make_node(f"n{i}", f"text {i}") for i in range(3)]
        engine = CorrectiveRAGEngine(config=config, llm=_make_llm(["RELEVANT"] * 3))
        result = engine.process("question", nodes)
        assert len(result.filtered_nodes) == 3
        assert result.query_rewritten is False

    def test_irrelevant_docs_filtered(self):
        config = _engine_config(threshold=0.0, max_retries=0)
        nodes = [_make_node(f"n{i}", f"text {i}") for i in range(3)]
        engine = CorrectiveRAGEngine(config=config, llm=_make_llm(["RELEVANT", "IRRELEVANT", "RELEVANT"]))
        result = engine.process("question", nodes)
        assert len(result.filtered_nodes) == 2
        assert result.irrelevant_count == 1

    def test_ambiguous_docs_kept(self):
        config = _engine_config(threshold=0.0, max_retries=0)
        nodes = [_make_node(f"n{i}", f"text {i}") for i in range(2)]
        engine = CorrectiveRAGEngine(config=config, llm=_make_llm(["AMBIGUOUS", "IRRELEVANT"]))
        result = engine.process("question", nodes)
        assert result.ambiguous_count == 1
        assert len(result.filtered_nodes) == 1  # ambiguous kept, irrelevant removed

    def test_rewrite_triggered_when_below_threshold(self):
        config = _engine_config(threshold=0.8, max_retries=1)
        llm = _make_llm([
            "IRRELEVANT", "IRRELEVANT", "RELEVANT",  # first grading
            "better reformulated query terms",          # rewrite
            "RELEVANT", "RELEVANT",                     # grading after re-retrieval
        ])
        nodes = [_make_node(f"n{i}", f"text {i}") for i in range(3)]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_make_node(f"rn{i}", f"new {i}") for i in range(2)]

        engine = CorrectiveRAGEngine(config=config, llm=llm, retriever=mock_retriever)
        result = engine.process("question", nodes)

        assert result.query_rewritten is True
        assert result.rewritten_query == "better reformulated query terms"

    def test_no_rewrite_when_max_retries_zero(self):
        config = _engine_config(threshold=0.8, max_retries=0)
        nodes = [_make_node(f"n{i}", f"text {i}") for i in range(2)]
        engine = CorrectiveRAGEngine(config=config, llm=_make_llm(["IRRELEVANT", "IRRELEVANT"]))
        result = engine.process("question", nodes)
        assert result.query_rewritten is False
        assert result.irrelevant_count == 2

    def test_empty_nodes_input(self):
        config = _engine_config()
        engine = CorrectiveRAGEngine(config=config, llm=_make_llm([]))
        result = engine.process("question", [])
        assert len(result.filtered_nodes) == 0
        assert result.total_retrieved == 0


# ---------------------------------------------------------------------------
# CorrectiveRAGResult
# ---------------------------------------------------------------------------


class TestCorrectiveRAGResult:
    def test_relevance_ratio_computation(self):
        result = CorrectiveRAGResult(
            filtered_nodes=[],
            graded_documents=[],
            total_retrieved=10,
            relevant_count=7,
            irrelevant_count=2,
            ambiguous_count=1,
        )
        assert result.relevance_ratio == 0.7

    def test_relevance_ratio_zero_when_no_docs(self):
        result = CorrectiveRAGResult(
            filtered_nodes=[], graded_documents=[],
            total_retrieved=0, relevant_count=0, irrelevant_count=0, ambiguous_count=0,
        )
        assert result.relevance_ratio == 0.0
