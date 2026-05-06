"""
Generation quality metrics: faithfulness and answer relevancy.

Uses an LLM judge (see judges/) to score responses.
Placeholder — prompts and scoring finalized in Phase 4.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from validation.evaluation.judges.base import BaseJudge


def compute_faithfulness(
    events: List[dict],
    judge: "BaseJudge",
    max_queries: Optional[int] = None,
) -> dict:
    """Score faithfulness for events that have both context and response.

    Faithfulness = is the response supported by the retrieved context?
    """
    scoreable = [
        ev for ev in events
        if ev.get("response") and ev.get("retrieved_docs")
    ]
    if max_queries:
        scoreable = scoreable[:max_queries]

    scores = []
    per_query = []

    for ev in scoreable:
        context = _build_context_text(ev)
        response = ev.get("response", "")
        score = judge.score_faithfulness(context, response)
        scores.append(score)
        per_query.append({
            "id": ev.get("query_id", "?"),
            "score": score,
        })

    mean = sum(scores) / len(scores) if scores else 0
    return {
        "mean_faithfulness": round(mean, 4),
        "n_scored": len(scores),
        "per_query": per_query,
    }


def compute_relevancy(
    events: List[dict],
    queries: List[dict],
    judge: "BaseJudge",
    max_queries: Optional[int] = None,
) -> dict:
    """Score answer relevancy for events.

    Answer relevancy = does the response actually answer the question?
    """
    query_map = {q["id"]: q["query"] for q in queries}

    scoreable = [
        ev for ev in events
        if ev.get("response") and ev.get("query_id") in query_map
    ]
    if max_queries:
        scoreable = scoreable[:max_queries]

    scores = []
    per_query = []

    for ev in scoreable:
        question = query_map[ev["query_id"]]
        response = ev.get("response", "")
        score = judge.score_relevancy(question, response)
        scores.append(score)
        per_query.append({
            "id": ev.get("query_id", "?"),
            "score": score,
        })

    mean = sum(scores) / len(scores) if scores else 0
    return {
        "mean_relevancy": round(mean, 4),
        "n_scored": len(scores),
        "per_query": per_query,
    }


def _build_context_text(event: dict) -> str:
    """Build a text representation of the retrieved context."""
    docs = event.get("retrieved_docs", [])
    if isinstance(docs, list):
        return "\n".join(str(d) for d in docs)
    return str(docs)
