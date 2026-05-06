"""
Abstention metrics: abstention accuracy, false abstention rate,
and adversarial abstention accuracy.

Detects abstention by pattern-matching the response against known
abstention templates from the prompt system.
"""

import re
from typing import List, Optional

# Abstention patterns from the prompt templates.
# The system uses academic_es_v2 which produces:
#   "No cuento con suficiente información como para responder"
# And strict_factual which produces:
#   "Información no encontrada en el contexto proporcionado."
# We match case-insensitively and with some flexibility.

ABSTENTION_PATTERNS = [
    r"no cuento con suficiente información",
    r"información no encontrada",
    r"no dispongo de.*información",
    r"no tengo.*información.*suficiente",
    r"no es posible responder",
    r"no se encuentra.*en el contexto",
    r"fuera del alcance",
    r"no puedo responder",
    r"no hay información.*disponible",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ABSTENTION_PATTERNS]


def is_abstention(response: str) -> bool:
    """Check if a response constitutes an abstention.

    Returns True if the response matches any known abstention pattern.
    """
    if not response:
        return True  # empty response = implicit abstention
    return any(p.search(response) for p in _COMPILED_PATTERNS)


def compute_abstention_metrics(
    queries: List[dict],
    responses: dict,
    partition_filter: Optional[str] = None,
) -> dict:
    """Compute abstention accuracy metrics.

    Args:
        queries: dataset queries with 'id', 'expected_abstention', 'partition'
        responses: {query_id: response_text}
        partition_filter: if set, restrict to this partition

    Returns:
        Dict with abstention_accuracy, false_abstention_rate,
        adversarial_abstention_accuracy, and per-query details.
    """
    if partition_filter:
        queries = [q for q in queries if q.get("partition") == partition_filter]

    per_query = []
    for q in queries:
        qid = q["id"]
        expected = q.get("expected_abstention", False)
        response = responses.get(qid, "")
        actual = is_abstention(response)

        per_query.append({
            "id": qid,
            "partition": q.get("partition", ""),
            "type": q.get("type", ""),
            "expected_abstention": expected,
            "actual_abstention": actual,
            "correct": expected == actual,
            "response_preview": response[:100] if response else "(empty)",
        })

    n = len(per_query)
    if n == 0:
        return {"n_queries": 0}

    # ── Abstention Accuracy (expected=true queries) ──
    expected_true = [q for q in per_query if q["expected_abstention"]]
    n_expected = len(expected_true)
    true_positives = sum(1 for q in expected_true if q["actual_abstention"])
    abstention_accuracy = true_positives / n_expected if n_expected > 0 else 0.0

    # ── False Abstention Rate (expected=false but system abstained) ──
    expected_false = [q for q in per_query if not q["expected_abstention"]]
    n_should_answer = len(expected_false)
    false_abstentions = sum(1 for q in expected_false if q["actual_abstention"])
    false_abstention_rate = false_abstentions / n_should_answer if n_should_answer > 0 else 0.0

    # ── Adversarial Abstention Accuracy ──
    adv_expected = [q for q in per_query
                    if q["expected_abstention"] and q["partition"] == "adversarial"]
    n_adv = len(adv_expected)
    adv_correct = sum(1 for q in adv_expected if q["actual_abstention"])
    adv_accuracy = adv_correct / n_adv if n_adv > 0 else 0.0

    return {
        "n_queries": n,
        "abstention_accuracy": round(abstention_accuracy, 4),
        "false_abstention_rate": round(false_abstention_rate, 4),
        "adversarial_abstention_accuracy": round(adv_accuracy, 4),
        "detail": {
            "expected_abstention_count": n_expected,
            "true_positives": true_positives,
            "false_negatives": n_expected - true_positives,
            "false_abstentions": false_abstentions,
            "adversarial_expected": n_adv,
            "adversarial_correct": adv_correct,
        },
        "per_query": per_query,
    }


def get_abstention_failures(per_query: List[dict]) -> dict:
    """Extract abstention failure tables.

    Returns:
        {
            "missed_abstentions": queries that should have abstained but didn't,
            "false_abstentions": queries that abstained but shouldn't have,
        }
    """
    missed = [q for q in per_query
              if q["expected_abstention"] and not q["actual_abstention"]]
    false = [q for q in per_query
             if not q["expected_abstention"] and q["actual_abstention"]]
    return {
        "missed_abstentions": missed,
        "false_abstentions": false,
    }
