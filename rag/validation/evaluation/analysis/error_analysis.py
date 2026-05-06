"""
Error analysis: failed retrieval tables, routing errors, abstention failures.
"""

from typing import Dict, List, Optional


def build_error_report(
    retrieval_results: Dict[str, List[dict]],
    routing_results: Optional[List[dict]] = None,
    abstention_results: Optional[dict] = None,
    k: int = 10,
) -> dict:
    """Build the comprehensive error analysis report.

    Args:
        retrieval_results: {config_id: per_query_results} from retrieval metrics
        routing_results: per_query routing results
        abstention_results: from abstention metrics

    Returns:
        Dict with failed_retrieval, routing_errors, abstention_failures
    """
    from validation.evaluation.metrics.retrieval import get_failures
    from validation.evaluation.metrics.routing import get_routing_errors
    from validation.evaluation.metrics.abstention import get_abstention_failures

    report = {}

    # 1. Failed retrieval per config
    report["failed_retrieval"] = {}
    for config_id, per_query in retrieval_results.items():
        failures = get_failures(per_query, k=k)
        if failures:
            report["failed_retrieval"][config_id] = failures

    # 2. Routing errors
    if routing_results:
        report["routing_errors"] = get_routing_errors(routing_results)

    # 3. Abstention failures
    if abstention_results and "per_query" in abstention_results:
        report["abstention_failures"] = get_abstention_failures(
            abstention_results["per_query"]
        )

    return report
