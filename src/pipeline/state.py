"""Shared state for the AML LangGraph inference pipeline."""

from typing import TypedDict


class AMLState(TypedDict):
    """State passed between all nodes in the inference pipeline.

    The scorer writes scores and flagged transactions.
    The risk assessor writes risk level, patterns, and summary.
    The SAR generator writes the draft report.
    """

    # Raw input
    transactions: list[dict]

    # Scorer output
    scores: list[dict]
    flagged_transactions: list[dict]

    # Risk assessor output
    risk_level: str
    detected_patterns: list[str]
    risk_summary: str

    # SAR generator output
    sar_draft: str

    # Metadata
    model_name: str
    total_transactions: int
    flagged_count: int
