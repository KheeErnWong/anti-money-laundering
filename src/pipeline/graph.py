"""Assemble the AML inference pipeline as a LangGraph StateGraph."""

from langgraph.graph import StateGraph, START, END

from src.pipeline.state import AMLState
from src.pipeline.agents.scorer import scorer_node
from src.pipeline.agents.risk_assessor import risk_assessor_node
from src.pipeline.agents.sar_generator import sar_generator_node


def route_after_risk(state: AMLState) -> str:
    """Skip SAR generation if risk is LOW with no flags.

    Saves a Bedrock API call for clean batches.

    Args:
        state: Pipeline state after risk assessment.

    Returns:
        Next node name or END.
    """
    if state["risk_level"] == "LOW" and state["flagged_count"] == 0:
        return END
    return "sar_generator"


def build_graph():
    """Build and compile the AML inference pipeline.

    Flow: scorer -> risk_assessor -> [conditional] -> sar_generator -> END

    Returns:
        Compiled LangGraph StateGraph.
    """
    builder = StateGraph(AMLState)

    builder.add_node("scorer", scorer_node)
    builder.add_node("risk_assessor", risk_assessor_node)
    builder.add_node("sar_generator", sar_generator_node)

    builder.add_edge(START, "scorer")
    builder.add_edge("scorer", "risk_assessor")
    builder.add_conditional_edges(
        "risk_assessor",
        route_after_risk,
        {"sar_generator": "sar_generator", END: END},
    )
    builder.add_edge("sar_generator", END)

    return builder.compile()


def run_pipeline(transactions: list[dict]) -> AMLState:
    """Run the full AML pipeline on a list of transaction dicts.

    Args:
        transactions: List of raw transaction dicts with keys matching
            the IBM AML dataset columns (Timestamp, From Bank, Account,
            To Bank, Account.1, Amount Received, Receiving Currency,
            Amount Paid, Payment Currency, Payment Format).

    Returns:
        Final pipeline state with scores, risk assessment, and SAR draft.
    """
    graph = build_graph()

    initial_state = AMLState(
        transactions=transactions,
        scores=[],
        flagged_transactions=[],
        risk_level="",
        detected_patterns=[],
        risk_summary="",
        sar_draft="",
        model_name="",
        total_transactions=0,
        flagged_count=0,
    )

    return graph.invoke(initial_state)
