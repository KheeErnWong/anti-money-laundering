"""Risk assessment agent — detects laundering patterns and assigns risk level."""

from collections import defaultdict

from src.pipeline.state import AMLState


def risk_assessor_node(state: AMLState) -> dict:
    """Analyze flagged transactions for known AML patterns.

    Checks for structuring, fan-out, fan-in, velocity spikes, and
    cross-currency transfers. Assigns a risk level and writes a
    plain-English summary for the SAR generator.

    Args:
        state: Pipeline state with flagged_transactions and scores.

    Returns:
        Dict with risk_level, detected_patterns, risk_summary.
    """
    flagged = state["flagged_transactions"]
    total = state["total_transactions"]
    flagged_count = state["flagged_count"]

    if flagged_count == 0:
        return {
            "risk_level": "LOW",
            "detected_patterns": [],
            "risk_summary": (
                f"Analyzed {total} transactions. "
                "No suspicious activity detected."
            ),
        }

    patterns = []
    scores = [f["risk_score"] for f in flagged]
    avg_score = sum(scores) / len(scores)

    # Structuring: multiple amounts just under $10K reporting threshold
    amounts = [f["Amount Paid"] for f in flagged]
    near_threshold = [a for a in amounts if 8000 <= a <= 9999]
    if len(near_threshold) >= 2:
        patterns.append(
            f"structuring ({len(near_threshold)} transactions "
            f"in $8,000-$9,999 range)"
        )

    # Fan-out: per sender, how many unique receivers among flagged
    sender_to_receivers = defaultdict(set)
    for txn in flagged:
        sender_to_receivers[txn["Account"]].add(txn["Account.1"])

    for sender, recvs in sender_to_receivers.items():
        if len(recvs) >= 3:
            patterns.append(
                f"fan-out (account {sender} to {len(recvs)} receivers)"
            )

    # Fan-in: per receiver, how many unique senders among flagged
    receiver_to_senders = defaultdict(set)
    for txn in flagged:
        receiver_to_senders[txn["Account.1"]].add(txn["Account"])

    for receiver, sndrs in receiver_to_senders.items():
        if len(sndrs) >= 3:
            patterns.append(
                f"fan-in ({len(sndrs)} senders to account {receiver})"
            )

    # Velocity spike: many flagged transactions in one batch
    if flagged_count >= 3:
        patterns.append(
            f"velocity spike ({flagged_count} flagged in batch)"
        )

    # Cross-currency transfers
    currency_mismatches = [
        f for f in flagged
        if f["Payment Currency"] != f["Receiving Currency"]
    ]
    if len(currency_mismatches) >= 2:
        patterns.append(
            f"cross-currency ({len(currency_mismatches)} transfers)"
        )

    # Determine risk level
    if avg_score >= 0.8 or flagged_count >= 5 or len(patterns) >= 2:
        risk_level = "HIGH"
    elif avg_score >= 0.5 or flagged_count >= 2 or len(patterns) >= 1:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    pattern_str = ", ".join(patterns) if patterns else "no specific pattern"
    summary = (
        f"Analyzed {total} transactions. "
        f"{flagged_count} flagged as suspicious "
        f"(avg risk score: {avg_score:.2f}). "
        f"Risk level: {risk_level}. "
        f"Detected patterns: {pattern_str}."
    )

    return {
        "risk_level": risk_level,
        "detected_patterns": patterns,
        "risk_summary": summary,
    }
