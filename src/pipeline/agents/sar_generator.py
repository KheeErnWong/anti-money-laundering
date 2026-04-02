"""SAR narrative generation agent using AWS Bedrock via boto3."""

import json

import boto3

from src.config import AWS_REGION, BEDROCK_MODEL_ID
from src.pipeline.state import AMLState


SAR_SYSTEM_PROMPT = """You are a compliance analyst assistant at a financial institution.
Generate a draft Suspicious Activity Report (SAR) based on the transaction
analysis provided.

Structure the report with these sections:
1. Subject Information — account identifiers involved
2. Suspicious Activity Summary — what happened, when, how much
3. Detected Patterns — structuring, layering, fan-out, etc.
4. Risk Assessment — overall risk level and confidence
5. Recommended Action — escalate, file SAR, dismiss, monitor

Rules:
- Write in formal, factual language suitable for regulatory filing
- Do NOT fabricate details beyond what is provided
- Do NOT include speculation or assumptions
- Keep the report between 200-400 words
- Reference specific transaction amounts and account IDs from the data"""


def _invoke_bedrock(system_prompt: str, user_prompt: str) -> str:
    """Call Bedrock Claude via boto3 and return the response text.

    Args:
        system_prompt: System instructions for the model.
        user_prompt: User message content.

    Returns:
        Model response text.
    """
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }),
        contentType="application/json",
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def sar_generator_node(state: AMLState) -> dict:
    """Generate a draft SAR narrative from pipeline state.

    Sends the risk assessment context and flagged transactions to
    Bedrock Claude, which produces a formal SAR document.

    Skipped when risk_level is LOW and no transactions are flagged
    (handled by conditional routing in the graph).

    Args:
        state: Pipeline state with risk assessment and flagged transactions.

    Returns:
        Dict with sar_draft string.
    """
    if state["risk_level"] == "LOW" and state["flagged_count"] == 0:
        return {
            "sar_draft": "No suspicious activity detected. SAR filing not recommended.",
        }

    # Limit to 10 flagged transactions to keep prompt size reasonable
    flagged_summary = json.dumps(
        state["flagged_transactions"][:10], indent=2, default=str
    )

    prompt = f"""Generate a draft SAR based on the following analysis:

Risk Level: {state['risk_level']}
Total Transactions Analyzed: {state['total_transactions']}
Flagged Transactions: {state['flagged_count']}
Detected Patterns: {', '.join(state['detected_patterns']) or 'None identified'}
Risk Summary: {state['risk_summary']}
Model Used: {state['model_name']}

Flagged Transaction Details:
{flagged_summary}

Generate the SAR draft now."""

    sar_text = _invoke_bedrock(SAR_SYSTEM_PROMPT, prompt)
    return {"sar_draft": sar_text}
