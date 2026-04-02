"""Gradio interface for AML transaction analysis."""

import gradio as gr
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pipeline.graph import run_pipeline


def analyze_batch(file):
    """Run pipeline on uploaded CSV.

    Args:
        file: Uploaded CSV file path.

    Returns:
        Tuple of (verdict, flagged_table, score_plot, sar_draft).
    """
    if file is None:
        return "Upload a CSV file to begin.", pd.DataFrame(), None, ""

    df = pd.read_csv(file)
    transactions = df.to_dict(orient="records")

    result = run_pipeline(transactions)

    # Verdict
    risk = result["risk_level"]
    flagged_count = result["flagged_count"]
    total = result["total_transactions"]
    patterns = result["detected_patterns"]

    verdict = f"## {risk} RISK\n\n"
    verdict += f"**{flagged_count}** of **{total}** transactions flagged.\n\n"
    if patterns:
        verdict += "**Detected patterns:**\n"
        for p in patterns:
            verdict += f"- {p}\n"

    # Flagged transactions table
    if result["flagged_transactions"]:
        flagged_df = pd.DataFrame(result["flagged_transactions"])
        display_cols = [
            c for c in [
                "Timestamp", "Account", "Account.1",
                "Amount Paid", "Payment Currency",
                "Payment Format", "risk_score",
            ]
            if c in flagged_df.columns
        ]
        flagged_df = flagged_df[display_cols] if display_cols else flagged_df
    else:
        flagged_df = pd.DataFrame({"Status": ["No suspicious transactions detected."]})

    # Score distribution plot
    fig, ax = plt.subplots(figsize=(8, 4))
    scores = [s["score"] for s in result["scores"]]
    ax.hist(scores, bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(x=0.15, color="red", linestyle="--", label="Threshold (0.15)")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Risk Score Distribution")
    ax.legend()
    plt.tight_layout()

    # SAR draft
    sar = result.get("sar_draft", "")

    return verdict, flagged_df, fig, sar


def analyze_single(
    from_bank, account, to_bank, account_to,
    amount_paid, payment_currency, amount_received,
    receiving_currency, payment_format,
):
    """Analyze a single transaction.

    Args:
        Individual transaction fields.

    Returns:
        Tuple of (verdict, risk_summary, sar_draft).
    """
    txn = {
        "Timestamp": pd.Timestamp.now().isoformat(),
        "From Bank": int(from_bank),
        "Account": account,
        "To Bank": int(to_bank),
        "Account.1": account_to,
        "Amount Paid": float(amount_paid),
        "Payment Currency": payment_currency,
        "Amount Received": float(amount_received),
        "Receiving Currency": receiving_currency,
        "Payment Format": payment_format,
    }

    result = run_pipeline([txn])

    score = result["scores"][0]["score"] if result["scores"] else 0
    risk = result["risk_level"]
    sar = result.get("sar_draft", "")

    verdict = f"## Risk Score: {score:.2%}\n\nLevel: **{risk}**"
    return verdict, result["risk_summary"], sar


with gr.Blocks() as app:
    gr.Markdown("# AML Transaction Analyzer")
    gr.Markdown(
        "Upload bank transactions or enter a single transaction "
        "to detect suspicious activity and generate a draft SAR."
    )

    with gr.Tab("Batch Analysis"):
        file_input = gr.File(label="Upload Transaction CSV", file_types=[".csv"])
        analyze_btn = gr.Button("Analyze", variant="primary")

        verdict_out = gr.Markdown(label="Verdict")
        flagged_table = gr.Dataframe(label="Flagged Transactions", interactive=False)
        score_plot = gr.Plot(label="Risk Score Distribution")
        sar_out = gr.Textbox(
            label="Draft Suspicious Activity Report",
            lines=20,
            interactive=False,
        )

        analyze_btn.click(
            fn=analyze_batch,
            inputs=[file_input],
            outputs=[verdict_out, flagged_table, score_plot, sar_out],
        )

    with gr.Tab("Single Transaction"):
        with gr.Row():
            from_bank = gr.Number(label="From Bank ID", value=1)
            account = gr.Textbox(label="Sender Account", value="ACCT001")
            to_bank = gr.Number(label="To Bank ID", value=2)
            account_to = gr.Textbox(label="Receiver Account", value="ACCT002")
        with gr.Row():
            amount_paid = gr.Number(label="Amount Paid", value=5000)
            payment_currency = gr.Dropdown(
                choices=[
                    "US Dollar", "Euro", "UK Pound", "Yen", "Yuan",
                    "Rupee", "Bitcoin", "Swiss Franc", "Australian Dollar",
                    "Canadian Dollar", "Mexican Peso", "Ruble",
                    "Saudi Riyal", "Shekel", "Brazil Real",
                ],
                value="US Dollar",
                label="Payment Currency",
            )
            amount_received = gr.Number(label="Amount Received", value=5000)
            receiving_currency = gr.Dropdown(
                choices=[
                    "US Dollar", "Euro", "UK Pound", "Yen", "Yuan",
                    "Rupee", "Bitcoin", "Swiss Franc", "Australian Dollar",
                    "Canadian Dollar", "Mexican Peso", "Ruble",
                    "Saudi Riyal", "Shekel", "Brazil Real",
                ],
                value="US Dollar",
                label="Receiving Currency",
            )
        payment_format = gr.Dropdown(
            choices=["Wire", "ACH", "Cheque", "Credit Card", "Cash", "Reinvestment", "Bitcoin"],
            value="Wire",
            label="Payment Format",
        )
        single_btn = gr.Button("Check Transaction", variant="primary")

        single_verdict = gr.Markdown(label="Result")
        single_summary = gr.Textbox(label="Risk Summary", lines=3, interactive=False)
        single_sar = gr.Textbox(label="SAR Draft", lines=15, interactive=False)

        single_btn.click(
            fn=analyze_single,
            inputs=[
                from_bank, account, to_bank, account_to,
                amount_paid, payment_currency, amount_received,
                receiving_currency, payment_format,
            ],
            outputs=[single_verdict, single_summary, single_sar],
        )


def main():
    app.launch(
        server_name="0.0.0.0",
        server_port=8080,
        title="AML Transaction Analyzer",
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
