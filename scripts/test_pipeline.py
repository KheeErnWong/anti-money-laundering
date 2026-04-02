"""Test the full AML pipeline with a sample of raw transactions."""

import pandas as pd

from src.config import RAW_DIR, DATASET_NAME
from src.pipeline.graph import run_pipeline


def main():
    raw = pd.read_csv(RAW_DIR / DATASET_NAME)

    # Mix more laundering + legitimate transactions to increase flag chance
    laundering = raw[raw["Is Laundering"] == 1].head(50)
    legit = raw[raw["Is Laundering"] == 0].head(50)
    sample = pd.concat([laundering, legit]).to_dict(orient="records")

    print(f"Running pipeline on {len(sample)} transactions...")
    result = run_pipeline(sample)

    # Debug: check score distribution
    scores = [s["score"] for s in result["scores"]]
    print(f"\nScore stats: min={min(scores):.4f}, max={max(scores):.4f}, mean={sum(scores)/len(scores):.4f}")
    top_5 = sorted(scores, reverse=True)[:5]
    print(f"Top 5 scores: {[f'{s:.4f}' for s in top_5]}")

    print(f"\nRisk Level: {result['risk_level']}")
    print(f"Flagged: {result['flagged_count']} / {result['total_transactions']}")
    print(f"Model: {result['model_name']}")
    print(f"Patterns: {result['detected_patterns']}")
    print(f"\nSAR Draft:\n{result['sar_draft']}")


if __name__ == "__main__":
    main()
