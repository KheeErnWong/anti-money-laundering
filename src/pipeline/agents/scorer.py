"""ML scoring agent — runs trained model on input transactions."""

import json

import pandas as pd

from src.config import MODELS_DIR
from src.data.preprocess import engineer_features
from src.models.train import load_model
from src.pipeline.state import AMLState


def _load_artifacts():
    """Load selected model and feature names from disk."""
    with open(MODELS_DIR / "selection_decision.json") as f:
        decision = json.load(f)

    model_name = decision["selected_model"]
    model = load_model(model_name)

    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    return model, model_name, feature_names


def scorer_node(state: AMLState) -> dict:
    """Score transactions using the selected ML model.

    Loads the trained model, engineers features from raw transactions,
    runs inference, and separates flagged transactions.

    Args:
        state: Pipeline state with transactions list.

    Returns:
        Dict with scores, flagged_transactions, model_name,
        total_transactions, flagged_count.
    """
    model, model_name, feature_names = _load_artifacts()

    df = pd.DataFrame(state["transactions"])

    # Add placeholder graph features (not available at inference time
    # without the full transaction graph, so default to 0)
    for col in ["fan_out", "fan_in", "sender_txn_count", "receiver_txn_count"]:
        if col not in df.columns:
            df[col] = 0

    df_features = engineer_features(df)

    # Align columns with training features
    for col in feature_names:
        if col not in df_features.columns:
            df_features[col] = 0
    df_features = df_features[feature_names]

    # Predict
    predictions = model.predict(df_features)
    probabilities = model.predict_proba(df_features)[:, 1]

    scores = []
    flagged = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        record = {
            "index": i,
            "score": round(float(prob), 4),
            "prediction": int(pred),
        }
        scores.append(record)

        if pred == 1:
            txn = state["transactions"][i].copy()
            txn["risk_score"] = round(float(prob), 4)
            flagged.append(txn)

    return {
        "scores": scores,
        "flagged_transactions": flagged,
        "model_name": model_name,
        "total_transactions": len(scores),
        "flagged_count": len(flagged),
    }
