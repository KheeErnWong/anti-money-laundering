"""Select the best model based on evaluation metrics."""

import json

import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, OUTPUTS_DIR
from src.data.preprocess import split_data
from src.models.train import load_model
from src.models.evaluate import evaluate_model


def select_best_model(comparison_df: pd.DataFrame) -> dict:
    """Pick the best model by minority-class F1 and save the decision.

    Args:
        comparison_df: DataFrame from compare_models(), sorted by f1_minority.

    Returns:
        Decision dict with selected model name, metrics, and justification.
    """
    best = comparison_df.iloc[0]
    decision = {
        "selected_model": best["model_name"],
        "f1_minority": float(best["f1_minority"]),
        "pr_auc": float(best["pr_auc"]),
        "mcc": float(best["mcc"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "justification": (
            f"Selected {best['model_name']} with minority-class F1 "
            f"of {best['f1_minority']:.4f} and PR-AUC of {best['pr_auc']:.4f}. "
            f"Precision: {best['precision']:.4f}, Recall: {best['recall']:.4f}, "
            f"MCC: {best['mcc']:.4f}."
        ),
    }

    path = MODELS_DIR / "selection_decision.json"
    with open(path, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"Selection decision saved to {path}")

    return decision


def save_feature_names(feature_names: list[str]) -> None:
    """Save training feature names for inference-time column alignment.

    Args:
        feature_names: List of column names from X_train.
    """
    path = MODELS_DIR / "feature_names.json"
    with open(path, "w") as f:
        json.dump(feature_names, f)
    print(f"Feature names saved to {path}")


def main():
    """Load models, evaluate on val set, select best, save artifacts."""
    print("Loading processed data...")
    df = pd.read_parquet(PROCESSED_DIR / "processed.parquet")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    models = {
        "xgboost": load_model("xgboost"),
        "random_forest": load_model("random_forest"),
    }

    # Evaluate on validation set for selection
    print("\n=== Validation Set Results ===")
    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val)
        metrics["model_name"] = name
        rows.append(metrics)
        print(f"  {name}: F1={metrics['f1_minority']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")

    comparison = pd.DataFrame(rows).sort_values("f1_minority", ascending=False).reset_index(drop=True)

    # Select and save
    decision = select_best_model(comparison)
    print(f"\nSelected: {decision['selected_model']}")
    print(f"Justification: {decision['justification']}")

    # Save feature names
    save_feature_names(list(X_train.columns))

    print("\nDone.")


if __name__ == "__main__":
    main()
