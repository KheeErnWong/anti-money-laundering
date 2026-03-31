"""Evaluate AML models with imbalance-appropriate metrics."""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.config import OUTPUTS_DIR, PROCESSED_DIR
from src.data.preprocess import split_data
from src.models.train import load_model


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute imbalance-aware metrics for a single model.

    Uses minority-class F1 and PR-AUC as primary metrics instead of
    accuracy, which is misleading at 99% majority class.

    Args:
        model: Fitted model with predict() and predict_proba().
        X: Feature matrix.
        y: True labels.

    Returns:
        Dict of metric names to values.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    precision_vals, recall_vals, _ = precision_recall_curve(y, y_proba)
    pr_auc_val = auc(recall_vals, precision_vals)

    return {
        "f1_minority": f1_score(y, y_pred, pos_label=1),
        "precision": precision_score(y, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y, y_pred, pos_label=1),
        "pr_auc": pr_auc_val,
        "mcc": matthews_corrcoef(y, y_pred),
    }


def compare_models(models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Evaluate multiple models and return a ranked comparison table.

    Args:
        models: Dict of model_name -> fitted model.
        X: Feature matrix.
        y: True labels.

    Returns:
        DataFrame sorted by f1_minority descending.
    """
    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, X, y)
        metrics["model_name"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = df.sort_values("f1_minority", ascending=False).reset_index(drop=True)
    return df


def plot_confusion_matrix(model, X, y, title: str) -> plt.Figure:
    """Generate a confusion matrix heatmap.

    Args:
        model: Fitted model.
        X: Feature matrix.
        y: True labels.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.set_xticklabels(["Legitimate", "Suspicious"])
    ax.set_yticklabels(["Legitimate", "Suspicious"])
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(model, X, y, title: str) -> plt.Figure:
    """Generate a precision-recall curve.

    Args:
        model: Fitted model.
        X: Feature matrix.
        y: True labels.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    y_proba = model.predict_proba(X)[:, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(y, y_proba)
    pr_auc_val = auc(recall_vals, precision_vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_vals, precision_vals, label=f"PR AUC = {pr_auc_val:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def main():
    """Load models, evaluate on val and test sets, save results."""
    print("Loading processed data...")
    df = pd.read_parquet(PROCESSED_DIR / "processed.parquet")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("Loading models...")
    models = {
        "xgboost": load_model("xgboost"),
        "random_forest": load_model("random_forest"),
    }

    # Evaluate on validation set
    print("\n=== Validation Set Results ===")
    val_comparison = compare_models(models, X_val, y_val)
    print(val_comparison.to_string(index=False))

    # Evaluate on test set
    print("\n=== Test Set Results ===")
    test_comparison = compare_models(models, X_test, y_test)
    print(test_comparison.to_string(index=False))

    # Save comparison CSV
    test_comparison.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    print(f"\nSaved comparison to {OUTPUTS_DIR / 'model_comparison.csv'}")

    # Save plots for each model (on test set)
    for name, model in models.items():
        fig_cm = plot_confusion_matrix(
            model, X_test, y_test, f"{name} — Confusion Matrix"
        )
        fig_cm.savefig(OUTPUTS_DIR / f"{name}_confusion_matrix.png", dpi=150)
        plt.close(fig_cm)

        fig_pr = plot_precision_recall_curve(
            model, X_test, y_test, f"{name} — PR Curve"
        )
        fig_pr.savefig(OUTPUTS_DIR / f"{name}_pr_curve.png", dpi=150)
        plt.close(fig_pr)

    print(f"Saved plots to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
