"""Train XGBoost and Random Forest classifiers for AML detection."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE
from src.data.preprocess import split_data


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Train XGBoost with scale_pos_weight for class imbalance.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted XGBClassifier.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """Train Random Forest with balanced class weights.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, name: str):
    """Save a trained model to disk.

    Args:
        model: Fitted sklearn/xgboost model.
        name: Filename without extension (e.g. "xgboost").
    """
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def load_model(name: str):
    """Load a trained model from disk.

    Args:
        name: Filename without extension (e.g. "xgboost").

    Returns:
        Fitted model.
    """
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


def main():
    """Load processed data, train both models, save to disk."""
    print("Loading processed data...")
    df = pd.read_parquet(PROCESSED_DIR / "processed.parquet")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.4%}")

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    save_model(xgb_model, "xgboost")

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, "random_forest")

    print("\nDone. Models saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
