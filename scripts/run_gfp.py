"""Run Graph Feature Preprocessor on the IBM AML dataset.

This script is intended to run on EC2 (Linux x86_64) where snapml is available.
It will NOT work on M1 Mac.

Steps:
1. Load raw CSV
2. Convert string account IDs to numeric IDs
3. Convert timestamps to numeric (seconds since epoch)
4. Format as edge list: [txn_id, sender_id, receiver_id, timestamp]
5. Run GFP to generate 44 graph features
6. Combine with other features (amount, currency, payment format)
7. Save enriched dataset as parquet
8. Retrain models on enriched features
9. Re-evaluate and compare against baseline
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from snapml import GraphFeaturePreprocessor
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_curve, auc, matthews_corrcoef,
)

from src.config import (
    RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR,
    DATASET_NAME, ENABLE_SUBSAMPLE, SAMPLE_SIZE,
    RANDOM_STATE, TARGET_COL,
)


def load_and_prepare():
    """Load raw CSV and prepare for GFP.

    Returns:
        Tuple of (df, edge_list, labels, account_map).
    """
    print("Loading raw data...")
    df = pd.read_csv(RAW_DIR / DATASET_NAME)
    print(f"Raw shape: {df.shape}, Illicit: {df[TARGET_COL].sum()}")

    # NOTE: Do NOT subsample here. GFP must see the full graph.
    # Subsampling happens AFTER GFP enrichment in main().

    # Convert string account IDs to numeric
    print("Mapping account IDs to numeric...")
    all_accounts = pd.concat([df["Account"], df["Account.1"]]).unique()
    account_map = {acc: idx for idx, acc in enumerate(all_accounts)}
    df["sender_id"] = df["Account"].map(account_map)
    df["receiver_id"] = df["Account.1"].map(account_map)

    # Convert timestamps to seconds since epoch
    print("Converting timestamps...")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    base_time = df["Timestamp"].min()
    df["timestamp_numeric"] = (df["Timestamp"] - base_time).dt.total_seconds()

    # Create edge list: [txn_id, sender_id, receiver_id, timestamp]
    edge_list = np.column_stack([
        np.arange(len(df)),           # transaction ID
        df["sender_id"].values,        # sender
        df["receiver_id"].values,      # receiver
        df["timestamp_numeric"].values, # timestamp
    ]).astype(np.float64)

    labels = df[TARGET_COL].values

    print(f"Edge list shape: {edge_list.shape}")
    return df, edge_list, labels, account_map


def run_gfp(edge_list):
    """Run GraphFeaturePreprocessor on the edge list.

    Args:
        edge_list: numpy array of shape (N, 4) with columns
            [txn_id, sender_id, receiver_id, timestamp].

    Returns:
        Enriched numpy array with graph features appended.
    """
    print("Running Graph Feature Preprocessor...")
    gfp = GraphFeaturePreprocessor()
    gfp.set_params({
        "num_threads": 4,
        "time_window": 16,
        "fan": True, "fan_tw": 16, "fan_bins": [2, 3],
        "degree": True, "degree_tw": 16, "degree_bins": [2, 3],
        "scatter-gather": True, "scatter-gather_tw": 16, "scatter-gather_bins": [2, 3],
        "temp-cycle": True, "temp-cycle_tw": 16, "temp-cycle_bins": [2, 3],
        "vertex_stats": True, "vertex_stats_cols": [3], "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10],
        "lc-cycle": False,
    })

    enriched = gfp.fit_transform(edge_list)
    print(f"GFP output shape: {enriched.shape} (was {edge_list.shape})")
    return enriched


def build_feature_matrix(df, enriched):
    """Combine GFP graph features with other transaction features.

    Args:
        df: Original DataFrame with raw columns.
        enriched: GFP output array (N, 48).

    Returns:
        DataFrame with all features ready for training.
    """
    print("Building feature matrix...")

    # GFP features (skip first 4 columns which are the raw edge list)
    gfp_cols = [f"gfp_{i}" for i in range(enriched.shape[1] - 4)]
    gfp_df = pd.DataFrame(enriched[:, 4:], columns=gfp_cols)

    # Additional features from original data
    features = pd.DataFrame()
    features["log_amount_paid"] = np.log1p(df["Amount Paid"])
    features["same_currency"] = (df["Payment Currency"] == df["Receiving Currency"]).astype(int)
    features["same_bank"] = (df["From Bank"] == df["To Bank"]).astype(int)
    features["self_transfer"] = (df["Account"] == df["Account.1"]).astype(int)

    # One-hot encode Payment Format
    payment_dummies = pd.get_dummies(df["Payment Format"], prefix="pf", drop_first=False)

    # Combine all features
    result = pd.concat([gfp_df, features, payment_dummies], axis=1)
    result[TARGET_COL] = df[TARGET_COL].values

    print(f"Final feature matrix: {result.shape}")
    return result


def train_and_evaluate(df):
    """Train models on enriched features and evaluate.

    Args:
        df: Feature matrix with target column.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 60/20/20 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp,
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Train XGBoost
    print("\nTraining XGBoost...")
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Evaluate both on test set
    print("\n=== Test Set Results (GFP-enriched) ===")
    for name, model in {"xgboost": xgb, "random_forest": rf}.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)

        metrics = {
            "f1_minority": f1_score(y_test, y_pred, pos_label=1),
            "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "recall": recall_score(y_test, y_pred, pos_label=1),
            "pr_auc": auc(rec_vals, prec_vals),
            "mcc": matthews_corrcoef(y_test, y_pred),
        }
        print(f"  {name}: F1={metrics['f1_minority']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, MCC={metrics['mcc']:.4f}")

    # Save models
    import joblib
    joblib.dump(xgb, MODELS_DIR / "xgboost_gfp.joblib")
    joblib.dump(rf, MODELS_DIR / "random_forest_gfp.joblib")
    print(f"\nSaved GFP models to {MODELS_DIR}")

    # Save feature names
    with open(MODELS_DIR / "feature_names_gfp.json", "w") as f:
        json.dump(list(X_train.columns), f)
    print(f"Saved feature names to {MODELS_DIR / 'feature_names_gfp.json'}")


def subsample_features(df):
    """Subsample the enriched feature matrix for training.

    GFP runs on the full dataset for complete graph features.
    Subsampling happens here to keep training manageable.

    Args:
        df: Full enriched feature DataFrame with target column.

    Returns:
        Subsampled DataFrame.
    """
    if not ENABLE_SUBSAMPLE or len(df) <= SAMPLE_SIZE:
        return df

    print(f"Subsampling enriched data to {SAMPLE_SIZE}...")
    pos = df[df[TARGET_COL] == 1]
    neg = df[df[TARGET_COL] == 0]
    n_neg = min(SAMPLE_SIZE - len(pos), len(neg))
    neg_sample = neg.sample(n=n_neg, random_state=RANDOM_STATE)
    result = pd.concat([pos, neg_sample], ignore_index=True)
    result = result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Subsampled: {result.shape}, Illicit: {result[TARGET_COL].sum()}")
    return result


def main():
    df, edge_list, labels, account_map = load_and_prepare()
    enriched = run_gfp(edge_list)
    feature_df = build_feature_matrix(df, enriched)

    # Save full enriched dataset
    out_path = PROCESSED_DIR / "processed_gfp.parquet"
    feature_df.to_parquet(out_path, index=False)
    print(f"Saved enriched dataset to {out_path}")

    # Save account mapping for inference
    with open(MODELS_DIR / "account_map.json", "w") as f:
        json.dump(account_map, f)
    print(f"Saved account mapping to {MODELS_DIR / 'account_map.json'}")

    # Subsample AFTER GFP enrichment for training
    feature_df = subsample_features(feature_df)

    train_and_evaluate(feature_df)
    print("\nDone. Compare these results against baseline in outputs/baseline/")


if __name__ == "__main__":
    main()
