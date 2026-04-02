"""Preprocess IBM AML transactions: subsample, engineer features, split."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DIR, PROCESSED_DIR, DATASET_NAME,
    ENABLE_SUBSAMPLE, SAMPLE_SIZE, RANDOM_STATE, TARGET_COL,
)


def compute_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute graph-based features on the FULL dataset before subsampling.

    These features reflect each account's behavior across all transactions,
    not just the subsample. Computing after subsampling would destroy the
    signal (most accounts would appear only once).

    Based on the Graph Feature Preprocessor paper (arXiv 2402.08593),
    fan-in/fan-out features alone improve F1 by +30 percentage points.

    Args:
        df: Full raw transaction DataFrame.

    Returns:
        DataFrame with graph feature columns added.
    """
    print("Computing graph features on full dataset...")
    df = df.copy()

    # Fan-out: unique receivers per sender (high = distributing money)
    fan_out = df.groupby("Account")["Account.1"].transform("nunique")
    df["fan_out"] = fan_out

    # Fan-in: unique senders per receiver (high = collecting from many)
    fan_in = df.groupby("Account.1")["Account"].transform("nunique")
    df["fan_in"] = fan_in

    # Sender transaction count (activity level)
    df["sender_txn_count"] = df.groupby("Account")["Account"].transform("count")

    # Receiver transaction count
    df["receiver_txn_count"] = df.groupby("Account.1")["Account.1"].transform("count")

    print(f"  fan_out range: {df['fan_out'].min()}-{df['fan_out'].max()}")
    print(f"  fan_in range: {df['fan_in'].min()}-{df['fan_in'].max()}")

    return df


def subsample(df: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Stratified subsample: keep all positives, downsample negatives.

    Args:
        df: DataFrame with graph features already computed.
        sample_size: Target total rows.

    Returns:
        Subsampled DataFrame.
    """
    if len(df) <= sample_size:
        return df

    pos = df[df[TARGET_COL] == 1]
    neg = df[df[TARGET_COL] == 0]

    n_pos = len(pos)
    n_neg = min(sample_size - n_pos, len(neg))

    neg_sample = neg.sample(n=n_neg, random_state=RANDOM_STATE)
    result = pd.concat([pos, neg_sample], ignore_index=True)
    result = result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Subsampled: {result.shape}, Illicit: {result[TARGET_COL].sum()}")
    return result


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load raw CSV.

    Args:
        path: Path to the raw CSV. Defaults to RAW_DIR/DATASET_NAME.

    Returns:
        Raw DataFrame.
    """
    if path is None:
        path = RAW_DIR / DATASET_NAME

    print(f"Loading {path}...")
    df = pd.read_csv(path)
    print(f"Raw shape: {df.shape}, Illicit: {df[TARGET_COL].sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal baseline feature engineering from EDA findings.

    Creates features:
        - log_amount_paid: log-transformed Amount Paid
        - same_currency: 1 if Payment Currency == Receiving Currency
        - same_bank: 1 if From Bank == To Bank
        - self_transfer: 1 if sender account == receiver account
        - pf_*: one-hot encoded Payment Format
        - fan_out: unique receivers per sender account
        - fan_in: unique senders per receiver account
        - sender_txn_count: total transactions sent by this account
        - receiver_txn_count: total transactions received by this account

    Drops raw columns not used as features (Timestamp, Account IDs,
    Bank IDs, raw amounts, currency strings, Payment Format).

    Args:
        df: Raw transaction DataFrame with original columns.

    Returns:
        DataFrame with engineered features and target column only.
    """
    df = df.copy()

    # Log-transformed amount (strong signal: laundering txns ~6x larger)
    df["log_amount_paid"] = np.log1p(df["Amount Paid"])

    # Same currency flag (cross-currency could indicate layering)
    df["same_currency"] = (
        df["Payment Currency"] == df["Receiving Currency"]
    ).astype(int)

    # Same bank flag
    df["same_bank"] = (df["From Bank"] == df["To Bank"]).astype(int)

    # Self-transfer flag (lower laundering rate per EDA)
    df["self_transfer"] = (df["Account"] == df["Account.1"]).astype(int)

    # One-hot encode Payment Format (ACH has 7.5x average laundering rate)
    payment_dummies = pd.get_dummies(
        df["Payment Format"], prefix="pf", drop_first=False
    )
    df = pd.concat([df, payment_dummies], axis=1)

    # Drop columns not used as features
    drop_cols = [
        "Timestamp", "From Bank", "To Bank",
        "Account", "Account.1",
        "Amount Received", "Amount Paid",
        "Receiving Currency", "Payment Currency",
        "Payment Format",
    ]
    df = df.drop(columns=drop_cols)

    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified 60/20/20 train/val/test split.

    Splits in two passes: first 80/20 (temp/test), then the 80%
    is split 75/25 to get 60/20 (train/val) of the total.

    Args:
        df: Feature-engineered DataFrame including the target column.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # First split: 80% temp, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    # Second split: 75% train, 25% val (0.25 * 0.8 = 0.2 of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_preprocessing(raw_path: Path | None = None) -> Path:
    """Full preprocessing pipeline. Saves processed data to disk.

    Flow: load raw -> graph features (on full data) -> subsample -> engineer features -> save.

    Args:
        raw_path: Path to the raw CSV. Defaults to RAW_DIR/DATASET_NAME.

    Returns:
        Path to the saved parquet file.
    """
    df = load_raw(raw_path)
    df = compute_graph_features(df)

    if ENABLE_SUBSAMPLE:
        df = subsample(df)
    else:
        print("Subsampling disabled, using full dataset.")

    df = engineer_features(df)

    out_path = PROCESSED_DIR / "processed.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved processed data to {out_path}")
    return out_path


if __name__ == "__main__":
    run_preprocessing()
