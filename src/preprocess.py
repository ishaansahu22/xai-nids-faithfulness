"""
Phase 2: Data Preprocessing & Leakage Mitigation
=================================================
Loads raw CSE-CIC-IDS2018 CSVs, cleans them, and produces
train/test splits ready for the A/B experiment.

Leakage Prevention Rules Enforced:
  1. Identifier columns dropped BEFORE any analysis
  2. Deduplication BEFORE splitting (prevents identical rows in train & test)
  3. No scaling or resampling here — those are train-time only (Phase 3)

Usage:
    python src/preprocess.py

Outputs:
    data/processed/X_train.csv
    data/processed/X_test.csv
    data/processed/y_train.csv
    data/processed/y_test.csv
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
REPORTS_DIR = "reports"

RAW_FILES = [
    "02-14-2018.csv",   # FTP/SSH Brute Force
    "03-01-2018.csv",   # Infiltration
]

# Columns to drop — identifiers / memorization risks
# NOTE: The Kaggle version already removed Flow ID, Src IP, Dst IP.
# Timestamp is still present and must be dropped.
# Dst Port is KEPT because it carries genuine behavioral signal (port 22=SSH, 21=FTP).
COLUMNS_TO_DROP = ["Timestamp"]

LABEL_COLUMN = "Label"
BENIGN_LABEL = "Benign"


def md5_hash(filepath: str) -> str:
    """Compute MD5 hash of a file for reproducibility tracking."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_data() -> pd.DataFrame:
    """Load and concatenate all raw CSV files."""
    frames = []
    for fname in RAW_FILES:
        fpath = os.path.join(RAW_DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[ERROR] Raw file not found: {fpath}")
            sys.exit(1)

        print(f"[1/7] Loading {fname}...", end=" ")
        df = pd.read_csv(fpath, low_memory=False)
        print(f"shape={df.shape}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"[1/7] Combined shape: {combined.shape}")
    return combined


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    original_cols = list(df.columns)
    df.columns = df.columns.str.strip()
    renamed = sum(1 for a, b in zip(original_cols, df.columns) if a != b)
    print(f"[2/7] Stripped whitespace from {renamed} column name(s)")
    return df


def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that could cause memorization / leakage."""
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    cols_missing = [c for c in COLUMNS_TO_DROP if c not in df.columns]

    if cols_missing:
        print(f"[3/7] WARNING: Columns not found (already removed?): {cols_missing}")

    df = df.drop(columns=cols_present)
    print(f"[3/7] Dropped {len(cols_present)} identifier column(s): {cols_present}")
    return df


def cast_and_clean_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns to float32, replace inf with NaN, drop NaN rows."""
    label_col = df[LABEL_COLUMN].copy()
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]

    # Cast to numeric (coerce errors to NaN)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cast to float32 for memory efficiency
    df[feature_cols] = df[feature_cols].astype(np.float32)

    # Replace infinities
    inf_count = np.isinf(df[feature_cols]).sum().sum()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    print(f"[4/7] Replaced {inf_count} infinite values with NaN")

    # Drop NaN rows
    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    print(f"[4/7] Dropped {rows_dropped} rows containing NaN ({rows_dropped/rows_before*100:.2f}%)")

    # Restore label column
    df[LABEL_COLUMN] = label_col.loc[df.index]
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows to prevent train/test leakage."""
    rows_before = len(df)
    df = df.drop_duplicates()
    rows_dropped = rows_before - len(df)
    print(f"[5/7] Removed {rows_dropped} duplicate rows ({rows_dropped/rows_before*100:.2f}%)")
    return df


def binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-class labels to binary: Benign=0, Attack=1."""
    print(f"\n[6/7] Original label distribution:")
    label_counts = df[LABEL_COLUMN].value_counts()
    for label, count in label_counts.items():
        print(f"       {label}: {count:,} ({count/len(df)*100:.2f}%)")

    df[LABEL_COLUMN] = (df[LABEL_COLUMN] != BENIGN_LABEL).astype(int)

    benign_count = (df[LABEL_COLUMN] == 0).sum()
    attack_count = (df[LABEL_COLUMN] == 1).sum()
    imbalance_ratio = benign_count / max(attack_count, 1)
    print(f"\n       Binary: Benign={benign_count:,}, Attack={attack_count:,}")
    print(f"       Imbalance ratio (Benign/Attack): {imbalance_ratio:.2f}")
    return df


def stratified_split(df: pd.DataFrame):
    """Perform stratified 80/20 train-test split."""
    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\n[7/7] Stratified split (seed={RANDOM_STATE}, test_size={TEST_SIZE}):")
    print(f"       X_train: {X_train.shape}  |  y_train: {y_train.shape}")
    print(f"       X_test:  {X_test.shape}  |  y_test:  {y_test.shape}")
    print(f"       Train attack ratio: {y_train.mean():.4f}")
    print(f"       Test  attack ratio: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test


def save_processed(X_train, X_test, y_train, y_test):
    """Save processed splits to CSV files."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    paths = {
        "X_train": os.path.join(PROCESSED_DATA_DIR, "X_train.csv"),
        "X_test": os.path.join(PROCESSED_DATA_DIR, "X_test.csv"),
        "y_train": os.path.join(PROCESSED_DATA_DIR, "y_train.csv"),
        "y_test": os.path.join(PROCESSED_DATA_DIR, "y_test.csv"),
    }

    X_train.to_csv(paths["X_train"], index=False)
    X_test.to_csv(paths["X_test"], index=False)
    y_train.to_csv(paths["y_train"], index=False)
    y_test.to_csv(paths["y_test"], index=False)

    print(f"\n[DONE] Saved processed data to {PROCESSED_DATA_DIR}/")
    for name, path in paths.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"       {name}: {size_mb:.1f} MB")

    return paths


def save_preprocessing_report(X_train, X_test, y_train, y_test, paths):
    """Save a JSON report with shapes, distributions, and hashes for reproducibility."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report = {
        "phase": "Phase 2 — Data Preprocessing",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "raw_files": RAW_FILES,
        "columns_dropped": COLUMNS_TO_DROP,
        "feature_count": X_train.shape[1],
        "feature_names": list(X_train.columns),
        "shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
            "y_train": list(y_train.shape),
            "y_test": list(y_test.shape),
        },
        "class_distribution": {
            "train": {
                "benign": int((y_train == 0).sum()),
                "attack": int((y_train == 1).sum()),
            },
            "test": {
                "benign": int((y_test == 0).sum()),
                "attack": int((y_test == 1).sum()),
            },
        },
        "file_hashes": {name: md5_hash(path) for name, path in paths.items()},
    }

    report_path = os.path.join(REPORTS_DIR, "preprocessing_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"       Report saved to {report_path}")

    return report


def main():
    print("=" * 60)
    print("  Phase 2: Data Preprocessing & Leakage Mitigation")
    print("=" * 60)
    print()

    # Step 1: Load raw data
    df = load_raw_data()

    # Step 2: Clean column names
    df = clean_columns(df)

    # Step 3: Drop identifier columns
    df = drop_identifiers(df)

    # Step 4: Cast numerics, replace inf, drop NaN
    df = cast_and_clean_numerics(df)

    # Step 5: Deduplicate
    df = deduplicate(df)

    # Step 6: Binarize labels
    df = binarize_labels(df)

    # Step 7: Stratified split
    X_train, X_test, y_train, y_test = stratified_split(df)

    # Save outputs
    paths = save_processed(X_train, X_test, y_train, y_test)

    # Save report
    report = save_preprocessing_report(X_train, X_test, y_train, y_test, paths)

    print("\n" + "=" * 60)
    print("  Phase 2 COMPLETE — Ready for Phase 3 (A/B Experiment)")
    print("=" * 60)
    print(f"\n  Features:     {report['feature_count']}")
    print(f"  Train samples: {report['shapes']['X_train'][0]:,}")
    print(f"  Test samples:  {report['shapes']['X_test'][0]:,}")
    print(f"  Train attacks: {report['class_distribution']['train']['attack']:,}")
    print(f"  Test attacks:  {report['class_distribution']['test']['attack']:,}")

    return report


if __name__ == "__main__":
    main()
