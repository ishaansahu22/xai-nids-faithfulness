"""
Phase 3, Step 3.1: Model A — SMOTE-Balanced XGBoost
====================================================
Applies SMOTE to the training set ONLY, then trains XGBClassifier.
Evaluates on the pristine (untouched) test set.

Usage:
    python src/train_model_A.py

Outputs:
    models/model_A_smote.json
    reports/figures/confusion_matrix_A.png
    reports/metrics_model_A.json
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RANDOM_STATE = 42
PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

MODEL_NAME = "model_A_smote"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.json")

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}


def load_data():
    """Load processed train/test splits."""
    print("[1/5] Loading processed data...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    print(f"       X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"       Train attack ratio: {y_train.mean():.4f}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to the training set ONLY."""
    print("\n[2/5] Applying SMOTE to training set...")
    print(f"       Before: Benign={int((y_train==0).sum()):,}, Attack={int((y_train==1).sum()):,}")

    smote = SMOTE(random_state=RANDOM_STATE)
    t0 = time.time()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    elapsed = time.time() - t0

    print(f"       After:  Benign={int((y_resampled==0).sum()):,}, Attack={int((y_resampled==1).sum()):,}")
    print(f"       SMOTE generated {len(X_resampled) - len(X_train):,} synthetic samples in {elapsed:.1f}s")
    return X_resampled, y_resampled


def train_model(X_train, y_train):
    """Train XGBClassifier on SMOTE-balanced data."""
    print("\n[3/5] Training XGBoost (Model A: SMOTE-balanced)...")
    model = XGBClassifier(**XGBOOST_PARAMS)

    t0 = time.time()
    model.fit(X_train, y_train, verbose=True)
    elapsed = time.time() - t0

    print(f"       Training completed in {elapsed:.1f}s")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"       Model saved to {MODEL_PATH}")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate on pristine test set and save metrics + confusion matrix."""
    print("\n[4/5] Evaluating on pristine test set...")

    t0 = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    inference_time = (time.time() - t0) / len(X_test) * 1000  # ms per sample

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n       === Model A (SMOTE) Results ===")
    print(f"       Accuracy:  {acc:.4f}")
    print(f"       Precision: {prec:.4f}")
    print(f"       Recall:    {rec:.4f}")
    print(f"       F1-Score:  {f1:.4f}")
    print(f"       Inference: {inference_time:.3f} ms/sample")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Benign', 'Attack'])}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Benign", "Attack"],
                yticklabels=["Benign", "Attack"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Model A (SMOTE-Balanced) — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix_A.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"       Confusion matrix saved to {cm_path}")

    # Save metrics JSON
    metrics = {
        "model": MODEL_NAME,
        "method": "SMOTE-balanced",
        "seed": RANDOM_STATE,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "inference_ms_per_sample": round(inference_time, 3),
        "confusion_matrix": cm.tolist(),
        "xgboost_params": XGBOOST_PARAMS,
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics_path = os.path.join(REPORTS_DIR, "metrics_model_A.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"       Metrics saved to {metrics_path}")

    return metrics


def main():
    print("=" * 60)
    print("  Phase 3, Step 3.1: Model A (SMOTE-Balanced XGBoost)")
    print("=" * 60)
    print()

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Apply SMOTE (training set only!)
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Train
    model = train_model(X_train_smote, y_train_smote)

    # Evaluate on pristine test set
    metrics = evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("  Step 3.1 COMPLETE — Model A saved")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
