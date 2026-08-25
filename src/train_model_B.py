"""
Phase 3, Step 3.2: Model B — Class-Weighted XGBoost
====================================================
Trains XGBClassifier on the ORIGINAL imbalanced training data
using scale_pos_weight for native cost-sensitive learning.
NO synthetic data is created.

Usage:
    python src/train_model_B.py

Outputs:
    models/model_B_weighted.json
    reports/figures/confusion_matrix_B.png
    reports/metrics_model_B.json
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
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RANDOM_STATE = 42
PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

MODEL_NAME = "model_B_weighted"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.json")


def load_data():
    """Load processed train/test splits."""
    print("[1/4] Loading processed data...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    print(f"       X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"       Train attack ratio: {y_train.mean():.4f}")
    return X_train, X_test, y_train, y_test


def compute_scale_pos_weight(y_train):
    """Compute class weight ratio: count(negative) / count(positive)."""
    n_benign = int((y_train == 0).sum())
    n_attack = int((y_train == 1).sum())
    weight = n_benign / n_attack
    print(f"\n[2/4] Computing scale_pos_weight...")
    print(f"       Benign: {n_benign:,}, Attack: {n_attack:,}")
    print(f"       scale_pos_weight = {weight:.4f}")
    return weight


def train_model(X_train, y_train, scale_pos_weight):
    """Train XGBClassifier with native class weighting (no SMOTE)."""
    print(f"\n[3/4] Training XGBoost (Model B: Class-weighted, spw={scale_pos_weight:.2f})...")

    # XGBoost hyperparameters — SAME as Model A except scale_pos_weight instead of SMOTE
    xgb_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    model = XGBClassifier(**xgb_params)

    t0 = time.time()
    model.fit(X_train, y_train, verbose=True)
    elapsed = time.time() - t0

    print(f"       Training completed in {elapsed:.1f}s")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"       Model saved to {MODEL_PATH}")

    return model, xgb_params


def evaluate_model(model, X_test, y_test, xgb_params):
    """Evaluate on pristine test set and save metrics + confusion matrix."""
    print("\n[4/4] Evaluating on pristine test set...")

    t0 = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    inference_time = (time.time() - t0) / len(X_test) * 1000  # ms per sample

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n       === Model B (Class-Weighted) Results ===")
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
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Greens",
                xticklabels=["Benign", "Attack"],
                yticklabels=["Benign", "Attack"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Model B (Class-Weighted) — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix_B.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"       Confusion matrix saved to {cm_path}")

    # Save metrics JSON
    metrics = {
        "model": MODEL_NAME,
        "method": "Class-weighted (scale_pos_weight)",
        "seed": RANDOM_STATE,
        "scale_pos_weight": round(xgb_params["scale_pos_weight"], 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "inference_ms_per_sample": round(inference_time, 3),
        "confusion_matrix": cm.tolist(),
        "xgboost_params": {k: v for k, v in xgb_params.items()},
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics_path = os.path.join(REPORTS_DIR, "metrics_model_B.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"       Metrics saved to {metrics_path}")

    return metrics


def main():
    print("=" * 60)
    print("  Phase 3, Step 3.2: Model B (Class-Weighted XGBoost)")
    print("=" * 60)
    print()

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Compute class weight (NO SMOTE!)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    # Train on ORIGINAL imbalanced data
    model, xgb_params = train_model(X_train, y_train, scale_pos_weight)

    # Evaluate on pristine test set
    metrics = evaluate_model(model, X_test, y_test, xgb_params)

    print("\n" + "=" * 60)
    print("  Step 3.2 COMPLETE — Model B saved")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
