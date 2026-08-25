"""
Phase 4, Step 4.3: MAPIE Conformal Prediction — Uncertainty Quantification
===========================================================================
Wraps Model B (class-weighted XGBoost) with MAPIE to produce conformal
prediction sets at α=0.05 (95% target coverage).

Triage categories:
    {0}     → "Confident Benign"
    {1}     → "Confident Attack"
    {0, 1}  → "Uncertain/Review"  (flag for SOC analyst)

Usage:
    python src/uncertainty.py

Outputs:
    reports/mapie_results.json
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from mapie.classification import SplitConformalClassifier

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RANDOM_STATE = 42
PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"
REPORTS_DIR = "reports"

ALPHA = 0.05                   # 95% target coverage
CALIBRATION_SIZE = 0.20        # 20% of training data for calibration


def load_data():
    """Load processed train/test data."""
    print("[1/5] Loading processed data...")

    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    print(f"       X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"       Train attack ratio: {y_train.mean():.4f}")
    return X_train, y_train, X_test, y_test


def load_model():
    """Load Model B (class-weighted) for deployment."""
    print("[2/5] Loading Model B (weighted)...")
    model = XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "model_B_weighted.json"))
    print("       Model B loaded")
    return model


def create_calibration_set(X_train, y_train):
    """Split a calibration set from training data."""
    print(f"\n[3/5] Creating calibration set ({CALIBRATION_SIZE*100:.0f}% of training)...")

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train,
        test_size=CALIBRATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print(f"       X_fit (not used): {X_fit.shape}")
    print(f"       X_cal:            {X_cal.shape}")
    print(f"       Cal attack ratio: {y_cal.mean():.4f}")

    return X_cal, y_cal


def run_mapie(model, X_cal, y_cal, X_test, y_test):
    """Wrap Model B with MAPIE and generate prediction sets."""
    print(f"\n[4/5] Running MAPIE conformal prediction (alpha={ALPHA})...")

    # Wrap with MAPIE (prefit = model is already trained)
    mapie = SplitConformalClassifier(
        estimator=model,
        confidence_level=1 - ALPHA,
        prefit=True,
        random_state=RANDOM_STATE,
    )

    # Calibrate
    t0 = time.time()
    mapie.conformalize(X_cal, y_cal)
    cal_time = time.time() - t0
    print(f"       Calibration done in {cal_time:.1f}s")

    # Predict with prediction sets
    t0 = time.time()
    y_pred, y_sets = mapie.predict_set(X_test)
    pred_time = time.time() - t0
    print(f"       Prediction done in {pred_time:.1f}s "
          f"({pred_time/len(X_test)*1000:.2f}ms/sample)")

    return y_pred, y_sets, mapie


def analyze_results(y_pred, y_sets, y_test):
    """Map prediction sets to triage categories and compute metrics."""
    print(f"\n[5/5] Analyzing prediction sets...")

    n = len(y_test)

    # y_sets shape: (n_samples, n_classes, 1) for single alpha
    # y_sets[i, c, 0] = True if class c is in the prediction set
    pred_sets = y_sets[:, :, 0]  # shape (n, 2)

    # Map to triage categories
    categories = []
    for i in range(n):
        in_set = set()
        if pred_sets[i, 0]:
            in_set.add(0)
        if pred_sets[i, 1]:
            in_set.add(1)

        if in_set == {0}:
            categories.append("Confident Benign")
        elif in_set == {1}:
            categories.append("Confident Attack")
        elif in_set == {0, 1}:
            categories.append("Uncertain/Review")
        else:
            # Empty set (very rare, means no class meets the threshold)
            categories.append("Empty Set")

    categories = np.array(categories)

    # Counts
    n_benign = int((categories == "Confident Benign").sum())
    n_attack = int((categories == "Confident Attack").sum())
    n_uncertain = int((categories == "Uncertain/Review").sum())
    n_empty = int((categories == "Empty Set").sum())

    print(f"\n       Triage Distribution:")
    print(f"       |-- Confident Benign:   {n_benign:>7,} ({n_benign/n*100:5.2f}%)")
    print(f"       |-- Confident Attack:   {n_attack:>7,} ({n_attack/n*100:5.2f}%)")
    print(f"       |-- Uncertain/Review:   {n_uncertain:>7,} ({n_uncertain/n*100:5.2f}%)")
    if n_empty > 0:
        print(f"       +-- Empty Set:          {n_empty:>7,} ({n_empty/n*100:5.2f}%)")
    print(f"       Total:                  {n:>7,}")

    # Empirical coverage: fraction of samples where true label is in pred set
    y_test_arr = np.array(y_test)
    covered = 0
    for i in range(n):
        true_label = y_test_arr[i]
        if pred_sets[i, true_label]:
            covered += 1
    empirical_coverage = covered / n

    print(f"\n       Empirical Coverage: {empirical_coverage:.4f} "
          f"(target: {1 - ALPHA:.4f})")
    if empirical_coverage >= 1 - ALPHA:
        print(f"       [OK] Coverage MEETS the {1-ALPHA:.0%} target")
    else:
        print(f"       [WARN] Coverage BELOW the {1-ALPHA:.0%} target")

    # Point prediction accuracy
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n       Point Prediction Accuracy: {acc:.4f}")
    print(f"       Point Prediction F1:       {f1:.4f}")

    return {
        "alpha": ALPHA,
        "target_coverage": 1 - ALPHA,
        "empirical_coverage": round(float(empirical_coverage), 6),
        "coverage_meets_target": bool(empirical_coverage >= 1 - ALPHA),
        "total_samples": n,
        "triage_distribution": {
            "confident_benign": n_benign,
            "confident_benign_pct": round(n_benign / n * 100, 2),
            "confident_attack": n_attack,
            "confident_attack_pct": round(n_attack / n * 100, 2),
            "uncertain_review": n_uncertain,
            "uncertain_review_pct": round(n_uncertain / n * 100, 2),
            "empty_set": n_empty,
            "empty_set_pct": round(n_empty / n * 100, 2),
        },
        "point_prediction": {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
        },
    }


def main():
    print("=" * 65)
    print("  Phase 4, Step 4.3: MAPIE Conformal Prediction")
    print("=" * 65)
    print()

    # Load
    X_train, y_train, X_test, y_test = load_data()
    model = load_model()

    # Calibration set
    X_cal, y_cal = create_calibration_set(X_train, y_train)

    # MAPIE
    y_pred, y_sets, mapie = run_mapie(model, X_cal, y_cal, X_test, y_test)

    # Analyze
    results = analyze_results(y_pred, y_sets, y_test)
    results["phase"] = "Phase 4, Step 4.3 — MAPIE Conformal Prediction"
    results["model"] = "Model B (class-weighted)"
    results["calibration_size"] = CALIBRATION_SIZE

    # Save
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results_path = os.path.join(REPORTS_DIR, "mapie_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n       Results saved to {results_path}")

    print("\n" + "=" * 65)
    print("  Step 4.3 COMPLETE — MAPIE Uncertainty Quantified")
    print("=" * 65)

    return results


if __name__ == "__main__":
    main()
