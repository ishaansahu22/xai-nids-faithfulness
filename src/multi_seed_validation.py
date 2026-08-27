"""
Phase 3, Step 3.4: 5-Seed Statistical Validation
==================================================
Trains both Model A (SMOTE) and Model B (Class-Weighted) across 5 random
seeds to quantify performance variance and test statistical significance.

The data split is FIXED (produced by preprocess.py with seed=42).
Only the model training seed and SMOTE seed are varied.

Seeds: [42, 101, 202, 303, 404]

Usage:
    python src/multi_seed_validation.py

Outputs:
    reports/multi_seed_results.json
    reports/figures/multi_seed_boxplot.png
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SEEDS = [42, 101, 202, 303, 404]
PROCESSED_DIR = os.path.join("data", "processed")
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# XGBoost hyperparameters (shared base — same as Phase 3)
XGBOOST_BASE_PARAMS = {
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
    "n_jobs": -1,
    "tree_method": "hist",
}


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_data():
    """Load the FIXED processed train/test splits."""
    print("[1/5] Loading processed data (fixed split)...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    print(f"       X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"       Train attack ratio: {y_train.mean():.4f}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# Single-seed training + evaluation
# ──────────────────────────────────────────────
def train_and_evaluate_model_a(X_train, y_train, X_test, y_test, seed):
    """Train Model A (SMOTE + XGBoost) with a given seed and return metrics."""
    # Apply SMOTE
    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Train XGBoost
    params = {**XGBOOST_BASE_PARAMS, "random_state": seed}
    model = XGBClassifier(**params)
    model.fit(X_res, y_res, verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }, model


def train_and_evaluate_model_b(X_train, y_train, X_test, y_test, seed):
    """Train Model B (class-weighted XGBoost) with a given seed and return metrics."""
    # Compute scale_pos_weight
    n_benign = int((y_train == 0).sum())
    n_attack = int((y_train == 1).sum())
    spw = n_benign / n_attack

    # Train XGBoost
    params = {**XGBOOST_BASE_PARAMS, "random_state": seed, "scale_pos_weight": spw}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }, model


# ──────────────────────────────────────────────
# Multi-seed loop
# ──────────────────────────────────────────────
def run_multi_seed(X_train, X_test, y_train, y_test):
    """Train both models across all seeds and collect metrics."""
    results_a = []
    results_b = []
    models_a = {}
    models_b = {}

    print(f"\n[2/5] Running multi-seed validation ({len(SEEDS)} seeds)...\n")

    for i, seed in enumerate(SEEDS):
        print(f"  -- Seed {seed} ({i+1}/{len(SEEDS)}) --")

        # Model A
        t0 = time.time()
        metrics_a, model_a = train_and_evaluate_model_a(
            X_train, y_train, X_test, y_test, seed
        )
        elapsed_a = time.time() - t0
        results_a.append(metrics_a)
        models_a[seed] = model_a
        print(f"     Model A (SMOTE):    F1={metrics_a['f1_score']:.4f}  "
              f"P={metrics_a['precision']:.4f}  R={metrics_a['recall']:.4f}  "
              f"({elapsed_a:.1f}s)")

        # Model B
        t0 = time.time()
        metrics_b, model_b = train_and_evaluate_model_b(
            X_train, y_train, X_test, y_test, seed
        )
        elapsed_b = time.time() - t0
        results_b.append(metrics_b)
        models_b[seed] = model_b
        print(f"     Model B (Weighted): F1={metrics_b['f1_score']:.4f}  "
              f"P={metrics_b['precision']:.4f}  R={metrics_b['recall']:.4f}  "
              f"({elapsed_b:.1f}s)")
        print()

    return results_a, results_b, models_a, models_b


# ──────────────────────────────────────────────
# Statistical analysis
# ──────────────────────────────────────────────
def compute_statistics(results_a, results_b):
    """Compute mean, std, and paired t-test for each metric."""
    print("[3/5] Computing statistics...")

    metrics_keys = ["accuracy", "precision", "recall", "f1_score"]
    summary = {"model_A_SMOTE": {}, "model_B_weighted": {}, "paired_t_test": {}}

    for key in metrics_keys:
        vals_a = np.array([r[key] for r in results_a])
        vals_b = np.array([r[key] for r in results_b])

        summary["model_A_SMOTE"][key] = {
            "mean": round(float(vals_a.mean()), 4),
            "std": round(float(vals_a.std()), 4),
            "values": [round(float(v), 4) for v in vals_a],
        }
        summary["model_B_weighted"][key] = {
            "mean": round(float(vals_b.mean()), 4),
            "std": round(float(vals_b.std()), 4),
            "values": [round(float(v), 4) for v in vals_b],
        }

        # Paired t-test (two-tailed)
        t_stat, p_value = stats.ttest_rel(vals_a, vals_b)
        summary["paired_t_test"][key] = {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant_at_005": bool(p_value < 0.05),
        }

        print(f"       {key}:")
        print(f"         Model A: {vals_a.mean():.4f} ± {vals_a.std():.4f}")
        print(f"         Model B: {vals_b.mean():.4f} ± {vals_b.std():.4f}")
        print(f"         t={t_stat:.4f}, p={p_value:.6f} "
              f"{'*** SIGNIFICANT' if p_value < 0.05 else '(not significant)'}")

    return summary


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def plot_boxplot(results_a, results_b):
    """Create a publication-quality box-plot comparing both models across seeds."""
    print("\n[4/5] Generating box-plot figure...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    metrics_keys = ["accuracy", "precision", "recall", "f1_score"]
    labels_nice = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "5-Seed Statistical Validation — Model A (SMOTE) vs Model B (Weighted)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    colors_a = "#e74c3c"
    colors_b = "#2ecc71"

    for ax, key, label in zip(axes, metrics_keys, labels_nice):
        vals_a = [r[key] for r in results_a]
        vals_b = [r[key] for r in results_b]

        bp = ax.boxplot(
            [vals_a, vals_b],
            labels=["Model A\n(SMOTE)", "Model B\n(Weighted)"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
        )

        bp["boxes"][0].set_facecolor(colors_a)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(colors_b)
        bp["boxes"][1].set_alpha(0.6)

        # Scatter individual seed points
        jitter_a = np.random.RandomState(42).uniform(-0.08, 0.08, len(vals_a))
        jitter_b = np.random.RandomState(42).uniform(-0.08, 0.08, len(vals_b))
        ax.scatter(
            1 + jitter_a, vals_a, color=colors_a, edgecolor="black",
            s=40, zorder=3, alpha=0.8
        )
        ax.scatter(
            2 + jitter_b, vals_b, color=colors_b, edgecolor="black",
            s=40, zorder=3, alpha=0.8
        )

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylabel("Score")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "multi_seed_boxplot.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"       Box-plot saved to {fig_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Phase 3, Step 3.4: 5-Seed Statistical Validation")
    print("=" * 65)
    print()

    # Load fixed data split
    X_train, X_test, y_train, y_test = load_data()

    # Train across all seeds
    results_a, results_b, models_a, models_b = run_multi_seed(
        X_train, X_test, y_train, y_test
    )

    # Statistical analysis
    summary = compute_statistics(results_a, results_b)

    # Box-plot
    plot_boxplot(results_a, results_b)

    # Save results
    print("\n[5/5] Saving results...")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report = {
        "phase": "Phase 3, Step 3.4 — 5-Seed Statistical Validation",
        "seeds": SEEDS,
        "data_split_seed": 42,
        "note": "Data split is FIXED (seed=42). Only model/SMOTE seeds are varied.",
        **summary,
    }

    report_path = os.path.join(REPORTS_DIR, "multi_seed_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"       Results saved to {report_path}")

    # Print summary table
    print("\n" + "=" * 65)
    print("  Step 3.4 COMPLETE — 5-Seed Validation Summary")
    print("=" * 65)
    print(f"\n  {'Metric':<12} {'Model A (SMOTE)':<22} {'Model B (Weighted)':<22} {'p-value':<12}")
    print(f"  {'-'*12} {'-'*22} {'-'*22} {'-'*12}")
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        ma = summary["model_A_SMOTE"][key]
        mb = summary["model_B_weighted"][key]
        pv = summary["paired_t_test"][key]["p_value"]
        sig = " ***" if pv < 0.05 else ""
        print(f"  {key:<12} {ma['mean']:.4f} ± {ma['std']:.4f}        "
              f"{mb['mean']:.4f} ± {mb['std']:.4f}        "
              f"{pv:.6f}{sig}")

    return report, models_a, models_b


if __name__ == "__main__":
    main()
