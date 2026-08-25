"""
Phase 4, Step 4.1: SHAP Faithfulness — Deletion / Insertion Curves
===================================================================
Measures how faithful SHAP feature-importance rankings are by iteratively
masking (deletion) or restoring (insertion) the top-k SHAP features and
recording the resulting change in predicted attack probability.

Key research question:
    Does SMOTE degrade SHAP explanation faithfulness?
    - Lower Deletion AUC  → more faithful (probability drops faster)
    - Higher Insertion AUC → more faithful (probability rises faster)

Usage:
    python src/evaluate_shap.py

Outputs:
    reports/figures/deletion_curves.png
    reports/figures/insertion_curves.png
    reports/shap_faithfulness.json
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RANDOM_STATE = 42
PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

SHAP_BACKGROUND_K = 1000        # k-means clusters for background
TRAIN_SAMPLE_SIZE = 5000        # rows of X_train for background
N_ATTACK_SAMPLES = 500          # test attack samples for faithfulness
MAX_K_FEATURES = 20             # mask/insert up to top-20 features

np.random.seed(RANDOM_STATE)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_data():
    """Load processed test data and a training sample for the background."""
    print("[1/6] Loading processed data...")

    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    X_train_sample = pd.read_csv(
        os.path.join(PROCESSED_DIR, "X_train.csv"),
        nrows=TRAIN_SAMPLE_SIZE,
    )

    print(f"       X_test:         {X_test.shape}")
    print(f"       X_train sample: {X_train_sample.shape}")
    print(f"       Attack samples in test: {int((y_test == 1).sum()):,}")

    return X_test, y_test, X_train_sample


def load_models():
    """Load both pre-trained XGBoost models."""
    print("[2/6] Loading trained models...")

    model_a = XGBClassifier()
    model_a.load_model(os.path.join(MODEL_DIR, "model_A_smote.json"))
    print("       Model A (SMOTE) loaded")

    model_b = XGBClassifier()
    model_b.load_model(os.path.join(MODEL_DIR, "model_B_weighted.json"))
    print("       Model B (Weighted) loaded")

    return model_a, model_b


# ──────────────────────────────────────────────
# SHAP + Faithfulness
# ──────────────────────────────────────────────
def create_background(X_train_sample):
    """Create background dataset shared across both models.

    Uses shap.maskers.Independent for SHAP >= 0.45 compatibility.
    We subsample to SHAP_BACKGROUND_K rows for speed.
    """
    print(f"\n[3/6] Creating background dataset (n={SHAP_BACKGROUND_K})...")
    t0 = time.time()
    # Subsample for background — deterministic via seed
    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X_train_sample), size=min(SHAP_BACKGROUND_K, len(X_train_sample)), replace=False)
    bg_data = X_train_sample.iloc[bg_idx].reset_index(drop=True)
    elapsed = time.time() - t0
    print(f"       Background created in {elapsed:.1f}s  (shape={bg_data.shape})")
    return bg_data


def compute_faithfulness_curves(model, explainer, X_attack, background_means,
                                 feature_names, model_label):
    """
    Compute deletion and insertion curves for a single model.

    Deletion: start from original sample, progressively mask top-k features
              (replace with background mean). Track P(attack).
    Insertion: start from baseline (all features = mean), progressively
               restore top-k features. Track P(attack).

    Returns:
        del_curve: np.ndarray of shape (N_ATTACK_SAMPLES, MAX_K_FEATURES + 1)
        ins_curve: np.ndarray of shape (N_ATTACK_SAMPLES, MAX_K_FEATURES + 1)
    """
    n_samples = len(X_attack)
    n_features = len(feature_names)

    del_curves = np.zeros((n_samples, MAX_K_FEATURES + 1))
    ins_curves = np.zeros((n_samples, MAX_K_FEATURES + 1))

    print(f"\n[4/6] Computing faithfulness curves for {model_label}...")
    print(f"       Samples: {n_samples}, Max-k: {MAX_K_FEATURES}")

    t0 = time.time()

    for i in range(n_samples):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_samples - i - 1) / rate if rate > 0 else 0
            print(f"       Sample {i+1}/{n_samples}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        sample = X_attack.iloc[i].values.astype(np.float64)

        # Get SHAP values for this sample
        explanation = explainer(X_attack.iloc[[i]])
        sv = explanation.values.flatten()
        # For binary classification with 2D output, take positive class
        if sv.ndim > 0 and len(sv) == 2 * len(feature_names):
            sv = sv[len(feature_names):]  # second class (Attack)
        elif hasattr(explanation, 'values') and explanation.values.ndim == 3:
            sv = explanation.values[0, :, 1]  # sample 0, all features, class 1

        # Rank features by absolute SHAP importance (descending)
        ranked_indices = np.argsort(-np.abs(sv))

        # ── Deletion Curve ──
        # k=0: original prediction
        del_sample = sample.copy()
        prob_del = model.predict_proba(
            pd.DataFrame([del_sample], columns=feature_names)
        )[0, 1]
        del_curves[i, 0] = prob_del

        for k in range(1, MAX_K_FEATURES + 1):
            if k - 1 < len(ranked_indices):
                feat_idx = ranked_indices[k - 1]
                del_sample[feat_idx] = background_means[feat_idx]
            prob_del = model.predict_proba(
                pd.DataFrame([del_sample], columns=feature_names)
            )[0, 1]
            del_curves[i, k] = prob_del

        # ── Insertion Curve ──
        # k=0: baseline (all features = mean)
        ins_sample = background_means.copy()
        prob_ins = model.predict_proba(
            pd.DataFrame([ins_sample], columns=feature_names)
        )[0, 1]
        ins_curves[i, 0] = prob_ins

        for k in range(1, MAX_K_FEATURES + 1):
            if k - 1 < len(ranked_indices):
                feat_idx = ranked_indices[k - 1]
                ins_sample[feat_idx] = sample[feat_idx]
            prob_ins = model.predict_proba(
                pd.DataFrame([ins_sample], columns=feature_names)
            )[0, 1]
            ins_curves[i, k] = prob_ins

    total_time = time.time() - t0
    print(f"       {model_label} done in {total_time:.1f}s "
          f"({total_time/n_samples*1000:.0f}ms/sample)")

    return del_curves, ins_curves


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def plot_curves(del_a, del_b, ins_a, ins_b, audc_a, audc_b, auic_a, auic_b):
    """Plot comparative deletion and insertion curves."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    k_values = np.arange(MAX_K_FEATURES + 1)

    # ── Publication-quality style ──
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    })

    # ── Deletion Curves ──
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_del_a = del_a.mean(axis=0)
    mean_del_b = del_b.mean(axis=0)
    std_del_a = del_a.std(axis=0)
    std_del_b = del_b.std(axis=0)

    ax.plot(k_values, mean_del_a, "o-", color="#e74c3c", linewidth=2,
            markersize=5, label=f"Model A (SMOTE) — AUDC={audc_a:.4f}")
    ax.fill_between(k_values, mean_del_a - std_del_a, mean_del_a + std_del_a,
                    alpha=0.15, color="#e74c3c")

    ax.plot(k_values, mean_del_b, "s-", color="#2ecc71", linewidth=2,
            markersize=5, label=f"Model B (Weighted) — AUDC={audc_b:.4f}")
    ax.fill_between(k_values, mean_del_b - std_del_b, mean_del_b + std_del_b,
                    alpha=0.15, color="#2ecc71")

    ax.set_xlabel("Number of Top-k Features Masked")
    ax.set_ylabel("Mean Predicted P(Attack)")
    ax.set_title("Deletion Curves — SHAP Faithfulness Comparison\n"
                 "(Lower AUDC = More Faithful Explanations)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_K_FEATURES)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    del_path = os.path.join(FIGURES_DIR, "deletion_curves.png")
    plt.savefig(del_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"       Deletion curve saved to {del_path}")

    # ── Insertion Curves ──
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_ins_a = ins_a.mean(axis=0)
    mean_ins_b = ins_b.mean(axis=0)
    std_ins_a = ins_a.std(axis=0)
    std_ins_b = ins_b.std(axis=0)

    ax.plot(k_values, mean_ins_a, "o-", color="#e74c3c", linewidth=2,
            markersize=5, label=f"Model A (SMOTE) — AUIC={auic_a:.4f}")
    ax.fill_between(k_values, mean_ins_a - std_ins_a, mean_ins_a + std_ins_a,
                    alpha=0.15, color="#e74c3c")

    ax.plot(k_values, mean_ins_b, "s-", color="#2ecc71", linewidth=2,
            markersize=5, label=f"Model B (Weighted) — AUIC={auic_b:.4f}")
    ax.fill_between(k_values, mean_ins_b - std_ins_b, mean_ins_b + std_ins_b,
                    alpha=0.15, color="#2ecc71")

    ax.set_xlabel("Number of Top-k Features Inserted")
    ax.set_ylabel("Mean Predicted P(Attack)")
    ax.set_title("Insertion Curves — SHAP Faithfulness Comparison\n"
                 "(Higher AUIC = More Faithful Explanations)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_K_FEATURES)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    ins_path = os.path.join(FIGURES_DIR, "insertion_curves.png")
    plt.savefig(ins_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"       Insertion curve saved to {ins_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Phase 4, Step 4.1: SHAP Faithfulness (Deletion/Insertion)")
    print("=" * 65)
    print()

    # ── Load ──
    X_test, y_test, X_train_sample = load_data()
    model_a, model_b = load_models()

    feature_names = list(X_test.columns)

    # ── Background (shared for fair comparison) ──
    background = create_background(X_train_sample)

    # Background mean values (for masking / baseline)
    background_means = np.array(X_train_sample.mean(axis=0).values,
                                dtype=np.float64)

    # ── Select 500 attack samples ──
    attack_idx = y_test[y_test == 1].index
    if len(attack_idx) > N_ATTACK_SAMPLES:
        rng = np.random.RandomState(RANDOM_STATE)
        selected_idx = rng.choice(attack_idx, size=N_ATTACK_SAMPLES,
                                  replace=False)
    else:
        selected_idx = attack_idx[:N_ATTACK_SAMPLES]

    X_attack = X_test.loc[selected_idx].reset_index(drop=True)
    print(f"\n       Selected {len(X_attack)} attack samples for evaluation")

    # ── SHAP explainers (SAME background for both) ──
    print("\n[3/6] Initializing TreeExplainers with shared background...")
    # Use maskers.Independent for SHAP >= 0.45 compatibility
    masker = shap.maskers.Independent(background, max_samples=SHAP_BACKGROUND_K)
    explainer_a = shap.TreeExplainer(model_a, masker)
    explainer_b = shap.TreeExplainer(model_b, masker)
    print("       Both explainers initialized")

    # ── Compute curves ──
    del_a, ins_a = compute_faithfulness_curves(
        model_a, explainer_a, X_attack, background_means,
        feature_names, "Model A (SMOTE)"
    )
    del_b, ins_b = compute_faithfulness_curves(
        model_b, explainer_b, X_attack, background_means,
        feature_names, "Model B (Weighted)"
    )

    # ── Compute AUC metrics ──
    print("\n[5/6] Computing AUC metrics...")
    k_values = np.arange(MAX_K_FEATURES + 1)

    # Normalize k to [0, 1] for comparable AUC
    k_norm = k_values / MAX_K_FEATURES

    # Mean curves
    mean_del_a = del_a.mean(axis=0)
    mean_del_b = del_b.mean(axis=0)
    mean_ins_a = ins_a.mean(axis=0)
    mean_ins_b = ins_b.mean(axis=0)

    audc_a = float(np.trapezoid(mean_del_a, k_norm))
    audc_b = float(np.trapezoid(mean_del_b, k_norm))
    auic_a = float(np.trapezoid(mean_ins_a, k_norm))
    auic_b = float(np.trapezoid(mean_ins_b, k_norm))

    print(f"       Model A (SMOTE):    AUDC={audc_a:.4f}, AUIC={auic_a:.4f}")
    print(f"       Model B (Weighted): AUDC={audc_b:.4f}, AUIC={auic_b:.4f}")
    print()

    # Interpretation
    if audc_b < audc_a:
        print("       [OK] RESULT: Model B has LOWER Deletion AUC -> "
              "MORE faithful SHAP explanations")
    else:
        print("       [!!] RESULT: Model A has LOWER Deletion AUC -> "
              "MORE faithful SHAP explanations")

    if auic_b > auic_a:
        print("       [OK] RESULT: Model B has HIGHER Insertion AUC -> "
              "MORE faithful SHAP explanations")
    else:
        print("       [!!] RESULT: Model A has HIGHER Insertion AUC -> "
              "MORE faithful SHAP explanations")

    # ── Plot ──
    print("\n[6/6] Generating comparative plots...")
    plot_curves(del_a, del_b, ins_a, ins_b, audc_a, audc_b, auic_a, auic_b)

    # ── Save metrics ──
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results = {
        "phase": "Phase 4, Step 4.1 — SHAP Faithfulness",
        "n_attack_samples": N_ATTACK_SAMPLES,
        "max_k_features": MAX_K_FEATURES,
        "shap_background_k": SHAP_BACKGROUND_K,
        "model_A_SMOTE": {
            "AUDC": round(audc_a, 6),
            "AUIC": round(auic_a, 6),
            "mean_deletion_curve": [round(v, 6) for v in mean_del_a.tolist()],
            "mean_insertion_curve": [round(v, 6) for v in mean_ins_a.tolist()],
        },
        "model_B_weighted": {
            "AUDC": round(audc_b, 6),
            "AUIC": round(auic_b, 6),
            "mean_deletion_curve": [round(v, 6) for v in mean_del_b.tolist()],
            "mean_insertion_curve": [round(v, 6) for v in mean_ins_b.tolist()],
        },
        "interpretation": {
            "deletion": (
                "Model B more faithful"
                if audc_b < audc_a
                else "Model A more faithful"
            ),
            "insertion": (
                "Model B more faithful"
                if auic_b > auic_a
                else "Model A more faithful"
            ),
            "note": ("Lower AUDC = explanations better identify important "
                     "features (faster probability drop). "
                     "Higher AUIC = explanations better identify important "
                     "features (faster probability rise)."),
        },
    }

    results_path = os.path.join(REPORTS_DIR, "shap_faithfulness.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"       Metrics saved to {results_path}")

    print("\n" + "=" * 65)
    print("  Step 4.1 COMPLETE — SHAP Faithfulness Evaluated")
    print("=" * 65)

    return results


if __name__ == "__main__":
    main()
