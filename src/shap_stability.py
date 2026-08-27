"""
Phase 4, Step 4.3: Multi-Seed SHAP Stability (Jaccard Top-5)
=============================================================
Measures how stable SHAP feature-importance rankings are across the
5 random seeds used in Step 3.4.

For each seed's trained model (Model A and Model B), we:
  1. Compute TreeSHAP values on the SAME set of test attack samples
  2. Extract the global Top-5 features (by mean |SHAP|)
  3. Compute pairwise Jaccard similarity of Top-5 sets across seeds

Higher Jaccard similarity → more stable/reproducible explanations.

This script retrains models per-seed (fast, since XGBoost on GPU/hist)
to avoid needing persisted multi-seed model files.

Usage:
    python src/shap_stability.py

Outputs:
    reports/shap_stability.json
    reports/figures/shap_stability_heatmap.png
"""

import os
import json
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SEEDS = [42, 101, 202, 303, 404]
PROCESSED_DIR = os.path.join("data", "processed")
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

SHAP_BACKGROUND_K = 1000        # background samples for SHAP
TRAIN_SAMPLE_SIZE = 5000        # rows of X_train for background creation
N_ATTACK_SAMPLES = 200          # test attack samples for SHAP (reduced for speed)
TOP_K = 5                       # Top-K features for Jaccard comparison

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
    """Load processed data splits."""
    print("[1/6] Loading processed data...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    print(f"       X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# Model training (per seed)
# ──────────────────────────────────────────────
def train_model_a(X_train, y_train, seed):
    """Train Model A (SMOTE + XGBoost) for a given seed."""
    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    params = {**XGBOOST_BASE_PARAMS, "random_state": seed}
    model = XGBClassifier(**params)
    model.fit(X_res, y_res, verbose=False)
    return model


def train_model_b(X_train, y_train, seed):
    """Train Model B (class-weighted XGBoost) for a given seed."""
    n_benign = int((y_train == 0).sum())
    n_attack = int((y_train == 1).sum())
    spw = n_benign / n_attack
    params = {**XGBOOST_BASE_PARAMS, "random_state": seed, "scale_pos_weight": spw}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


# ──────────────────────────────────────────────
# SHAP Top-K extraction
# ──────────────────────────────────────────────
def _patch_shap_xgboost_compat():
    """Monkey-patch SHAP to handle XGBoost 3.x base_score format '[5E-1]'.

    XGBoost 3.x wraps base_score in brackets, but SHAP 0.49.x expects
    a plain float string. This one-time patch fixes the parser.
    """
    import shap.explainers._tree as _tree_module

    OrigLoader = _tree_module.XGBTreeModelLoader
    if getattr(OrigLoader, "_patched", False):
        return  # already patched

    _orig_init = OrigLoader.__init__

    def _patched_init(self, xgb_model):
        # Temporarily patch float() to strip brackets
        import builtins
        _orig_float = builtins.float

        def _safe_float(val):
            if isinstance(val, str):
                val = val.strip("[]")
            return _orig_float(val)

        builtins.float = _safe_float
        try:
            _orig_init(self, xgb_model)
        finally:
            builtins.float = _orig_float

    OrigLoader.__init__ = _patched_init
    OrigLoader._patched = True


# Apply patch at import time
_patch_shap_xgboost_compat()


def get_top_k_features(model, X_attack, feature_names, k=5):
    """
    Compute TreeSHAP values and return the global Top-K features
    ranked by mean absolute SHAP value.

    Uses TreeExplainer (TreeSHAP uses the tree structure directly).
    """
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_attack)

    # shap_values may be a list [class_0, class_1] for binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 = Attack

    # Handle 3D output
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Mean absolute SHAP across all samples
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Top-K feature indices and names
    top_k_idx = np.argsort(-mean_abs_shap)[:k]
    top_k_names = [feature_names[i] for i in top_k_idx]

    return set(top_k_names), mean_abs_shap


def jaccard_similarity(set_a, set_b):
    """Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Phase 4, Step 4.3: Multi-Seed SHAP Stability (Jaccard Top-5)")
    print("=" * 65)
    print()

    # Load data
    X_train, X_test, y_train, y_test = load_data()
    feature_names = list(X_test.columns)

    # Select attack samples (FIXED across seeds for fair comparison)
    attack_idx = y_test[y_test == 1].index
    rng = np.random.RandomState(42)
    selected_idx = rng.choice(attack_idx, size=min(N_ATTACK_SAMPLES, len(attack_idx)),
                              replace=False)
    X_attack = X_test.loc[selected_idx].reset_index(drop=True)
    print(f"       Selected {len(X_attack)} attack samples for SHAP evaluation")

    # Note: TreeExplainer uses tree structure directly, no background needed
    print("\n[2/6] TreeSHAP uses tree structure directly (no background needed)")

    # ── Train models and extract Top-K per seed ──
    top_k_a = {}  # seed -> set of top-k feature names
    top_k_b = {}
    shap_rankings_a = {}
    shap_rankings_b = {}

    print(f"\n[3/6] Training models and computing SHAP for {len(SEEDS)} seeds...\n")

    for i, seed in enumerate(SEEDS):
        print(f"  -- Seed {seed} ({i+1}/{len(SEEDS)}) --")

        # Model A
        t0 = time.time()
        model_a = train_model_a(X_train, y_train, seed)
        print(f"     Model A trained ({time.time()-t0:.1f}s)", end=" -> ")
        t0 = time.time()
        top_set_a, mean_abs_a = get_top_k_features(
            model_a, X_attack, feature_names, TOP_K
        )
        top_k_a[seed] = top_set_a
        shap_rankings_a[seed] = sorted(
            zip(feature_names, mean_abs_a.tolist()),
            key=lambda x: -x[1]
        )[:TOP_K]
        print(f"SHAP Top-{TOP_K}: {sorted(top_set_a)} ({time.time()-t0:.1f}s)")

        # Model B
        t0 = time.time()
        model_b = train_model_b(X_train, y_train, seed)
        print(f"     Model B trained ({time.time()-t0:.1f}s)", end=" -> ")
        t0 = time.time()
        top_set_b, mean_abs_b = get_top_k_features(
            model_b, X_attack, feature_names, TOP_K
        )
        top_k_b[seed] = top_set_b
        shap_rankings_b[seed] = sorted(
            zip(feature_names, mean_abs_b.tolist()),
            key=lambda x: -x[1]
        )[:TOP_K]
        print(f"SHAP Top-{TOP_K}: {sorted(top_set_b)} ({time.time()-t0:.1f}s)")
        print()

    # ── Pairwise Jaccard similarities ──
    print("[4/6] Computing pairwise Jaccard similarities...")

    seed_pairs = list(itertools.combinations(SEEDS, 2))

    jaccard_a = {}
    jaccard_b = {}
    jaccard_vals_a = []
    jaccard_vals_b = []

    for s1, s2 in seed_pairs:
        j_a = jaccard_similarity(top_k_a[s1], top_k_a[s2])
        j_b = jaccard_similarity(top_k_b[s1], top_k_b[s2])
        jaccard_a[f"{s1}_vs_{s2}"] = round(j_a, 4)
        jaccard_b[f"{s1}_vs_{s2}"] = round(j_b, 4)
        jaccard_vals_a.append(j_a)
        jaccard_vals_b.append(j_b)

    mean_jaccard_a = float(np.mean(jaccard_vals_a))
    mean_jaccard_b = float(np.mean(jaccard_vals_b))

    print(f"       Model A mean Jaccard: {mean_jaccard_a:.4f}")
    print(f"       Model B mean Jaccard: {mean_jaccard_b:.4f}")

    if mean_jaccard_b > mean_jaccard_a:
        winner = "Model B has MORE stable SHAP explanations"
    elif mean_jaccard_a > mean_jaccard_b:
        winner = "Model A has MORE stable SHAP explanations"
    else:
        winner = "Both models have EQUAL SHAP stability"
    print(f"       -> {winner}")

    # ── Heatmap plot ──
    print("\n[5/6] Generating stability heatmap...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"SHAP Top-{TOP_K} Feature Stability - Pairwise Jaccard Similarity",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, model_label, top_k_dict, color_map in [
        (axes[0], "Model A (SMOTE)", top_k_a, "Reds"),
        (axes[1], "Model B (Weighted)", top_k_b, "Greens"),
    ]:
        n = len(SEEDS)
        matrix = np.ones((n, n))
        for (i, s1), (j, s2) in itertools.combinations(enumerate(SEEDS), 2):
            j_val = jaccard_similarity(top_k_dict[s1], top_k_dict[s2])
            matrix[i, j] = j_val
            matrix[j, i] = j_val

        sns.heatmap(
            matrix, ax=ax, annot=True, fmt=".2f", cmap=color_map,
            xticklabels=[f"Seed {s}" for s in SEEDS],
            yticklabels=[f"Seed {s}" for s in SEEDS],
            vmin=0, vmax=1,
            linewidths=0.5,
        )
        mean_j = np.mean([matrix[i, j] for i, j in itertools.combinations(range(n), 2)])
        ax.set_title(f"{model_label}\nMean Jaccard = {mean_j:.4f}", fontsize=12)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "shap_stability_heatmap.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"       Heatmap saved to {fig_path}")

    # ── Save results ──
    print("\n[6/6] Saving results...")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Convert sets to sorted lists for JSON serialization
    top_k_a_json = {str(k): sorted(list(v)) for k, v in top_k_a.items()}
    top_k_b_json = {str(k): sorted(list(v)) for k, v in top_k_b.items()}

    shap_rankings_a_json = {
        str(k): [{"feature": f, "mean_abs_shap": round(v, 6)} for f, v in ranking]
        for k, ranking in shap_rankings_a.items()
    }
    shap_rankings_b_json = {
        str(k): [{"feature": f, "mean_abs_shap": round(v, 6)} for f, v in ranking]
        for k, ranking in shap_rankings_b.items()
    }

    report = {
        "phase": "Phase 4, Step 4.3 — Multi-Seed SHAP Stability",
        "seeds": SEEDS,
        "top_k": TOP_K,
        "n_attack_samples": N_ATTACK_SAMPLES,
        "shap_background_k": SHAP_BACKGROUND_K,
        "model_A_SMOTE": {
            "top_k_features_per_seed": top_k_a_json,
            "shap_rankings_per_seed": shap_rankings_a_json,
            "pairwise_jaccard": jaccard_a,
            "mean_jaccard": round(mean_jaccard_a, 4),
        },
        "model_B_weighted": {
            "top_k_features_per_seed": top_k_b_json,
            "shap_rankings_per_seed": shap_rankings_b_json,
            "pairwise_jaccard": jaccard_b,
            "mean_jaccard": round(mean_jaccard_b, 4),
        },
        "interpretation": {
            "result": winner,
            "note": (
                "Jaccard similarity of 1.0 = identical Top-5 feature sets across seeds. "
                "Higher mean Jaccard = more reproducible/stable SHAP explanations."
            ),
        },
    }

    report_path = os.path.join(REPORTS_DIR, "shap_stability.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"       Results saved to {report_path}")

    # Print summary
    print("\n" + "=" * 65)
    print("  Step 4.3 COMPLETE - SHAP Stability Summary")
    print("=" * 65)
    print(f"\n  Model A (SMOTE)    - Mean Jaccard: {mean_jaccard_a:.4f}")
    print(f"  Model B (Weighted) - Mean Jaccard: {mean_jaccard_b:.4f}")
    print(f"  -> {winner}")

    return report


if __name__ == "__main__":
    main()
