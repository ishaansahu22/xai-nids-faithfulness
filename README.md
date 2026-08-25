# Faithful and Calibrated Explainable AI for Network Intrusion Detection

## 📋 How to Execute This Project — Step-by-Step Guide

> **Target:** IEEE / Springer Conference Submission  
> **Type:** A/B Scientific Experiment comparing SMOTE vs Class-Weighting on XAI faithfulness  
> **Dataset:** CSE-CIC-IDS2018 (Kaggle)

---

## 🏗️ Project Directory Structure

After full execution, the project will look like this:

```
xai-nids-faithfulness/
├── PROJECT_STATE.md          ← Master progress tracker (update after EVERY step)
├── README.md                 ← This file
├── requirements.txt          ← Python dependencies
│
├── dataset/                  ← Raw Kaggle CSVs (already downloaded)
│   ├── 02-14-2018.csv        ← FTP/SSH Brute Force (primary)
│   ├── 03-01-2018.csv        ← Infiltration (primary)
│   └── ... (8 other CSVs)
│
├── data/
│   ├── raw/                  ← Copied primary CSVs for the experiment
│   │   ├── 02-14-2018.csv
│   │   └── 03-01-2018.csv
│   └── processed/            ← Clean, split data (output of preprocessing)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── src/                      ← Core experiment scripts
│   ├── preprocess.py         ← Phase 2: Data cleaning & splitting
│   ├── train_model_A.py      ← Phase 3: SMOTE + XGBoost
│   ├── train_model_B.py      ← Phase 3: Class-weighted XGBoost
│   ├── evaluate_shap.py      ← Phase 4: SHAP faithfulness (Deletion/Insertion)
│   └── uncertainty.py        ← Phase 4: MAPIE Conformal Prediction
│
├── models/                   ← Saved model weights
│   ├── model_A_smote.json
│   └── model_B_weighted.json
│
├── reports/                  ← Results, figures, and metrics
│   ├── metrics.json
│   └── figures/
│       ├── confusion_matrix_A.png
│       ├── confusion_matrix_B.png
│       ├── deletion_curves.png
│       ├── insertion_curves.png
│       └── shap_summary.png
│
├── api/                      ← FastAPI live demo endpoint
│   └── main.py
│
└── notebooks/                ← (Optional) Jupyter exploration notebooks
    └── eda.ipynb
```

---

## 🚀 Execution Instructions

### Prerequisites
- **Python 3.10+** installed
- **pip** package manager
- **~7 GB free disk space** (for dataset + processed outputs + models)

---

### PHASE 1: Environment Setup & Data Ingestion

#### Step 1.1 — Install Dependencies
```bash
cd d:\xai-nids-faithfulness
pip install -r requirements.txt
```

#### Step 1.2 — Organize Raw Data
The Kaggle dataset is already downloaded in `dataset/`. We need to copy the two primary CSVs into the experiment's `data/raw/` directory:
```bash
mkdir data\raw
copy dataset\02-14-2018.csv data\raw\
copy dataset\03-01-2018.csv data\raw\
```

#### Step 1.3 — Verify Data Integrity
```bash
python -c "import pandas as pd; df=pd.read_csv('data/raw/02-14-2018.csv'); print(f'02-14: {df.shape}, Labels: {df.Label.value_counts().to_dict()}')"
python -c "import pandas as pd; df=pd.read_csv('data/raw/03-01-2018.csv'); print(f'03-01: {df.shape}, Labels: {df.Label.value_counts().to_dict()}')"
```

> ✅ **Update `PROJECT_STATE.md`** — Mark Phase 1 steps as complete, record row/column counts.

---

### PHASE 2: Data Preprocessing & Leakage Mitigation

#### Step 2.1 — Run the Preprocessing Pipeline
```bash
python src/preprocess.py
```

**What `preprocess.py` does internally:**
1. Loads both raw CSVs and concatenates them
2. Strips whitespace from all column names
3. Drops identifier/leakage columns: `Timestamp` (and `Dst Port` if desired)
4. Casts all numeric columns to `float32`
5. Replaces `inf` / `-inf` → `NaN`, then drops rows with `NaN`
6. Deduplicates identical rows (`drop_duplicates()`)
7. Binarizes the `Label` column: `Benign` → 0, everything else → 1
8. Performs stratified 80/20 train-test split (seed=42)
9. Saves `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` to `data/processed/`

> ⚠️ **Critical Rule:** SMOTE and feature scaling must NEVER touch the test set. They are applied only during model training (Phase 3).

> ✅ **Update `PROJECT_STATE.md`** — Record data shapes, class distributions.

---

### PHASE 3: The A/B ML Experiment

#### Step 3.1 — Train Model A (SMOTE-Balanced)
```bash
python src/train_model_A.py
```
- Loads `data/processed/X_train.csv` and `y_train.csv`
- Applies `SMOTE()` on the training set only
- Trains `XGBClassifier` on the balanced data
- Saves to `models/model_A_smote.json`

#### Step 3.2 — Train Model B (Class-Weighted)
```bash
python src/train_model_B.py
```
- Loads the same training data (NO SMOTE)
- Computes `scale_pos_weight = count(benign) / count(attack)`
- Trains `XGBClassifier` with native cost-sensitive learning
- Saves to `models/model_B_weighted.json`

#### Step 3.3 — Evaluate Both Models
Both training scripts automatically evaluate on the pristine test set and output:
- Confusion Matrix (saved to `reports/figures/`)
- Precision, Recall, F1-Score
- Metrics logged to `reports/metrics.json`

> ✅ **Update `PROJECT_STATE.md`** — Record F1 scores, seed results.

---

### PHASE 4: XAI Faithfulness & Uncertainty Quantification

#### Step 4.1 — SHAP Faithfulness Analysis
```bash
python src/evaluate_shap.py
```
- Initializes `shap.TreeExplainer` for both models
- Uses k-means clustered background dataset (1000 samples) for <50ms latency
- **Deletion Curves:** Masks top-k SHAP features from 500 attack samples, measures probability decay
- **Insertion Curves:** Starts from baseline, inserts top SHAP features, measures probability recovery
- Calculates Area Under Deletion/Insertion Curves (AUC)
- Plots comparative curves → `reports/figures/deletion_curves.png`

> 🔬 **This is the CORE research result.** Lower Deletion AUC = more faithful explanations.

#### Step 4.2 — MAPIE Conformal Prediction
```bash
python src/uncertainty.py
```
- Wraps Model B with `MapieClassifier(method="score")`
- Calibrates on a holdout validation set
- Generates prediction sets at α=0.05 (95% confidence)
- Maps outputs to: `Confident Benign`, `Confident Attack`, `Uncertain/Review`

> ✅ **Update `PROJECT_STATE.md`** — Record Deletion AUCs, MAPIE coverage.

---

### PHASE 5: FastAPI Live Demo

#### Step 5.1 — Start the API Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Step 5.2 — Test the Endpoint
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": [0.1, 0.2, ...]}"
```

**Response format:**
```json
{
  "predicted_class": "Attack",
  "confidence_set": ["Attack"],
  "uncertainty_label": "Confident Attack",
  "top_3_shap_features": [
    {"feature": "Fwd Pkt Len Max", "importance": 0.42},
    {"feature": "Flow Duration", "importance": 0.31},
    {"feature": "Bwd Pkt Len Mean", "importance": 0.18}
  ],
  "latency_ms": 47.2
}
```

---

## 🧪 Multi-Seed Validation

For statistical rigor (required for IEEE), all experiments must be repeated across 5 random seeds:

```python
RANDOM_SEEDS = [42, 101, 202, 303, 404]
```

Report results as **Mean ± Std** for Precision, Recall, F1, and Deletion AUC.

---

## 📊 Expected Key Results (Hypothesis)

| Metric               | Model A (SMOTE)     | Model B (Weighted)  | Winner   |
|----------------------|---------------------|---------------------|----------|
| F1-Score             | ~0.95               | ~0.93               | Model A  |
| Deletion AUC ↓       | Higher (worse)      | **Lower (better)**  | Model B  |
| Insertion AUC ↑      | Lower (worse)       | **Higher (better)** | Model B  |
| SHAP Stability       | Lower Jaccard       | **Higher Jaccard**  | Model B  |
| MAPIE 95% Coverage   | N/A                 | ≥ 0.95              | Model B  |

**Hypothesis:** Model A achieves marginally better classification accuracy due to balanced training data, but Model B produces **more faithful and stable explanations** because it never trains on synthetic (fabricated) data points.

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `MemoryError` during SMOTE | Reduce training data or use `SMOTE(sampling_strategy=0.5)` |
| SHAP latency > 50ms | Reduce `SHAP_BACKGROUND_K` from 1000 to 500 |
| `inf` values in CSVs | Already handled by `preprocess.py` |
| Permission denied on D:\ | Run terminal as Administrator or work in Downloads |

---

## 📝 Important Notes

1. **Never apply SMOTE or scaling on the test set** — this introduces data leakage
2. **Always update `PROJECT_STATE.md`** after completing each step
3. **The Kaggle dataset columns do NOT include** `Flow ID`, `Src IP`, or `Dst IP` — these were already removed by the Kaggle uploader
4. **The `Timestamp` column IS present** and must be dropped during preprocessing
5. **`Dst Port`** is a borderline feature — it carries behavioral information but can also cause memorization. The default approach keeps it, but this is a design decision worth documenting
