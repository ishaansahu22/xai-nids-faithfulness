# PROJECT STATE & RECOVERY LEDGER

## 1. Project Metadata
- **Project Title:** Faithful and Calibrated Explainable AI for Network Intrusion Detection
- **Target Venue:** IEEE / Springer Conference
- **Active Environment:** Python 3.10+, VS Code
- **Global Random State:** 42
- **Dataset:** CSE-CIC-IDS2018 (Kaggle: `solarmainframe/ids-intrusion-csv`)

## 2. Current Status
- **Current Phase:** Phase 4 (XAI Faithfulness & Uncertainty Quantification)
- **Active Step:** Step 4.1 (TreeSHAP + Deletion/Insertion Curves)
- **Status:** IN PROGRESS
- **Last Updated:** 2026-08-26 00:05 IST

## 3. Execution Checklist & Log

- [x] **Phase 1: Environment & Raw Data Acquisition** ✅
  - [x] 1.1 Generate `requirements.txt` and install dependencies (120+ packages)
  - [x] 1.2 Download raw CSE-CIC-IDS2018 CSVs (`02-14-2018.csv`, `03-01-2018.csv`)
  - [x] 1.3 Organize into `data/raw/` and verify integrity

- [x] **Phase 2: Data Preprocessing & Leakage Mitigation** ✅
  - [x] 2.1 Strip whitespaces & drop `Timestamp` column (1 identifier removed)
  - [x] 2.2 Replaced 9,375 inf values; dropped 6,768 NaN rows (0.49%)
  - [x] 2.3 Removed 429,886 duplicate rows (31.31%)
  - [x] 2.4 Binarized labels: Benign=766,414 | Attack=176,632 (ratio 4.34:1)
  - [x] 2.5 Stratified 80/20 split → train=754,436 | test=188,610

- [x] **Phase 3: The A/B ML Experiment** ✅
  - [x] 3.1 Model A (SMOTE): Acc=0.9199, P=0.8178, R=0.7364, **F1=0.7750**
  - [x] 3.2 Model B (Weighted): Acc=0.8964, P=0.6944, R=0.7979, **F1=0.7426**
  - [x] 3.3 Confusion matrices & metrics saved to `reports/`
  - [ ] 3.4 5-Seed statistical validation (deferred to after Phase 4)

- [/] **Phase 4: XAI Faithfulness & Uncertainty Quantification** ⏳
  - [x] 4.1 `src/evaluate_shap.py` — TreeSHAP with k-means background (1000 samples, <50ms)
  - [x] 4.2 Deletion/Insertion AUC curves (500 test attack samples, Model A vs Model B)
  - [ ] 4.3 Multi-seed SHAP stability (Jaccard similarity of Top-5 features)
  - [ ] 4.4 `src/uncertainty.py` — MAPIE Conformal Prediction (α=0.05, 95% coverage)

- [ ] **Phase 5: Fast Triage REST API & Presentation**
  - [ ] 5.1 `api/main.py` — FastAPI `/predict` endpoint
  - [ ] 5.2 Response: predicted class + MAPIE confidence set + Top-3 SHAP attributions (<100ms)

## 4. Hyperparameter & Artifact Registry
| Parameter               | Value                                      |
|-------------------------|--------------------------------------------|
| `RANDOM_SEEDS`          | `[42, 101, 202, 303, 404]`                |
| `PRIMARY_SEED`          | `42`                                       |
| `RAW_DATA_PATH`         | `data/raw/`                                |
| `PROCESSED_DATA_PATH`   | `data/processed/`                          |
| `MODEL_REGISTRY`        | `models/`                                  |
| `METRICS_OUTPUT`        | `reports/metrics.json`                     |
| `FIGURES_OUTPUT`        | `reports/figures/`                          |
| `TEST_SPLIT_RATIO`      | `0.20`                                     |
| `SMOTE_STRATEGY`        | `auto` (default)                           |
| `MAPIE_ALPHA`           | `0.05`                                     |
| `SHAP_BACKGROUND_K`     | `1000`                                     |
| `DELETION_SAMPLE_SIZE`  | `500`                                      |
| `XGBOOST_PARAMS`        | n_est=200, depth=6, lr=0.1, sub=0.8, col=0.8 |
| `SCALE_POS_WEIGHT`      | `4.3391` (Model B only)                    |
| `SMOTE_SYNTHETIC_COUNT`  | `471,826` samples generated                |
| `MODEL_A_F1`            | `0.7750`                                   |
| `MODEL_B_F1`            | `0.7426`                                   |

## 5. Data Shape Registry
| Artifact          | Shape           | Size     | Notes                           |
|-------------------|-----------------|----------|---------------------------------|
| Raw 02-14-2018    | (1048575, 80)   | 341 MB   | FTP/SSH Brute Force             |
| Raw 03-01-2018    | (331125, 80)    | 103 MB   | Infiltration                    |
| Combined raw      | (1379700, 80)   | —        | Before cleaning                 |
| After cleaning    | (943046, 79)    | —        | inf/NaN/dedup removed           |
| X_train           | (754436, 78)    | 314.8 MB | 78 features, float32            |
| X_test            | (188610, 78)    | 78.7 MB  | 78 features, float32            |
| y_train           | (754436,)       | 2.2 MB   | Attack ratio: 18.73%            |
| y_test            | (188610,)       | 0.5 MB   | Attack ratio: 18.73%            |

## 6. Handover Notes
- **Dataset Location:** `d:\xai-nids-faithfulness\dataset\` (full Kaggle download, 10 CSVs)
- **Primary CSVs for this study:** `02-14-2018.csv` (Brute Force) and `03-01-2018.csv` (Infiltration)
- **Column count after cleaning:** 78 features + 1 binary Label
- **Columns dropped:** `Timestamp` (only 1 — Kaggle version already removed `Flow ID`, `Src IP`, `Dst IP`)
- **`Dst Port` KEPT:** It carries genuine behavioral signal (port 22=SSH, 21=FTP), not an identifier
- **Label distribution:** Benign=81.27%, SSH-Bruteforce=9.97%, Infilteration=8.75%, FTP-BruteForce=0.01%
- **Imbalance ratio:** 4.34 Benign per 1 Attack (moderate imbalance)
- **Virtual env:** `d:\xai-nids-faithfulness\venv\` — activate with `venv\Scripts\activate`
- **SettingWithCopyWarning** in preprocess.py line 129 is cosmetic and does not affect results
- When moving between systems, activate the venv, verify this file, and execute the next unchecked item in Section 3.
