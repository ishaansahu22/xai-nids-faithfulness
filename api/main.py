"""
Phase 5, Step 5.1: FastAPI Triage Endpoint
===========================================
Serves Model B predictions with MAPIE uncertainty and Top-3 SHAP
feature attributions via a REST API.

Usage:
    uvicorn api.main:app --reload --port 8000

Endpoint:
    POST /predict
    Body: {"features": [78 float values]}
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from contextlib import asynccontextmanager
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from mapie.classification import SplitConformalClassifier

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RANDOM_STATE = 42
MODEL_DIR = "models"
PROCESSED_DIR = os.path.join("data", "processed")
ALPHA = 0.05
CALIBRATION_SIZE = 0.20
SHAP_BACKGROUND_K = 100  # Smaller for API latency (< 100ms target)
TRAIN_SAMPLE_SIZE = 1000  # Smaller sample for fast startup


# ──────────────────────────────────────────────
# Global state (loaded once at startup)
# ──────────────────────────────────────────────
class AppState:
    model: XGBClassifier = None
    mapie: SplitConformalClassifier = None
    explainer: shap.TreeExplainer = None
    feature_names: list = None


state = AppState()


# ──────────────────────────────────────────────
# Startup / Shutdown
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, MAPIE, and SHAP explainer once at startup."""
    print("=" * 55)
    print("  SentinelIQ API — Starting up...")
    print("=" * 55)

    # 1. Load Model B
    print("[1/4] Loading Model B...")
    model = XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "model_B_weighted.json"))
    state.model = model
    print("       Model B loaded")

    # 2. Load feature names from X_test header
    print("[2/4] Loading feature names...")
    X_test_header = pd.read_csv(
        os.path.join(PROCESSED_DIR, "X_test.csv"), nrows=0
    )
    state.feature_names = list(X_test_header.columns)
    print(f"       {len(state.feature_names)} features loaded")

    # 3. Initialize MAPIE with calibration data
    print("[3/4] Calibrating MAPIE...")
    X_train = pd.read_csv(
        os.path.join(PROCESSED_DIR, "X_train.csv"),
        nrows=50000,  # Sufficient for calibration
    )
    y_train = pd.read_csv(
        os.path.join(PROCESSED_DIR, "y_train.csv"),
        nrows=50000,
    ).squeeze()

    _, X_cal, _, y_cal = train_test_split(
        X_train, y_train,
        test_size=CALIBRATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    mapie = SplitConformalClassifier(
        estimator=model,
        confidence_level=1 - ALPHA,
        prefit=True,
        random_state=RANDOM_STATE,
    )
    mapie.conformalize(X_cal, y_cal)
    state.mapie = mapie
    print("       MAPIE calibrated")

    # 4. Initialize SHAP TreeExplainer
    print("[4/4] Initializing SHAP TreeExplainer...")
    X_bg = pd.read_csv(
        os.path.join(PROCESSED_DIR, "X_train.csv"),
        nrows=TRAIN_SAMPLE_SIZE,
    )
    background = shap.maskers.Independent(X_bg, max_samples=SHAP_BACKGROUND_K)
    state.explainer = shap.TreeExplainer(model, background)
    print("       SHAP explainer ready")

    print("\n" + "=" * 55)
    print("  ✅ API Ready — POST /predict")
    print("=" * 55 + "\n")

    yield  # App runs here

    print("Shutting down...")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="SentinelIQ — XAI Network Intrusion Detection",
    description=(
        "Explainable AI-powered NIDS with SHAP feature attributions "
        "and MAPIE conformal prediction uncertainty."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=78,
        max_length=78,
        description="78 float feature values in the same order as training data",
    )


class ShapFeature(BaseModel):
    feature: str
    importance: float


class PredictResponse(BaseModel):
    predicted_class: str
    confidence_set: List[str]
    uncertainty_label: str
    top_3_shap_features: List[ShapFeature]
    latency_ms: float


# ──────────────────────────────────────────────
# Predict endpoint
# ──────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Classify a network flow and return:
    - Predicted class (Benign / Attack)
    - MAPIE confidence set with uncertainty label
    - Top-3 SHAP feature attributions
    - End-to-end latency in ms
    """
    t0 = time.time()

    try:
        # Build input DataFrame
        X_input = pd.DataFrame(
            [request.features],
            columns=state.feature_names,
        )

        # 1. Point prediction
        y_pred = state.model.predict(X_input)[0]
        predicted_class = "Attack" if y_pred == 1 else "Benign"

        # 2. MAPIE prediction set
        _, y_sets = state.mapie.predict_set(X_input)
        pred_set = y_sets[0, :, 0]  # shape (2,)

        confidence_set = []
        if pred_set[0]:
            confidence_set.append("Benign")
        if pred_set[1]:
            confidence_set.append("Attack")

        # Map to triage label
        if confidence_set == ["Benign"]:
            uncertainty_label = "Confident Benign"
        elif confidence_set == ["Attack"]:
            uncertainty_label = "Confident Attack"
        elif set(confidence_set) == {"Benign", "Attack"}:
            uncertainty_label = "Uncertain/Review"
        else:
            uncertainty_label = "Empty Set"

        # 3. SHAP explanations
        explanation = state.explainer(X_input)
        sv = explanation.values.flatten()
        # For binary classification with 2D output, take positive class
        if sv.ndim > 0 and len(sv) == 2 * len(state.feature_names):
            sv = sv[len(state.feature_names):]  # second class (Attack)
        elif hasattr(explanation, 'values') and explanation.values.ndim == 3:
            sv = explanation.values[0, :, 1]  # sample 0, all features, class 1

        # Top-3 by absolute importance
        top3_idx = np.argsort(-np.abs(sv))[:3]
        top_3_shap = [
            ShapFeature(
                feature=state.feature_names[idx],
                importance=round(float(sv[idx]), 4),
            )
            for idx in top3_idx
        ]

        latency_ms = round((time.time() - t0) * 1000, 1)

        return PredictResponse(
            predicted_class=predicted_class,
            confidence_set=confidence_set,
            uncertainty_label=uncertainty_label,
            top_3_shap_features=top_3_shap,
            latency_ms=latency_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": state.model is not None,
        "mapie_loaded": state.mapie is not None,
        "shap_loaded": state.explainer is not None,
        "n_features": len(state.feature_names) if state.feature_names else 0,
    }


@app.get("/")
async def root():
    return {
        "service": "SentinelIQ — XAI Network Intrusion Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Classify a network flow with SHAP + MAPIE",
            "GET /health": "Health check",
        },
    }
