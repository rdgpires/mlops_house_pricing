from __future__ import annotations

import json
import os
import pathlib
import pickle
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple, Dict, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Config ----
MODEL_FAMILY = "house-price"
MODELS_ROOT = pathlib.Path("models") / MODEL_FAMILY         # e.g., models/house-price
LATEST_MARKER = MODELS_ROOT / "latest"                      # contains 'vN'
DEMOGRAPHICS_CSV = pathlib.Path(os.getenv("DEMOGRAPHICS_CSV", "data/zipcode_demographics.csv"))

# --- Schemas to match future_unseen_examples.csv ---
class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: Union[int, str] # we will coerce to str and join demographics
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

class PredictRequest(BaseModel):
    items: List[HouseFeatures] = Field(..., description="Batch of minimal rows to make predictions, base in the future_unseen_examples.csv")

class PredictResponse(BaseModel):
    predictions: List[float]
    metadata: dict

# --- Minimal input schema (basic model subset from house sales) ---
class BasicHouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: Union[int, str]  # we will coerce to str and join demographics

class PredictBasicRequest(BaseModel):
    items: List[BasicHouseFeatures] = Field(..., description="Batch of minimal rows to score")

# --- Evaluation schemas (basic features + ground-truth) ---
class BasicHouseWithPrice(BasicHouseFeatures):
    price: float

class EvaluateBasicRequest(BaseModel):
    items: List[BasicHouseWithPrice] = Field(
        ..., description="Batch with BasicHouseFeatures + ground-truth 'price'."
    )

class EvaluateResponse(BaseModel):
    predictions: List[float]
    y_true: List[float]
    per_item: List[dict]
    metrics: dict
    metadata: dict


# --- App with lifespan: load demographics once to guarantee scalability ---
app = FastAPI(title="House Price Inference API (latest-by-request)", version="1.0.0")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load demographics once
    if not DEMOGRAPHICS_CSV.exists():
        raise RuntimeError(f"Demographics CSV not found at {DEMOGRAPHICS_CSV}")
    app.state.demographics = pd.read_csv(DEMOGRAPHICS_CSV, dtype={"zipcode": str})
    yield
    # Shutdown: nothing to clean up

app.router.lifespan_context = lifespan

# --- Model loading (read 'latest' marker and artifacts each request to guaratee updates without stopping the service) ---
def _resolve_latest_dir() -> pathlib.Path:
    version = LATEST_MARKER.read_text(encoding="utf-8").strip()
    model_dir = MODELS_ROOT / version
    return model_dir

def _load_latest_bundle() -> Tuple[object, List[str], str, str, str]:
    """
    Returns: (model, expected_features, version, model_path_str, features_path_str)
    Loads from disk on each call to pick up new 'latest' without redeploy.
    """
    model_dir = _resolve_latest_dir()
    model_pkl = model_dir / "model.pkl"
    features_json = model_dir / "model_features.json"

    try:
        with open(model_pkl, "rb") as f:
            model = pickle.load(f)
        with open(features_json, "r", encoding="utf-8") as f:
            expected_features = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model artifacts: {e}")

    return model, expected_features, model_dir.name, str(model_pkl), str(features_json)

# --- Helpers ---
def _prepare_frame(items: List[HouseFeatures], expected_features: List[str]) -> tuple[pd.DataFrame, List[str]]:
    df = pd.DataFrame([i.model_dump() for i in items])

    # Ensure zipcode is string for join
    if "zipcode" in df.columns:
        df["zipcode"] = df["zipcode"].astype(str)

    # Server-side demographics merge, then drop zipcode
    df = df.merge(app.state.demographics, how="left", on="zipcode").drop(columns="zipcode")

    # Align columns to what the model expects
    missing = [c for c in expected_features if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    df = df[expected_features]  # order & drop extras columns
    return df, missing

def _mape_pct(y_true, y_pred) -> Optional[float]:
    """
    Mean Absolute Percentage Error in percent.
    Ignores rows where y_true is 0 (or extremely close) to avoid division blow-ups.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.abs(yt) > 1e-9
    if not np.any(mask):
        return None
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)

def _compute_metrics(y_true, y_pred_list):
    mae = float(mean_absolute_error(y_true, y_pred_list))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_list)))
    r2 = float(r2_score(y_true, y_pred_list)) if len(y_true) >= 2 else None
    mape = _mape_pct(y_true, y_pred_list)
    return {"rmse": rmse, "mae": mae, "mape_pct": mape, "r2": r2, "n_items": len(y_true)}

# --- Routes ---
@app.get("/health_check")
def health_check():
    try:
        # Checks inference endpoint health by cheking if latest model can be loaded and if demographics is loaded 
        mdl_dir = _resolve_latest_dir()
        _ = app.state.demographics
    except HTTPException as e:
        return {"status": "degraded", "detail": e.detail}
    except AttributeError:
        return {"status": "degraded", "detail": "Demographics not loaded"}
    return {"status": "ok", "latest_dir": str(mdl_dir)}

# predict route accepts columns present in match future_unseen_examples.csv and merges demographics before predicting
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="No items provided.")

    # Load current latest model bundle on each request
    model, expected_features, version, model_path, features_path = _load_latest_bundle()

    t0 = time.time()
    X, missing = _prepare_frame(req.items, expected_features)

    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference failed: {e}")

    latency_ms = int((time.time() - t0) * 1000)
    return PredictResponse(
        predictions=[float(x) for x in preds],
        metadata={
            "n_inputs": len(req.items),
            "model_version": version,
            "expected_features_path": features_path,
            "missing_features_added_as_nan": missing,
            "model_path": model_path,
            "latency_ms": latency_ms,
            "demographics_source": str(DEMOGRAPHICS_CSV),
        },
    )

# predict_basic route accepts only the minimal fields and still merges demographics before predicting.
@app.post("/predict_basic", response_model=PredictResponse)
def predict_basic(req: PredictBasicRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="No items provided.")

    # Load the current latest model and its expected feature list
    model, expected_features, version, model_path, features_path = _load_latest_bundle()

    # Build DF from the minimal inputs
    df = pd.DataFrame([i.model_dump() for i in req.items])
    df["zipcode"] = df["zipcode"].astype(str)

    # Merge demographics server-side and align columns the model expects
    demo = app.state.demographics
    df = df.merge(demo, how="left", on="zipcode").drop(columns="zipcode")

    # Align to expected (missing cols become NaN; the pipeline's SimpleImputer will handle them)
    missing = [c for c in expected_features if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    X = df[expected_features]

    t0 = time.time()
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference failed: {e}")
    latency_ms = int((time.time() - t0) * 1000)

    return PredictResponse(
        predictions=[float(x) for x in preds],
        metadata={
            "n_inputs": len(req.items),
            "required_features_only": True,
            "model_version": version,
            "expected_features_path": features_path,
            "missing_features_added_as_nan": missing,
            "model_path": model_path,
            "latency_ms": latency_ms,
            "demographics_source": str(DEMOGRAPHICS_CSV),
        },
    )

# --- NEW: evaluation route (predictions + aggregate metrics over ground truth) ---
@app.post("/evaluate")
def evaluate_basic(req: EvaluateBasicRequest) -> Dict[str, Any]:
    """
    Accepts items with only the variables used in training (BasicHouseFeatures) + 'price'.
    Returns per-item predictions and aggregate RMSE/MAE/MAPE/R2.
    """
    if not req.items:
        raise HTTPException(status_code=400, detail="No items provided.")

    # Load model bundle
    model, expected_features, version, model_path, features_path = _load_latest_bundle()

    # y_true + features
    y_true = [float(it.price) for it in req.items]
    df = pd.DataFrame([{
        "bedrooms": it.bedrooms,
        "bathrooms": it.bathrooms,
        "sqft_living": it.sqft_living,
        "sqft_lot": it.sqft_lot,
        "floors": it.floors,
        "sqft_above": it.sqft_above,
        "sqft_basement": it.sqft_basement,
        "zipcode": str(it.zipcode),
    } for it in req.items])

    # Merge demographics & align features (same as /predict_basic)
    demo = app.state.demographics
    df = df.merge(demo, how="left", on="zipcode").drop(columns="zipcode")

    missing = [c for c in expected_features if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    X = df[expected_features]

    t0 = time.time()
    try:
        y_pred = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference failed during evaluation: {e}")
    latency_ms = int((time.time() - t0) * 1000)

    y_pred_list = [float(x) for x in y_pred]
    metrics_dict = _compute_metrics(y_true, y_pred_list)

    per_item = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred_list)):
        # Safe percentage error (None if yt ~ 0)
        pct_err = (abs(yt - yp) / yt * 100.0) if abs(yt) > 1e-9 else None
        per_item.append({
            "index": i,
            "y_true": yt,
            "y_pred": yp,
            "abs_error": float(abs(yt - yp)),
            "squared_error": float((yt - yp) ** 2),
            "abs_pct_error": float(pct_err) if pct_err is not None else None
        })

    return {
        "predictions": y_pred_list,
        "y_true": [float(v) for v in y_true],
        "per_item": per_item,
        "metrics": metrics_dict,   # includes mape_pct
        "metadata": {
            "n_inputs": len(req.items),
            "required_features_only": True,
            "model_version": version,
            "expected_features_path": features_path,
            "missing_features_added_as_nan": missing,
            "model_path": model_path,
            "latency_ms": latency_ms,
            "demographics_source": str(DEMOGRAPHICS_CSV),
        },
    }