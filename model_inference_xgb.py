from __future__ import annotations

import json
import os
import re
import time
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------- Config ----------------
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "models/experiments_xgb_optuna/artifacts"))
DEMOGRAPHICS_CSV = Path(os.getenv("DEMOGRAPHICS_CSV", "data/zipcode_demographics.csv"))
LOG_TARGET = True  # model trained on log1p(price) -> expm1 at inference


# ---------------- Schemas ----------------
class BasicHouseFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: Union[int, str]


class PredictOneResponse(BaseModel):
    prediction: float
    metadata: dict


class PredictBatchRequest(BaseModel):
    items: List[BasicHouseFeatures] = Field(..., description="Batch of rows to score")


class PredictBatchResponse(BaseModel):
    predictions: List[float]
    metadata: dict


# --------------- App & Lifespan ---------------
app = FastAPI(title="House Price Inference API (XGB + Optuna)", version="1.0.0")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load demographics once (scalable)
    if not DEMOGRAPHICS_CSV.exists():
        raise RuntimeError(f"Demographics CSV not found at {DEMOGRAPHICS_CSV}")
    app.state.demographics = pd.read_csv(DEMOGRAPHICS_CSV, dtype={"zipcode": str})
    # Pre-compute medians for robust single-item fallback on unseen zipcodes
    app.state.demo_medians = (
        app.state.demographics.drop(columns=["zipcode"], errors="ignore")
        .median(numeric_only=True)
        .to_dict()
    )
    yield
    # nothing to clean up

app.router.lifespan_context = lifespan


# --------------- Artifacts Loading ---------------
def _latest_stamp(art_dir: Path) -> str:
    """Find the latest numeric stamp shared by model & feature files."""
    pkl = {m.group(1) for p in art_dir.glob("xgb_final_*.pkl")
           if (m := re.match(r"xgb_final_(\d+)\.pkl$", p.name))}
    feats = {m.group(1) for p in art_dir.glob("model_features_*.json")
             if (m := re.match(r"model_features_(\d+)\.json$", p.name))}
    common = sorted(pkl & feats, key=int)
    if not common:
        raise HTTPException(
            status_code=500,
            detail=f"No matching model/features pair in {art_dir} "
                   f"(models={sorted(pkl)}, features={sorted(feats)})"
        )
    return common[-1]


def _load_bundle(art_dir: Path, stamp: str | None = None) -> tuple[object, list[str], str, str, str]:
    """
    Returns: (model, expected_features, stamp, model_path, features_path)
    Loads fresh each request so you can drop new artifacts without restarting.
    """
    st = stamp or _latest_stamp(art_dir)
    model_path = art_dir / f"xgb_final_{st}.pkl"
    feats_path = art_dir / f"model_features_{st}.json"

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(feats_path, "r", encoding="utf-8") as f:
            expected = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load artifacts: {e}")

    return model, expected, st, str(model_path), str(feats_path)


# --------------- Feature Engineering (same as training) ---------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rooms"] = out["bedrooms"] + out["bathrooms"]
    bed_nonzero = out["bedrooms"].replace(0, np.nan)
    out["baths_per_bed"] = (out["bathrooms"] / bed_nonzero).fillna(out["bathrooms"])
    out["bed_bath_x"] = out["bedrooms"] * out["bathrooms"]
    out["lot_coverage"] = (out["sqft_living"] / (out["sqft_lot"] + 1.0)).clip(0, 1.0)
    out["is_basement"] = (out["sqft_basement"] > 0).astype(int)
    out["basement_ratio"] = out["sqft_basement"] / (out["sqft_living"] + 1.0)
    out["above_ratio"] = out["sqft_above"] / (out["sqft_living"] + 1.0)
    out["sqft_living_log"] = np.log1p(out["sqft_living"])
    out["sqft_lot_log"] = np.log1p(out["sqft_lot"])
    out["bathrooms_sq"] = out["bathrooms"] ** 2
    out["floors_sq"] = out["floors"] ** 2
    for col in ["median_income", "population", "pop_density", "median_age", "household_size"]:
        if col in out.columns:
            out[f"{col}_log"] = np.log1p(out[col])
    return out


# --------------- Prep Helpers ---------------
def _merge_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Merge zip demographics; fill missing demo cols with medians; drop zipcode."""
    demo = app.state.demographics
    meds = app.state.demo_medians

    df = df.copy()
    df["zipcode"] = df["zipcode"].astype(str)
    merged = df.merge(demo, how="left", on="zipcode")

    for col, med in meds.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(med)

    return merged.drop(columns=["zipcode"], errors="ignore")


def _align_columns(X: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    for c in expected_cols:
        if c not in X.columns:
            X[c] = 0.0
    return X[expected_cols]


# --------------- Routes ---------------
@app.get("/health_check")
def health_check():
    try:
        mdl, feats, st, mpath, fpath = _load_bundle(ARTIFACTS_DIR)
        _ = app.state.demographics  # ensure loaded
    except HTTPException as e:
        return {"status": "degraded", "detail": e.detail}
    except Exception as e:
        return {"status": "degraded", "detail": str(e)}
    return {"status": "ok", "stamp": st, "model_path": mpath, "features_path": fpath,
            "demographics_source": str(DEMOGRAPHICS_CSV)}


@app.post("/predict_one", response_model=PredictOneResponse)
def predict_one(item: BasicHouseFeatures):
    model, expected, stamp, model_path, feats_path = _load_bundle(ARTIFACTS_DIR)

    base = pd.DataFrame([item.model_dump()])
    merged = _merge_demographics(base)
    engineered = add_features(merged)
    X = _align_columns(engineered, expected)

    t0 = time.time()
    try:
        pred = float(model.predict(X)[0])
        if LOG_TARGET:
            pred = float(np.expm1(pred))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference failed: {e}")

    latency_ms = int((time.time() - t0) * 1000)
    return PredictOneResponse(
        prediction=pred,
        metadata={
            "model_stamp": stamp,
            "model_path": model_path,
            "features_path": feats_path,
            "latency_ms": latency_ms,
            "demographics_source": str(DEMOGRAPHICS_CSV),
        },
    )


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="No items provided.")

    model, expected, stamp, model_path, feats_path = _load_bundle(ARTIFACTS_DIR)

    df = pd.DataFrame([it.model_dump() for it in req.items])
    merged = _merge_demographics(df)
    engineered = add_features(merged)
    X = _align_columns(engineered, expected)

    t0 = time.time()
    try:
        preds = model.predict(X)
        preds = np.expm1(preds) if LOG_TARGET else preds
        preds = [float(x) for x in preds]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference failed: {e}")

    latency_ms = int((time.time() - t0) * 1000)
    return PredictBatchResponse(
        predictions=preds,
        metadata={
            "n_inputs": len(preds),
            "model_stamp": stamp,
            "model_path": model_path,
            "features_path": feats_path,
            "latency_ms": latency_ms,
            "demographics_source": str(DEMOGRAPHICS_CSV),
        },
    )