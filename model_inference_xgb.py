from __future__ import annotations

import json
import os
import re
import time
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Tuple, Union, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

class EvalItem(BasicHouseFeatures):
    price: float

class EvaluateRequest(BaseModel):
    items: List[EvalItem]

class EvaluateResponse(BaseModel):
    predictions: List[float]
    y_true: List[float]
    metrics: dict
    per_item: List[dict]
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


def _safe_log1p(s: pd.Series) -> pd.Series:
    # evita negativos/NaN; funciona mesmo com valores faltantes
    return np.log1p(s.clip(lower=0).fillna(0.0))


def _as_fraction(s: pd.Series) -> pd.Series:
    """Converte percentuais em frações 0–1; assume que valores podem estar em 0–100 ou 0–1; faz clamp."""
    s = s.astype(float)
    # heurística: se a mediana > 1.5, provavelmente está em porcentagem 0–100
    median = s.replace([np.inf, -np.inf], np.nan).dropna().median() if len(s) else 0
    if pd.notna(median) and median > 1.5:
        s = s / 100.0
    return s.clip(lower=0.0, upper=1.0).fillna(0.0)


def _entropy_from_parts(parts: pd.DataFrame) -> pd.Series:
    """Entropia normalizada (0–1) de distribuição de escolaridade."""
    vals = parts.fillna(0.0).astype(float)
    total = vals.sum(axis=1)
    # evita divisão por zero
    frac = vals.div(total.replace(0, np.nan), axis=0).fillna(0.0)
    # entropia de Shannon
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(frac * np.log(frac.where(frac > 0, 1.0))).sum(axis=1)
    # normaliza por log(k) com k = número de categorias
    k = vals.shape[1] if vals.shape[1] > 0 else 1
    ent_norm = ent / np.log(k)
    return ent_norm.clip(lower=0.0, upper=1.0).fillna(0.0)

def add_demo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering para variáveis demográficas do arquivo anexado.
    Opera defensivamente: só cria o que existir nas colunas.
    """
    out = df.copy()

    # Nomes esperados (do CSV anexado)
    qty_cols = [
        "ppltn_qty", "urbn_ppltn_qty", "sbrbn_ppltn_qty", "farm_ppltn_qty",
        "non_farm_qty",
        "edctn_less_than_9_qty", "edctn_9_12_qty", "edctn_high_schl_qty",
        "edctn_some_clg_qty", "edctn_assoc_dgre_qty",
        "edctn_bchlr_dgre_qty", "edctn_prfsnl_qty",
    ]
    per_cols = [
        "per_urbn", "per_sbrbn", "per_farm", "per_non_farm",
        "per_less_than_9", "per_9_to_12", "per_hsd",
        "per_some_clg", "per_assoc", "per_bchlr", "per_prfsnl",
    ]
    money_cols = ["medn_hshld_incm_amt", "medn_incm_per_prsn_amt", "hous_val_amt"]

    # 1) Logs para escalas assimétricas (população, rendas, valor de imóvel)
    for c in qty_cols + money_cols:
        if c in out.columns:
            out[f"{c}_log"] = _safe_log1p(out[c])

    # 2) Percentuais como frações 0–1 + clamp
    for c in per_cols:
        if c in out.columns:
            out[f"{c}_frac"] = _as_fraction(out[c])

    # 3) Razões renda <-> valor de imóvel
    if "hous_val_amt" in out.columns and "medn_hshld_incm_amt" in out.columns:
        out["house_value_to_income"] = out["hous_val_amt"] / (out["medn_hshld_incm_amt"] + 1.0)
        out["income_to_house_value"] = out["medn_hshld_incm_amt"] / (out["hous_val_amt"] + 1.0)

    # 4) Índices de escolaridade
    #    - participação de ensino superior (bachelor + professional)
    high_edu_parts = []
    for c in ["per_bchlr", "per_prfsnl"]:
        if c in out.columns:
            high_edu_parts.append(f"{c}_frac")
            if f"{c}_frac" not in out.columns:  # garante existência se ainda não criado
                out[f"{c}_frac"] = _as_fraction(out[c])
    if high_edu_parts:
        out["share_high_edu"] = out[high_edu_parts].sum(axis=1).clip(0.0, 1.0)

    #    - entropia da distribuição de escolaridade (usando as quantidades se existirem)
    edu_qty_present = [c for c in qty_cols if c.startswith("edctn_") and c in out.columns]
    if edu_qty_present:
        out["edu_entropy"] = _entropy_from_parts(out[edu_qty_present])

    # 5) Urbanização
    #    - diferença entre urbano e rural (faz sentido como sinal de acesso/infra)
    if "per_urbn" in out.columns:
        if "per_farm" in out.columns:
            out["urbanization_idx"] = _as_fraction(out["per_urbn"]) - _as_fraction(out["per_farm"])
        else:
            out["urbanization_idx"] = _as_fraction(out["per_urbn"])
    elif "urbn_ppltn_qty" in out.columns and "ppltn_qty" in out.columns:
        out["urbanization_idx"] = (out["urbn_ppltn_qty"] / (out["ppltn_qty"] + 1.0)).clip(0.0, 1.0)

    # 6) Non-farm share se existir apenas como quantidade
    if "non_farm_qty" in out.columns and "ppltn_qty" in out.columns and "per_non_farm_frac" not in out.columns:
        out["per_non_farm_frac"] = (out["non_farm_qty"] / (out["ppltn_qty"] + 1.0)).clip(0.0, 1.0)

    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering seguro, sem vazamento de alvo.
    Inclui engenharia dos atributos 'structural' da casa e dos demográficos.
    """
    out = df.copy()

    # ---------- 1) Casa: combinações e razões simples ----------
    out["rooms"] = out["bedrooms"] + out["bathrooms"]
    bed_nonzero = out["bedrooms"].replace(0, np.nan)
    out["baths_per_bed"] = (out["bathrooms"] / bed_nonzero).fillna(out["bathrooms"])
    out["bed_bath_x"] = out["bedrooms"] * out["bathrooms"]
    out["lot_coverage"] = (out["sqft_living"] / (out["sqft_lot"] + 1.0)).clip(0, 1.0)

    # ---------- 2) Estrutura acima/baixo ----------
    out["is_basement"] = (out["sqft_basement"] > 0).astype(int)
    out["basement_ratio"] = out["sqft_basement"] / (out["sqft_living"] + 1.0)
    out["above_ratio"] = out["sqft_above"] / (out["sqft_living"] + 1.0)

    # ---------- 3) Logs para escalas tendenciosas ----------
    out["sqft_living_log"] = np.log1p(out["sqft_living"])
    out["sqft_lot_log"] = np.log1p(out["sqft_lot"])

    # ---------- 4) Não linearidades leves ----------
    out["bathrooms_sq"] = out["bathrooms"] ** 2
    out["floors_sq"] = out["floors"] ** 2

    # ---------- 5) Demografia (engenharia dedicada) ----------
    out = add_demo_features(out)

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

def _mape_pct(y_true, y_pred) -> Optional[float]:
    """Mean Absolute Percentage Error in percent. Skips near-zero truths."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.abs(yt) > 1e-9
    if not np.any(mask):
        return None
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)

def _compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    mape = _mape_pct(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "mape_pct": mape, "r2": r2, "n_items": len(y_true)}

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

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="No items provided.")

    # Load current artifacts
    model, expected, stamp, model_path, feats_path = _load_bundle(ARTIFACTS_DIR)

    # Build dataframe from request
    base = pd.DataFrame([it.model_dump() for it in req.items])
    y_true = base.pop("price").astype(float).tolist()

    # Merge demographics, engineer features, align columns
    merged = _merge_demographics(base)
    engineered = add_features(merged)
    X = _align_columns(engineered, expected)

    # Predict (to price space)
    t0 = time.time()
    try:
        preds = model.predict(X)
        preds = np.expm1(preds) if LOG_TARGET else preds
        preds = [float(x) for x in preds]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference failed during evaluation: {e}")
    latency_ms = int((time.time() - t0) * 1000)

    # Metrics
    metrics = _compute_metrics(y_true, preds)

    # Per-item details (incl. % error when safe)
    per_item = []
    for i, (yt, yp) in enumerate(zip(y_true, preds)):
        ape = (abs(yt - yp) / yt * 100.0) if abs(yt) > 1e-9 else None
        per_item.append({
            "index": i,
            "y_true": float(yt),
            "y_pred": float(yp),
            "abs_error": float(abs(yt - yp)),
            "abs_pct_error": float(ape) if ape is not None else None
        })

    return EvaluateResponse(
        predictions=preds,
        y_true=[float(v) for v in y_true],
        metrics=metrics,
        per_item=per_item,
        metadata={
            "n_inputs": len(preds),
            "model_stamp": stamp,
            "model_path": model_path,
            "features_path": feats_path,
            "latency_ms": latency_ms,
            "demographics_source": str(DEMOGRAPHICS_CSV),
        },
    )