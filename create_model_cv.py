# create_model_cv.py
from __future__ import annotations

import json
import pathlib
import pickle
import re
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, neighbors, pipeline, preprocessing, impute

# ---------------- Config ----------------
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
SALES_COLUMN_SELECTION = [
    "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "sqft_above", "sqft_basement", "zipcode"
]

MODEL_FAMILY = "house-price"
MODELS_ROOT = pathlib.Path("models") / MODEL_FAMILY


# --------------- Helpers ----------------
def _next_incremental_version(models_root: pathlib.Path) -> str:
    """Scan models_root for vN folders and return next version like 'v3'."""
    models_root.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in models_root.glob("v*"):
        m = re.fullmatch(r"v(\d+)", p.name)
        if m:
            nums.append(int(m.group(1)))
    nxt = (max(nums) + 1) if nums else 1
    return f"v{nxt}"


def _save_latest_alias(models_root: pathlib.Path, version: str) -> None:
    models_root.mkdir(parents=True, exist_ok=True)
    with open(models_root / "latest", "w", encoding="utf-8") as f:
        f.write(version + "\n")


# --------------- Data IO ----------------
def load_data(
    sales_path: str,
    demographics_path: str,
    sales_column_selection: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the target and feature data by merging sales and demographics.
    Returns (X, y)
    """
    sales = pd.read_csv(
        sales_path,
        usecols=sales_column_selection,
        dtype={"zipcode": str},
    )
    demographics = pd.read_csv(
        demographics_path,
        dtype={"zipcode": str},
    )

    merged = sales.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    y = merged.pop("price")
    X = merged
    return X, y


# --------------- Main ----------------
def main() -> None:
    # Load full dataset
    X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    # Define pipeline (imputer added, as referenced in your report/manifest)
    model = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="median"),
        preprocessing.RobustScaler(),
        neighbors.KNeighborsRegressor(),
    )

    # ---- Cross-Validation (evaluate) ----
    cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    cv_res = model_selection.cross_validate(
        model,
        X,
        y,
        scoring=scoring,
        cv=cv,
        return_train_score=False,
        n_jobs=-1,
    )

    rmse_scores = -cv_res["test_neg_rmse"]
    mae_scores = -cv_res["test_neg_mae"]
    r2_scores = cv_res["test_r2"]

    # ---- Fit final model on ALL data (for export) ----
    model.fit(X, y)

    # ---- Versioning / folders ----
    version = _next_incremental_version(MODELS_ROOT)
    output_dir = (MODELS_ROOT / version)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = output_dir / "v_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ---- Persist evaluation ----
    report = {
        "cv": {
            "n_splits": 5,
            "rmse_mean": float(rmse_scores.mean()),
            "rmse_std": float(rmse_scores.std()),
            "mae_mean": float(mae_scores.mean()),
            "mae_std": float(mae_scores.std()),
            "r2_mean": float(r2_scores.mean()),
            "r2_std": float(r2_scores.std()),
            "per_fold": {
                "rmse": rmse_scores.tolist(),
                "mae": mae_scores.tolist(),
                "r2": r2_scores.tolist(),
            },
        },
        "n_rows": int(len(X)),
        "model": "KNeighborsRegressor",
        "scaler": "RobustScaler",
        "imputer": "SimpleImputer(median)",
    }
    with open(eval_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---- Persist artifacts ----
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(output_dir / "model_features.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f)

    manifest = {
        "model_family": MODEL_FAMILY,
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_count": int(X.shape[1]),
        "train_rows": int(len(X)),
        "scaler": "RobustScaler",
        "imputer": "SimpleImputer(median)",
        "estimator": "KNeighborsRegressor",
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _save_latest_alias(MODELS_ROOT, version)
    print(f"Saved model version: {version} at {output_dir}")
    print(f"Updated 'latest' marker in {MODELS_ROOT/'latest'}")


if __name__ == "__main__":
    main()