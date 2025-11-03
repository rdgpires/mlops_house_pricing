from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
import pickle

# ---------------- Config ----------------
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"

SALES_COLS = [
    "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "sqft_above", "sqft_basement", "zipcode"
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
EARLY_STOP = 100
N_ESTIMATORS = 2000

# Optuna persistence (resume-able)
STUDY_NAME = "xgb_houseprice_311"
STORAGE = "sqlite:///optuna_xgb.db"

OUTDIR = pathlib.Path("models/experiments_xgb_optuna")
OUTDIR.mkdir(parents=True, exist_ok=True)

n_jobs=8


# -------------- Helpers ----------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_log1p(s: pd.Series) -> pd.Series:
    # avoid negatives/NaN; works with missing values as well
    return np.log1p(s.clip(lower=0).fillna(0.0))


def _as_fraction(s: pd.Series) -> pd.Series:
    """Converts percentages into fractions (numbers between o and 1)."""
    s = s.astype(float)
    # heuristics: if median > 1.5, it is probably a percentage
    median = s.replace([np.inf, -np.inf], np.nan).dropna().median() if len(s) else 0
    if pd.notna(median) and median > 1.5:
        s = s / 100.0
    return s.clip(lower=0.0, upper=1.0).fillna(0.0)


def _entropy_from_parts(parts: pd.DataFrame) -> pd.Series:
    """Normalized entropy."""
    vals = parts.fillna(0.0).astype(float)
    total = vals.sum(axis=1)
    # avoids division by zero
    frac = vals.div(total.replace(0, np.nan), axis=0).fillna(0.0)
    # Shannon entropy
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(frac * np.log(frac.where(frac > 0, 1.0))).sum(axis=1)
    # normalizes by log of the number of categories
    k = vals.shape[1] if vals.shape[1] > 0 else 1
    ent_norm = ent / np.log(k)
    return ent_norm.clip(lower=0.0, upper=1.0).fillna(0.0)


def load_xy(engineer: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    # Coerce dtype={"zipcode": str}
    sales = pd.read_csv(SALES_PATH, usecols=SALES_COLS, dtype={"zipcode": str})
    demo = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})

    # Merge demographics
    df = sales.merge(demo, on="zipcode", how="left").drop(columns="zipcode")

    y = df.pop("price").to_numpy()

    # Feature engineering
    X = add_features(df) if engineer else df
    return X, y


def add_demo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for demographic variables.
    """
    out = df.copy()

    # Quantity columns
    qty_cols = [
        "ppltn_qty", "urbn_ppltn_qty", "sbrbn_ppltn_qty", "farm_ppltn_qty",
        "non_farm_qty",
        "edctn_less_than_9_qty", "edctn_9_12_qty", "edctn_high_schl_qty",
        "edctn_some_clg_qty", "edctn_assoc_dgre_qty",
        "edctn_bchlr_dgre_qty", "edctn_prfsnl_qty",
    ]
    # Percentage columns
    per_cols = [
        "per_urbn", "per_sbrbn", "per_farm", "per_non_farm",
        "per_less_than_9", "per_9_to_12", "per_hsd",
        "per_some_clg", "per_assoc", "per_bchlr", "per_prfsnl",
    ]
    money_cols = ["medn_hshld_incm_amt", "medn_incm_per_prsn_amt", "hous_val_amt"]

    # 1) Logs for assymetric columns
    for c in qty_cols + money_cols:
        if c in out.columns:
            out[f"{c}_log"] = _safe_log1p(out[c])

    # 2) Apply fractions to percentage columns
    for c in per_cols:
        if c in out.columns:
            out[f"{c}_frac"] = _as_fraction(out[c])

    # 3) Ratio house sales price and income
    if "hous_val_amt" in out.columns and "medn_hshld_incm_amt" in out.columns:
        out["house_value_to_income"] = out["hous_val_amt"] / (out["medn_hshld_incm_amt"] + 1.0)
        out["income_to_house_value"] = out["medn_hshld_incm_amt"] / (out["hous_val_amt"] + 1.0)

    # 4) Apply normalized entropy into superior education
    high_edu_parts = []
    for c in ["per_bchlr", "per_prfsnl"]:
        if c in out.columns:
            high_edu_parts.append(f"{c}_frac")
            if f"{c}_frac" not in out.columns:
                out[f"{c}_frac"] = _as_fraction(out[c])
    if high_edu_parts:
        out["share_high_edu"] = out[high_edu_parts].sum(axis=1).clip(0.0, 1.0)

    edu_qty_present = [c for c in qty_cols if c.startswith("edctn_") and c in out.columns]
    if edu_qty_present:
        out["edu_entropy"] = _entropy_from_parts(out[edu_qty_present])

    # 5) Urbanization treatment
    if "per_urbn" in out.columns:
        if "per_farm" in out.columns:
            out["urbanization_idx"] = _as_fraction(out["per_urbn"]) - _as_fraction(out["per_farm"])
        else:
            out["urbanization_idx"] = _as_fraction(out["per_urbn"])
    elif "urbn_ppltn_qty" in out.columns and "ppltn_qty" in out.columns:
        out["urbanization_idx"] = (out["urbn_ppltn_qty"] / (out["ppltn_qty"] + 1.0)).clip(0.0, 1.0)

    # 6) Non-farm share as a quantity
    if "non_farm_qty" in out.columns and "ppltn_qty" in out.columns and "per_non_farm_frac" not in out.columns:
        out["per_non_farm_frac"] = (out["non_farm_qty"] / (out["ppltn_qty"] + 1.0)).clip(0.0, 1.0)

    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Feature engineering
    """
    out = df.copy()

    # ---------- 1) Simple combinations and ratios ----------
    out["rooms"] = out["bedrooms"] + out["bathrooms"]
    bed_nonzero = out["bedrooms"].replace(0, np.nan)
    out["baths_per_bed"] = (out["bathrooms"] / bed_nonzero).fillna(out["bathrooms"])
    out["bed_bath_x"] = out["bedrooms"] * out["bathrooms"]
    out["lot_coverage"] = (out["sqft_living"] / (out["sqft_lot"] + 1.0)).clip(0, 1.0)

    # ---------- 2) Structure ----------
    out["is_basement"] = (out["sqft_basement"] > 0).astype(int)
    out["basement_ratio"] = out["sqft_basement"] / (out["sqft_living"] + 1.0)
    out["above_ratio"] = out["sqft_above"] / (out["sqft_living"] + 1.0)

    # ---------- 3) Logs for assymetric values  ----------
    out["sqft_living_log"] = np.log1p(out["sqft_living"])
    out["sqft_lot_log"] = np.log1p(out["sqft_lot"])

    # ---------- 4) Add non linearity ----------
    out["bathrooms_sq"] = out["bathrooms"] ** 2
    out["floors_sq"] = out["floors"] ** 2

    # ---------- 5) Apply demography feature engieneering ----------
    out = add_demo_features(out)

    return out


@dataclass
class FitResult:
    model: XGBRegressor
    rmse: float
    mae: float
    r2: float
    best_iteration: int
    fit_time_sec: float


def train_eval_xgb(params: dict, early_stopping_rounds=EARLY_STOP, log_target=True) -> FitResult:
    X, y_raw = load_xy()
    X_tr, X_va, y_tr_raw, y_va_raw = train_test_split(
        X, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    y_tr = np.log1p(y_tr_raw) if log_target else y_tr_raw
    y_va = np.log1p(y_va_raw) if log_target else y_va_raw

    base = dict(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1e-3,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        eval_metric="rmse",
        early_stopping_rounds=early_stopping_rounds,
    )
    cfg = {**base, **params}
    model = XGBRegressor(**cfg)

    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    secs = round(time.time() - t0, 2)

    pred = model.predict(X_va)
    if log_target:
        pred = np.expm1(pred)

    return FitResult(
        model=model,
        rmse=rmse(y_va_raw, pred),
        mae=mean_absolute_error(y_va_raw, pred),
        r2=r2_score(y_va_raw, pred),
        best_iteration=int(model.best_iteration),
        fit_time_sec=secs,
    )


# -------------- Optuna Objective --------------
def objective(trial: optuna.Trial) -> float:
    # Espaço de busca enxuto e eficiente
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1e-1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

    # early_stopping_rounds como categórico pequeno
    es_rounds = trial.suggest_categorical("early_stopping_rounds", [50, 100, 200])

    # Treino & score (RMSE na escala original)
    res = train_eval_xgb(params, early_stopping_rounds=es_rounds, log_target=True)

    # Report & prune
    trial.report(res.rmse, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return res.rmse


# -------------- Run Study --------------
def run_study(n_trials: int = 40, timeout: int | None = None) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=True)
    return study


# -------------- K-Fold CV on Best Params --------------
def kfold_cv_on_best(best_params: dict, best_es: int, n_splits: int = 5, log_target: bool = True) -> Dict[str, Any]:
    """
    Runs K-fold CV using hyperparameter from optuna optimization.
    """
    X, y_raw = load_xy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    per_fold = {
        "rmse": [], "mae": [], "r2": [], "best_iteration": [], "fit_time_sec": [],
    }

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr_raw, y_va_raw = y_raw[tr_idx], y_raw[va_idx]

        y_tr = np.log1p(y_tr_raw) if log_target else y_tr_raw
        y_va = np.log1p(y_va_raw) if log_target else y_va_raw

        cfg = {
            **best_params,
            "n_estimators": N_ESTIMATORS,
            "tree_method": "hist",
            "eval_metric": "rmse",
            "random_state": RANDOM_STATE,
            "n_jobs": 0,
            "early_stopping_rounds": best_es,
        }
        model = XGBRegressor(**cfg)

        t0 = time.time()
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        secs = round(time.time() - t0, 2)

        pred = model.predict(X_va)
        if log_target:
            pred = np.expm1(pred)

        per_fold["rmse"].append(rmse(y_va_raw, pred))
        per_fold["mae"].append(float(mean_absolute_error(y_va_raw, pred)))
        per_fold["r2"].append(float(r2_score(y_va_raw, pred)))
        per_fold["best_iteration"].append(int(model.best_iteration))
        per_fold["fit_time_sec"].append(secs)

    agg = {
        "n_splits": n_splits,
        "rmse_mean": float(np.mean(per_fold["rmse"])),
        "rmse_std":  float(np.std(per_fold["rmse"], ddof=1)),
        "mae_mean":  float(np.mean(per_fold["mae"])),
        "mae_std":   float(np.std(per_fold["mae"], ddof=1)),
        "r2_mean":   float(np.mean(per_fold["r2"])),
        "r2_std":    float(np.std(per_fold["r2"], ddof=1)),
        "per_fold": per_fold,
    }

    # n_estimators final as best_iteration median
    final_n_estimators = int(max(1, int(np.median(per_fold["best_iteration"]))))

    return {"cv_summary": agg, "final_n_estimators": final_n_estimators}


# -------------- Retrain Final + Save --------------
def retrain_final(best_params: dict, final_n_estimators: int, log_target=True, cv_report: Dict[str, Any] | None = None):
    X, y_raw = load_xy()

    cfg_final = {
        **best_params,
        "n_estimators": max(1, final_n_estimators),
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": 0,
    }
    cfg_final.pop("early_stopping_rounds", None)

    y_all = np.log1p(y_raw) if log_target else y_raw
    final = XGBRegressor(**cfg_final)
    t0 = time.time()
    final.fit(X, y_all, verbose=False)
    total_secs = round(time.time() - t0, 2)

    # Save artifacts
    stamp = int(time.time())
    (OUTDIR / "artifacts").mkdir(parents=True, exist_ok=True)

    with open(OUTDIR / "artifacts" / f"xgb_final_{stamp}.pkl", "wb") as f:
        pickle.dump(final, f)

    with open(OUTDIR / "artifacts" / f"model_features_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    report: Dict[str, Any] = {
        "best_params": best_params,
        "final_n_estimators": int(cfg_final["n_estimators"]),
        "log_target": log_target,
        "columns": list(X.columns),
        "final_fit_time_sec": total_secs,
    }
    if cv_report is not None:
        report["cv"] = cv_report["cv_summary"]

    with open(OUTDIR / "artifacts" / f"report_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved model/features/report to: {OUTDIR / 'artifacts'}")
    return final, report


# -------------- Script Entry --------------
if __name__ == "__main__":
    # Tuning
    study = run_study(n_trials=40)

    print("\n== Optuna Best ==")
    print("Best RMSE:", study.best_value)
    print("Params:", study.best_params)

    # Remove ES from best params params
    best_es = study.best_params.pop("early_stopping_rounds", EARLY_STOP)

    # Apply CV in best params
    cv_out = kfold_cv_on_best(study.best_params, best_es=best_es, n_splits=5, log_target=True)
    print("\n== CV Summary (Best Params) ==")
    print(json.dumps({
        "rmse_mean": cv_out["cv_summary"]["rmse_mean"],
        "rmse_std":  cv_out["cv_summary"]["rmse_std"],
        "mae_mean":  cv_out["cv_summary"]["mae_mean"],
        "mae_std":   cv_out["cv_summary"]["mae_std"],
        "r2_mean":   cv_out["cv_summary"]["r2_mean"],
        "r2_std":    cv_out["cv_summary"]["r2_std"],
        "final_n_estimators": cv_out["final_n_estimators"],
    }, indent=2))

    # final train
    _final, report = retrain_final(study.best_params, cv_out["final_n_estimators"], log_target=True, cv_report=cv_out)
