#!/usr/bin/env bash
set -euo pipefail

cd /app

log() { echo "[startup] $*"; }

# ---- Paths (aligned with your training/inference scripts) ----
CV_FAMILY="house-price"
CV_ROOT="models/${CV_FAMILY}"
CV_LATEST="${CV_ROOT}/latest"
XGB_ART_DIR="models/experiments_xgb_optuna/artifacts"

# ---- Existence checks ----
has_cv_model() {
  # Require 'latest' marker pointing to a version dir that contains model.pkl and model_features.json
  [[ -f "${CV_LATEST}" ]] || return 1
  local ver; ver="$(cat "${CV_LATEST}" | tr -d '[:space:]')"
  [[ -n "${ver}" ]] || return 1
  local vdir="${CV_ROOT}/${ver}"
  [[ -d "${vdir}" ]] || return 1
  [[ -f "${vdir}/model.pkl" ]] || return 1
  [[ -f "${vdir}/model_features.json" ]] || return 1
  return 0
}

has_xgb_model() {
  # Look for a "stamp" that exists for BOTH the .pkl and the .json
  [[ -d "${XGB_ART_DIR}" ]] || return 1
  shopt -s nullglob
  local pkls=("${XGB_ART_DIR}"/xgb_final_*.pkl)
  local feats=("${XGB_ART_DIR}"/model_features_*.json)
  (( ${#pkls[@]} > 0 && ${#feats[@]} > 0 )) || return 1
  # Find intersection of stamps between model and features
  for p in "${pkls[@]}"; do
    local stamp="${p##*/}"; stamp="${stamp#xgb_final_}"; stamp="${stamp%.pkl}"
    if [[ -f "${XGB_ART_DIR}/model_features_${stamp}.json" ]]; then
      return 0
    fi
  done
  return 1
}

# ---- Train only if model artifacts are missing in the mounted directories----
if [[ -f "create_model_cv.py" ]]; then
  if has_cv_model; then
    log "CV: artifacts found — skipping training."
  else
    log "CV: artifacts missing — starting training..."
    micromamba run -n housing python create_model_cv.py
  fi
else
  log "CV: create_model_cv.py not found — skipping."
fi

if [[ -f "create_model_xgb.py" ]]; then
  if has_xgb_model; then
    log "XGB: artifacts found — skipping training."
  else
    log "XGB: artifacts missing — starting training..."
    micromamba run -n housing python create_model_xgb.py
  fi
else
  log "XGB: create_model_xgb.py not found — skipping."
fi

# ---- Start APIs ----
WORKERS="${WORKERS:-2}"
log "Starting APIs with ${WORKERS} workers each..."

# 8000: baseline model (house-price);
micromamba run -n housing uvicorn model_inference:app \
  --host 0.0.0.0 --port 8000 --workers "${WORKERS}" &
# 8001: XGB (XGB_ART_DIR directory)
micromamba run -n housing uvicorn model_inference_xgb:app \
  --host 0.0.0.0 --port 8001 --workers "${WORKERS}" &

# ---- Graceful shutdown ----
trap "pkill -TERM -P $$; wait; exit 143" SIGINT SIGTERM
wait -n