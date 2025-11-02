# House Price Inference – README

This repository deploys **two REST APIs** to predict house prices in Seattle using the provided data and an augmented feature set with demographic information joined **server-side**. It satisfies the project’s requirements for: endpoint inputs/outputs, backend demographics join, scalability considerations, **model update with no downtime**, a **bonus minimal-feature endpoint**, a simple test path, and **model evaluation**. 

---

## 1) What’s included

* **Baseline CV model (KNN pipeline)** with 5-fold cross-validation, versioned as `models/house-price/vN` with a `latest` alias, and persisted artifacts (`model.pkl`, `model_features.json`, `evaluation.json`, `manifest.json`).  
* **XGBoost (Optuna)** track with study + K-Fold CV on best params; final retrain and synchronized artifacts saved under `models/experiments_xgb_optuna/artifacts/` (model, features, report with CV summary).  
* **Two FastAPI apps**:

  * `model_inference.py` (Baseline, **port 8000**): `/predict`, `/predict_basic`, `/evaluate_basic`, `/health_check`. Loads **latest** model on each request to allow **hot-swap** updates without restarting.   
  * `model_inference_xgb.py` (XGB, **port 8001**): `/predict_one`, `/predict_batch`, `/health_check`. Finds the newest **artifact stamp** so you can drop new artifacts and serve them immediately.  

---

## 2) Checking solution requirements

1. **Deploy the model as a REST endpoint** receiving JSON (inputs = columns from `future_unseen_examples.csv`).

   * `/predict` accepts that schema and merges demographics **server-side** before scoring. The response returns predictions + metadata (version, latency, artifact paths).   
   * The demographics table **must not** be sent by the client; the service adds it on the backend. This is implemented in the request path (coercing `zipcode` to string, joining demographics, aligning to expected features).   
2. **Scalability + updates without downtime.**

   * Containers (Uvicorn workers) can scale horizontally; each request **loads the current artifacts** based on `latest` (baseline) or **latest stamp** (XGB). That enables **zero-restart model updates** by just writing new artifacts and flipping the marker (baseline) or saving a newer stamped pair (XGB).  
3. **Bonus endpoint (minimal features).**

   * `/predict_basic` accepts only the subset the basic model uses (e.g., bedrooms, bathrooms, sqft, floors, zipcode). Demographics are still joined on the server; missing expected columns are safely handled. 
4. **Test script / simple demonstration.**

   * The APIs accept batches (or one item, in the XGB app). Using rows from `data/future_unseen_examples.csv` satisfies the test requirement. (Example curl below.) 
5. **Evaluate model performance.**

   * `create_model_cv.py` runs 5-fold CV and persists **RMSE/MAE/R²** means/std per fold. A manifest is stored with model metadata.  
   * The XGB track writes a **report** with best params, CV summary, and final estimators. 

> The above items map directly to the “Deliverables/Requirements” section in the assignment README (inputs, backend demographics join, scaling, hot model updates, minimal-feature endpoint, test script, and evaluation). 

---

## 3) Training the models (local)

> Optional if you rely on the container’s entrypoint; it can detect artifacts and skip retraining.

* **Create baseline CV artifacts** (versioned `vN` under `models/house-price/`):

  ```bash
  conda env create -f conda_environment.yml
  conda activate housing
  python create_model_cv.py
  ```

  Artifacts: `model.pkl`, `model_features.json`, `v_eval/evaluation.json`, `manifest.json`, and `latest` alias.   

* **Create XGB artifacts** (under `models/experiments_xgb_optuna/artifacts/`):

  ```bash
  conda activate housing
  python create_model_xgb.py
  ```

  Saves `xgb_final_<stamp>.pkl`, `model_features_<stamp>.json`, and `report_<stamp>.json`. 

---

## 4) Build & run the container

> The container starts **both** APIs: baseline at **:8000** and XGB at **:8001**.

### Build

```bash
docker build -t housing-inference .
```

### Run

```bash
docker run --rm \
  -p 8000:8000 -p 8001:8001 \
  --cpus=8 --cpuset-cpus=0-7 \
  -v "${PWD}:/app" \
  housing-inference
```

* Mounting `${PWD}` lets the APIs read/write `models/…` and `data/…` on your host.
* On startup, training scripts are invoked **only if** valid artifacts are missing; otherwise the container **reuses** existing versions. This ensures **idempotent** restarts and **no retrain loops**. (Baseline artifacts are detected via `models/house-price/latest` → `vN` folder; XGB via matching stamped model/features files.)  

---

## 5) API quickstart

### Health checks

```bash
curl -s http://localhost:8000/health_check
curl -s http://localhost:8001/health_check
```

Both endpoints validate that model artifacts and demographics are available (and show the resolved `latest`/`stamp`).

### Predict (baseline `/predict`, full schema)

Send rows from `data/future_unseen_examples.csv`; the service joins demographics on the backend and aligns to the model’s expected features.

### Predict (baseline `/predict_basic`, minimal schema)

Only the subset used by the basic model; demographics are still merged server-side. Useful when the upstream system can’t send the full set. 

### Predict (XGB `/predict_one` or `/predict_batch`)

XGB app applies the same **feature engineering** used at training (e.g., `rooms`, `baths_per_bed`, `lot_coverage`, log transforms, ratios) after merging demographics; it selects the **latest stamped** artifact automatically. 

---

## 6) Model evaluation artifacts

* Baseline: 5-fold CV with persisted **RMSE/MAE/R²** (means/std + per-fold) under `v_eval/evaluation.json`. 
* XGB: `report_<stamp>.json` includes **best params**, **final n_estimators**, and CV summary. 

These files support presentation of generalization quality and comparison across versions.

---

## 7) Operational notes

* **Hot model updates** without redeploy:

  * Baseline: train a new version (e.g., `v3`) and write `models/house-price/latest` = `v3`. New requests will load `v3`. 
  * XGB: drop a newer `xgb_final_<stamp>.pkl` + matching `model_features_<stamp>.json`. New requests will use the newest common stamp.
* **Demographics join** is performed server-side in every API (zipcode cast to string, merge, then feature alignment). Clients **must not** send demographic columns.
  * I understand that reloading the model after each request may add some latency to the inference, however in a cloud scenario we may use ML tools that allow a dynamic loading: When a new version is marked as production-ready in the registry, the serving service can load it dynamically without requiring a full application redeployment or restart.

---

## 8) Repository structure (key files)

/app
├── create_model_cv.py - Baseline pipeline + 5-fold CV + versioning/artifacts
├── create_model_xgb.py - Optuna tuning + CV on best params + final retrain + artifacts
├── model_inference.py - Baseline API (port 8000): /predict, /predict_basic, /evaluate_basic, /health_check
├── model_inference_xgb.py - XGB API (port 8001): /predict_one, /predict_batch, /health_check
├── conda_environment.yml - Reproducible environment definition - Added a few dependencies with pip to avoid errors
├── dockerfile - Container build instructions
├── entrypoint.sh - Startup script that trains models only if artifacts are missing
└── models/ - Directory where all of the model artifacts are stored (this is an attempt to mimic the strcuture of a ML model registry)
    ├── house-price/
    │   ├── latest - Marker pointing to the current active version (e.g., v1, v2, v3)
    │   └── vN/
    │       ├── model.pkl
    │       ├── model_features.json
    │       ├── evaluation.json - evaluation of the trained model
    │       └── manifest.json
    └── experiments_xgb_optuna/ - Artifacts for xgb models
        └── artifacts/
            ├── xgb_final_<timestamp>.pkl
            ├── model_features_<timestamp>.json
            └── report_<timestamp>.json
