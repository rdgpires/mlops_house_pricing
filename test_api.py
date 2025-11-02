import pandas as pd
import requests
import json

# Load a few test rows
df = pd.read_csv("data/future_unseen_examples.csv").head(3)
rows = df.to_dict(orient="records")

# Baseline kNN model (port 8000)
print("\n*** Baseline API ***")
r = requests.post("http://localhost:8000/predict", json={"items": rows})
print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2))

# Basic endpoint (subset of features)
basic_cols = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "sqft_above", "sqft_basement", "zipcode"
]
rows_basic = [{k: v for k, v in r_.items() if k in basic_cols} for r_ in rows]

r2 = requests.post("http://localhost:8000/predict_basic", json={"items": rows_basic})
print("\n*** Baseline /predict_basic ***")
print("Status:", r2.status_code)
print(json.dumps(r2.json(), indent=2))

# XGB API (port 8001)
print("\n*** XGB API ***")
r3 = requests.post("http://localhost:8001/predict_batch", json={"items": rows_basic})
print("Status:", r3.status_code)
print(json.dumps(r3.json(), indent=2))
