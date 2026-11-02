"""Flask API + web form for churn prediction.

Run:  python -m app.app           (dev)
      gunicorn app.app:app        (production)
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.features import RAW_CATEGORICAL, RAW_FEATURES, RAW_NUMERIC  # noqa: F401 (pickle needs src.features)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(BASE, "templates"))
MODEL = joblib.load(os.path.join(BASE, "models", "best_model.joblib"))
META = json.load(open(os.path.join(BASE, "models", "model_meta.json")))
THRESHOLD = META["threshold"]

REQUIRED = ["tenure_months", "contract", "internet_service", "payment_method",
            "monthly_charges", "support_calls"]
DEFAULTS = {"age": 44, "region": "Central", "num_services": 2, "total_charges": None,
            "late_payments": 0, "paperless_billing": "No"}


def _prepare(record):
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    rec = {**DEFAULTS, **record}
    if rec["total_charges"] is None:
        rec["total_charges"] = float(rec["monthly_charges"]) * float(rec["tenure_months"])
    for k in RAW_NUMERIC:
        rec[k] = float(rec[k])
    return {k: rec[k] for k in RAW_FEATURES}


def _risk(p):
    return "High" if p >= 0.6 else "Medium" if p >= THRESHOLD else "Low"


def _predict(records):
    df = pd.DataFrame([_prepare(r) for r in records])
    probs = MODEL.predict_proba(df)[:, 1]
    return [{"churn_probability": round(float(p), 4), "will_churn": bool(p >= THRESHOLD),
             "risk_level": _risk(p)} for p in probs]


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok", model=META["model"], threshold=THRESHOLD)


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Send a JSON object"), 400
    try:
        return jsonify(_predict([body])[0])
    except (ValueError, KeyError, TypeError) as e:
        return jsonify(error=str(e)), 400


@app.post("/predict_batch")
def predict_batch():
    body = request.get_json(silent=True)
    if not isinstance(body, list) or not body:
        return jsonify(error="Send a non-empty JSON list"), 400
    try:
        return jsonify(predictions=_predict(body))
    except (ValueError, KeyError, TypeError) as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
