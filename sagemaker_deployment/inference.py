"""SageMaker inference handlers for the anomaly detector.

SageMaker calls model_fn once when the server starts, then input_fn -> predict_fn -> output_fn
for every request. A request holds one or many claims (JSON, or headerless CSV in FEATURES order);
the response holds one result per claim, in the same order. Each result also echoes the claim's
Doctor ID and, when the request includes it, Claim ID.
"""

import json
import os
from io import StringIO

import joblib
import pandas as pd

MODEL_FILE = "anomaly_detector.joblib"

# Same names and order as the notebook; model_fn checks them against the saved model.
FEATURES = [
    "CLAIM_DOLLAR_AMOUNT_PAID", "CLAIM_SERVICES_FULL_SUBMISSION", "n_lines", "n_services", "n_examinations",
    "PATIENT_ID", "DOCTOR_ID", "PRACTICE_ID", "PRIMARY_EXAMINATION_CODE",
]
TEXT_COL = "PRIMARY_EXAMINATION_CODE"  # every other feature is a number
ID_COL = "CLAIM_ID"  # optional in the request; echoed back, never given to the model


def model_fn(model_dir):
    """
    Purpose: Loads the model and verifies the script's list of Features matches exactly the model's
    Frequency: Function runs only once at startup
    """
    model = joblib.load(os.path.join(model_dir, MODEL_FILE))
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError(f"Model was trained on {list(model.feature_names_in_)}, not {FEATURES}")
    return model


def input_fn(request_body, content_type):
    """
    Purpose: Verifies the format of every request validating the JSON or the CSV
    Frequency: Function runs at every request
    """
    # A content type with parameters (e.g. "; charset=utf-8") can arrive as bytes.
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    mime = (content_type or "").split(";")[0].strip().lower()

    if mime == "application/json":
        payload = json.loads(request_body)
        claims = pd.DataFrame(payload if isinstance(payload, list) else [payload])
    elif mime == "text/csv":
        claims = pd.read_csv(StringIO(request_body), header=None, dtype=str)
        if claims.shape[1] != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} values per row, got {claims.shape[1]}")
        claims.columns = FEATURES
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    return _clean(claims)


def _clean(claims):
    missing = [c for c in FEATURES if c not in claims.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    keep = FEATURES + ([ID_COL] if ID_COL in claims.columns else [])
    claims = claims[keep].copy()
    numbers = [c for c in FEATURES if c != TEXT_COL]
    # IDs sent as text would silently encode as "never seen", so force real numbers.
    claims[numbers] = claims[numbers].apply(pd.to_numeric, errors="coerce").astype("float64")
    bad = [c for c in keep if claims[c].isna().any()]
    if bad:
        raise ValueError(f"Missing or invalid values in: {bad}")

    claims[TEXT_COL] = claims[TEXT_COL].astype(str)
    return claims


def predict_fn(claims, model):
    decision = model.decision_function(claims[FEATURES])  # sklearn: negative = outlier
    results = pd.DataFrame({"anomaly_score": -decision, "flagged": decision < 0})
    results.insert(0, "DOCTOR_ID", claims["DOCTOR_ID"].astype("int64").to_numpy())
    if ID_COL in claims.columns:
        results.insert(0, ID_COL, claims[ID_COL].to_numpy())
    return results


def output_fn(results, accept):
    """
    Purpose: Sends the response
    Frequency: Function runs at every request
    """
    if (accept or "").split(";")[0].strip().lower() == "text/csv":
        return results.astype({"flagged": int}).to_csv(header=False, index=False), "text/csv"
    return json.dumps({"predictions": results.to_dict("records")}), "application/json"
