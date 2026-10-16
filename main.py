import gc
import json
import os
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import requests
from catboost import CatBoostClassifier, CatBoostRegressor
from fastapi import FastAPI, HTTPException

from feature_engineering import build_current_features


app = FastAPI(
    title="Horse Prediction Service",
    version="1.0.0",
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "horse_future_v1_20261015",
)

MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        f"/models/{MODEL_VERSION}",
    )
)

HORSE_BACKEND_URL = os.getenv(
    "HORSE_BACKEND_URL",
    "https://horseracesbackend-production.up.railway.app",
).rstrip("/")

REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("REQUEST_TIMEOUT_SECONDS", "60")
)

REQUIRED_FILES = [
    "next_tf_mean.cbm",
    "next_tf_p10.cbm",
    "next_tf_p90.cbm",
    "peak_tf_12m_mean.cbm",
    "peak_tf_12m_p10.cbm",
    "peak_tf_12m_p90.cbm",
    "stakes_win_24m.cbm",
    "stakes_win_24m_calibrator.joblib",
    "earnings_12m_mean_log.cbm",
    "earnings_12m_p25.cbm",
    "earnings_12m_p50.cbm",
    "earnings_12m_p75.cbm",
    "earnings_12m_p90.cbm",
    "feature_schema.json",
    "manifest.json",
]


def safe_float(value, digits=None):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return round(value, digits) if digits is not None else value


def load_schema():
    path = MODEL_DIR / "feature_schema.json"
    if not path.exists():
        raise RuntimeError(f"Feature schema not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_horse_history(horse_name):
    url = f"{HORSE_BACKEND_URL}/api/APIData_Table2/horse"

    try:
        response = requests.get(
            url,
            params={"horseName": horse_name},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to contact horse-data backend: {exc}",
        )

    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for horse '{horse_name}'.",
        )

    if not response.ok:
        raise HTTPException(
            status_code=502,
            detail=f"Horse-data backend returned HTTP {response.status_code}.",
        )

    payload = response.json()

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = (
            payload.get("data")
            or payload.get("rows")
            or payload.get("results")
            or []
        )
        if isinstance(rows, dict):
            rows = [rows]
    else:
        rows = []

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for horse '{horse_name}'.",
        )

    return rows


def predict_regression(model_file, X):
    path = MODEL_DIR / model_file
    if not path.exists():
        raise RuntimeError(f"Missing model: {path}")

    model = CatBoostRegressor()
    model.load_model(str(path))
    try:
        value = float(model.predict(X)[0])
    finally:
        del model
        gc.collect()
    return value


def predict_stakes_probability(X):
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "stakes_win_24m.cbm"))
    try:
        raw = float(model.predict_proba(X)[0, 1])
    finally:
        del model
        gc.collect()

    calibrator = joblib.load(
        MODEL_DIR / "stakes_win_24m_calibrator.joblib"
    )
    try:
        calibrated = float(calibrator.predict(np.array([raw]))[0])
    finally:
        del calibrator
        gc.collect()

    return min(max(calibrated, 0.0), 1.0)


def calculate_confidence(X, peak_p10, peak_p90, runs_so_far):
    key_features = [
        "current_tf",
        "tf_avg_3",
        "tf_avg_5",
        "career_high_tf",
        "current_tfig",
        "finish_avg_3",
        "days_since_prev_run",
        "distance",
        "raceClass",
    ]

    available = [c for c in key_features if c in X.columns]
    completeness = (
        float(X[available].notna().mean(axis=1).iloc[0])
        if available else 0.5
    )

    history_score = min(max(float(runs_so_far) / 10.0, 0.0), 1.0)
    interval_width = max(float(peak_p90) - float(peak_p10), 0.0)
    interval_score = 1.0 - min(max(interval_width / 40.0, 0.0), 1.0)

    score = round(
        100 * (
            0.50 * completeness
            + 0.30 * history_score
            + 0.20 * interval_score
        )
    )

    return int(min(max(score, 0), 100))


@app.get("/")
def root():
    return {
        "service": "Horse Prediction Service",
        "status": "running",
        "modelVersion": MODEL_VERSION,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelVersion": MODEL_VERSION,
        "modelDirectory": str(MODEL_DIR),
    }


@app.get("/health/models")
def model_health():
    existing = []
    missing = []

    for filename in REQUIRED_FILES:
        path = MODEL_DIR / filename
        if path.exists():
            existing.append({
                "name": filename,
                "sizeMB": round(path.stat().st_size / 1024 / 1024, 2),
            })
        else:
            missing.append(filename)

    return {
        "status": "ok" if not missing else "incomplete",
        "modelVersion": MODEL_VERSION,
        "modelDirectory": str(MODEL_DIR),
        "directoryExists": MODEL_DIR.exists(),
        "modelsReady": len(missing) == 0,
        "filesFound": len(existing),
        "filesExpected": len(REQUIRED_FILES),
        "existingFiles": existing,
        "missingFiles": missing,
    }


@app.get("/predict/{horse_name}")
def predict_horse(horse_name: str):
    try:
        schema = load_schema()
        rows = fetch_horse_history(horse_name)
        X, meta = build_current_features(rows, schema)

        min_runs = int(schema.get("min_runs_for_snapshot", 2))

        if meta["runsSoFar"] < min_runs:
            raise HTTPException(
                status_code=422,
                detail=f"Not enough racing history. Minimum runs: {min_runs}.",
            )

        if meta["currentTF"] is None:
            raise HTTPException(
                status_code=422,
                detail="Horse has no valid current Timeform performance rating.",
            )

        next_vals = sorted([
            predict_regression("next_tf_p10.cbm", X),
            predict_regression("next_tf_mean.cbm", X),
            predict_regression("next_tf_p90.cbm", X),
        ])
        next_p10, next_mean, next_p90 = next_vals

        peak_vals = sorted([
            predict_regression("peak_tf_12m_p10.cbm", X),
            predict_regression("peak_tf_12m_mean.cbm", X),
            predict_regression("peak_tf_12m_p90.cbm", X),
        ])
        peak_p10, peak_mean, peak_p90 = peak_vals

        if meta["stakesWinsSoFar"] > 0:
            stakes_probability = None
            stakes_status = "already_achieved"
        else:
            stakes_probability = predict_stakes_probability(X)
            stakes_status = "predicted"

        earnings_log = predict_regression(
            "earnings_12m_mean_log.cbm", X
        )
        expected_earnings = max(float(np.expm1(earnings_log)), 0.0)

        earnings_vals = sorted([
            max(predict_regression("earnings_12m_p25.cbm", X), 0.0),
            max(predict_regression("earnings_12m_p50.cbm", X), 0.0),
            max(predict_regression("earnings_12m_p75.cbm", X), 0.0),
            max(predict_regression("earnings_12m_p90.cbm", X), 0.0),
        ])
        ep25, ep50, ep75, ep90 = earnings_vals

        confidence = calculate_confidence(
            X,
            peak_p10,
            peak_p90,
            meta["runsSoFar"],
        )

        return {
            "horseName": horse_name,
            "predictionDate": date.today().isoformat(),
            "modelVersion": MODEL_VERSION,
            "lastRunDate": meta["lastRunDate"],
            "runsUsed": meta["runsSoFar"],

            "currentTF": safe_float(meta["currentTF"], 1),

            "nextRunTF": safe_float(next_mean, 1),
            "nextRunTFP10": safe_float(next_p10, 1),
            "nextRunTFP90": safe_float(next_p90, 1),

            "projectedPeakTF12m": safe_float(peak_mean, 1),
            "projectedPeakTF12mP10": safe_float(peak_p10, 1),
            "projectedPeakTF12mP90": safe_float(peak_p90, 1),

            "projectedUpsideTF": safe_float(
                peak_mean - meta["currentTF"], 1
            ),

            "stakesWinnerProbability24m": (
                safe_float(stakes_probability, 4)
                if stakes_probability is not None
                else None
            ),
            "stakesWinnerProbability24mPct": (
                safe_float(stakes_probability * 100, 1)
                if stakes_probability is not None
                else None
            ),
            "stakesStatus": stakes_status,

            "expectedEarnings12m": safe_float(expected_earnings, 0),
            "earningsP25": safe_float(ep25, 0),
            "earningsP50": safe_float(ep50, 0),
            "earningsP75": safe_float(ep75, 0),
            "earningsP90": safe_float(ep90, 0),

            "confidenceScore": confidence,

            "disclaimer": "Model estimates are probabilistic forecasts, not guarantees.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )
