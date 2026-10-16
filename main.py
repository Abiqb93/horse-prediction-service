import os
from pathlib import Path

from fastapi import FastAPI

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
                "sizeMB": round(
                    path.stat().st_size / 1024 / 1024,
                    2,
                ),
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