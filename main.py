import os

from fastapi import FastAPI

app = FastAPI(
    title="Horse Prediction Service",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "service": "Horse Prediction Service",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelVersion": os.getenv(
            "MODEL_VERSION",
            "horse_future_v1_20261015"
        )
    }