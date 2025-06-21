from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import joblib
import pandas as pd
from typing import List, Dict, Any
from contextlib import asynccontextmanager

MODEL_PATH='models/lr_best_model.joblib'

model = None  # Global model variable

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    try:
        with open(MODEL_PATH, "rb") as f:
            model = joblib.load(f)
        print("Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    yield
    # Optionally, clean up model here if needed
    # model = None

app = FastAPI(
    title="CS611 MLE Group Project – Model Inference API",
    lifespan=lifespan,
)
# --------------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Schema for a single instance prediction request."""

    features: Dict[str, Any] = Field(..., description="Mapping of feature names to values for a single instance.")


class BatchPredictRequest(BaseModel):
    """Schema for a batch prediction request."""

    instances: List[Dict[str, Any]] = Field(
        ..., description="List of feature mappings. Each mapping represents a single instance."
    )

@app.get("/")
async def read_root():
    """root path, for health check"""
    return {"message": "Model service is running!"}

# --------------------------------------------------------------------------------------
# Inference endpoints
# --------------------------------------------------------------------------------------
@app.post("/predict", summary="Predict for a single instance")
async def predict(request: PredictRequest):
    """
    Predicts the class of a given set of features.

    Args:
        request (PredictRequest): A request object containing a dictionary of features to predict.
            e.g. {"features": {"feature1": 1, "feature2": 2}}

    Returns:
        dict: A dictionary containing the predicted class.
            e.g. {"prediction": 0}
    """
    try:
        df = pd.DataFrame([request.features])
        prediction = model.predict(df)[0]
        return {"prediction": int(prediction)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict_batch", summary="Predict for multiple instances")
async def predict_batch(request: BatchPredictRequest):
    """
    Predicts the classes for multiple sets of features in a batch.

    Args:
        request (BatchPredictRequest): A request object containing a list of feature dictionaries.
            e.g. {"instances": [{"feature1": 1, "feature2": 2}, {"feature1": 3, "feature2": 4}]}

    Returns:
        dict: A dictionary containing the list of predicted classes.
            e.g. {"predictions": [0, 1]}
    """
    try:
        df = pd.DataFrame(request.instances)
        # Access the RandomForest model from the loaded dictionary
        predictions = model.predict(df).tolist()
        return {"predictions": [int(p) for p in predictions]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) 