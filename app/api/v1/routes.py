from fastapi import APIRouter, HTTPException
import numpy as np
import logging

from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import xgb_model, feature_names

router = APIRouter()
logger = logging.getLogger(__name__)


# ===============================
# Health Check
# ===============================
@router.get("/health")
def health():
    return {"status": "ok"}


# ===============================
# Feature List Endpoint
# ===============================
@router.get("/features")
def get_features():
    return {"features": feature_names}


# ===============================
# Model Metadata
# ===============================
@router.get("/model-info")
def model_info():
    return {
        "model_type": "XGBoost",
        "num_features": len(feature_names),
        "version": "1.0.0"
    }


# ===============================
# Prediction Endpoint
# ===============================
@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = request.features
        expected_length = len(feature_names)

        # Validate feature length
        if len(features) != expected_length:
            raise HTTPException(
                status_code=422,
                detail=f"Expected {expected_length} features but received {len(features)}"
            )

        input_array = np.array(features).reshape(1, -1)

        probability = float(xgb_model.predict_proba(input_array)[0][1])
        prediction = int(probability > 0.5)

        logger.info("Prediction successful")

        return PredictionResponse(
            probability=probability,
            prediction=prediction
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")