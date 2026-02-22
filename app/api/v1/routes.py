from fastapi import APIRouter, HTTPException
import numpy as np
import logging

from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import xgb_model, feature_names

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = request.features

        # Convert to numpy
        input_array = np.array(features).reshape(1, -1)

        probability = float(xgb_model.predict_proba(input_array)[0][1])
        prediction = int(probability > 0.5)

        logger.info("Prediction successful")

        return PredictionResponse(
            probability=probability,
            prediction=prediction
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")