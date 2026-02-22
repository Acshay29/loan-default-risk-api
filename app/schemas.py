from pydantic import BaseModel, validator
from typing import List
from app.model_loader import feature_names
import math


class PredictionRequest(BaseModel):
    features: List[float]

    @validator("features")
    def validate_features(cls, v):
        # Validate feature length dynamically
        if len(v) != len(feature_names):
            raise ValueError(
                f"Expected {len(feature_names)} features, got {len(v)}"
            )

        # Check for NaN values
        if any(f != f for f in v):
            raise ValueError("Features must not contain NaN values")

        # Check for infinite values
        if any(math.isinf(f) for f in v):
            raise ValueError("Features must not contain infinite values")

        return v


class PredictionResponse(BaseModel):
    probability: float
    prediction: int