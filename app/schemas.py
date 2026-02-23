from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    probability: float
    prediction: int