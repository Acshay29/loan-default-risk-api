import numpy as np
import logging
import pandas as pd
from app.model_loader import xgb_model, lgb_model, feature_names

logger = logging.getLogger(__name__)


def predict(features: list) -> tuple[float, int]:
    """
    Run ensemble prediction using XGBoost (60%) + LightGBM (40%).

    Args:
        features: List of 202 numeric features (already validated)

    Returns:
        (probability, prediction) tuple
    """
    input_df = pd.DataFrame([features], columns=feature_names)

    xgb_prob = xgb_model.predict_proba(input_df)[:, 1]
    lgb_prob = lgb_model.predict_proba(input_df)[:, 1]

    final_prob = 0.6 * xgb_prob + 0.4 * lgb_prob
    prediction = int(final_prob[0] >= 0.5)

    logger.info(f"Prediction: prob={final_prob[0]:.4f}, label={prediction}")
    return float(final_prob[0]), prediction