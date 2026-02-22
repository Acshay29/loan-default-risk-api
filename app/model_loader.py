import joblib
import logging
from app.config import XGB_MODEL_PATH, LGB_MODEL_PATH, FEATURE_NAMES_PATH

logger = logging.getLogger(__name__)


def load_models():
    """Load XGBoost, LightGBM models and feature names from disk."""

    try:
        xgb_model = joblib.load(XGB_MODEL_PATH)
        logger.info(f"XGBoost model loaded from {XGB_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load XGBoost model: {e}")
        raise

    try:
        lgb_model = joblib.load(LGB_MODEL_PATH)
        logger.info(f"LightGBM model loaded from {LGB_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load LightGBM model: {e}")
        raise

    try:
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        logger.info("Feature names loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load feature names: {e}")
        raise

    return xgb_model, lgb_model, feature_names


# Load once at import time
xgb_model, lgb_model, feature_names = load_models()