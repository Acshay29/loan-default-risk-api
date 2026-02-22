from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
XGB_MODEL_PATH = BASE_DIR / "models" / "xgb_advanced.pkl"
LGB_MODEL_PATH = BASE_DIR / "models" / "lgbm_model.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "models" / "feature_names.pkl"