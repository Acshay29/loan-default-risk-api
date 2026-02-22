import os
import pandas as pd
import numpy as np
import joblib
import optuna

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# ── Load Data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING AND PREPROCESSING DATA")
print("=" * 60)

df = pd.read_csv("data/application_train.csv")
print("Original Shape:", df.shape)

# ── Drop High-Missing Columns ─────────────────────────────────────────────────
missing_percent = (df.isnull().sum() / len(df)) * 100
high_missing_cols = missing_percent[missing_percent > 60].index
df = df.drop(columns=high_missing_cols)
print(f"Dropped {len(high_missing_cols)} columns with >60% missing")
print("Shape After Dropping:", df.shape)

# ── Fill Missing Values ───────────────────────────────────────────────────────
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Remaining Missing Values:", df.isnull().sum().sum())

# ── Encode Categorical Variables ──────────────────────────────────────────────
df = pd.get_dummies(df, drop_first=True)
print("Shape After Encoding:", df.shape)

# ── Features and Target ───────────────────────────────────────────────────────
X = df.drop("TARGET", axis=1)
y = df["TARGET"]
print("\nFeature Shape:", X.shape)
print("Target Distribution:\n", y.value_counts())

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ── Handle Class Imbalance with SMOTE ────────────────────────────────────────
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Convert back to DataFrame (important for column name tracking)
X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
X_test = pd.DataFrame(X_test, columns=X_train.columns)

# Clean feature names for LightGBM compatibility
X_train_res.columns = X_train_res.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

print("Resampled Train Shape:", X_train_res.shape)
print("Resampled Target Distribution:\n", pd.Series(y_train_res).value_counts())

# ── Baseline: Logistic Regression ────────────────────────────────────────────
print("\n" + "=" * 60)
print("BASELINE: LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_res, y_train_res)

y_prob_lr = lr.predict_proba(X_test)[:, 1]
y_pred_lr = (y_prob_lr >= 0.3).astype(int)

print("\nLogistic Regression Report (Threshold=0.3):")
print(classification_report(y_test, y_pred_lr))
print("Logistic Regression ROC-AUC:", round(roc_auc_score(y_test, y_prob_lr), 4))

# ── Random Forest ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_res, y_train_res)

y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = (y_prob_rf >= 0.3).astype(int)

print("\nRandom Forest Report (Threshold=0.3):")
print(classification_report(y_test, y_pred_rf))
print("Random Forest ROC-AUC:", round(roc_auc_score(y_test, y_prob_rf), 4))

# ── XGBoost with Optuna Tuning ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("XGBOOST WITH OPTUNA HYPERPARAMETER TUNING")
print("=" * 60)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "eval_metric": "logloss",
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    model.fit(X_train_res, y_train_res)
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best Parameters:", study.best_params)
print("Best ROC-AUC (Optuna):", round(study.best_value, 4))

# ── Train Final XGBoost ───────────────────────────────────────────────────────
final_xgb = XGBClassifier(
    **study.best_params,
    eval_metric="logloss",
    random_state=42
)
final_xgb.fit(X_train_res, y_train_res)

y_prob_xgb = final_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_prob_xgb >= 0.3).astype(int)

print("\nFinal XGBoost Report (Threshold=0.3):")
print(classification_report(y_test, y_pred_xgb))
print("Final XGBoost ROC-AUC:", round(roc_auc_score(y_test, y_prob_xgb), 4))

# ── Train LightGBM ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LIGHTGBM")
print("=" * 60)

lgb = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
lgb.fit(X_train_res, y_train_res)

y_prob_lgb = lgb.predict_proba(X_test)[:, 1]
y_pred_lgb = (y_prob_lgb >= 0.3).astype(int)

print("\nLightGBM Report (Threshold=0.3):")
print(classification_report(y_test, y_pred_lgb))
print("LightGBM ROC-AUC:", round(roc_auc_score(y_test, y_prob_lgb), 4))

# ── Ensemble (60% XGB + 40% LGB) ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ENSEMBLE: XGBoost (60%) + LightGBM (40%)")
print("=" * 60)

final_probs = (0.6 * y_prob_xgb) + (0.4 * y_prob_lgb)
final_preds = (final_probs >= 0.3).astype(int)

print("\nEnsemble Report (Threshold=0.3):")
print(classification_report(y_test, final_preds))
print("Ensemble ROC-AUC:", round(roc_auc_score(y_test, final_probs), 4))

# ── Save Models ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

os.makedirs("models", exist_ok=True)
joblib.dump(final_xgb, "models/xgb_advanced.pkl")
joblib.dump(lgb, "models/lgbm_model.pkl")
joblib.dump(X_train_res.columns.tolist(), "models/feature_names.pkl")
print("Saved: models/feature_names.pkl")
print("Saved: models/xgb_advanced.pkl")
print("Saved: models/lgbm_model.pkl")

# ── Feature Count Check (important for API) ───────────────────────────────────
print(f"\n>>> Feature count used by models: {X_train_res.shape[1]}")
print(">>> Update schemas.py validator if this number is not 202!")
print("\nTraining complete.")