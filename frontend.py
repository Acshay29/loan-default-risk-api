import streamlit as st
import requests
import numpy as np
import joblib
import os

# ----------------------------
# CONFIG
# ----------------------------
API_URL = "http://127.0.0.1:8000/api/v1/predict"

st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.title("🏦 Loan Default Risk Prediction System")
st.markdown("Predict loan default probability using trained ML model.")

# ----------------------------
# LOAD FEATURE NAMES
# ----------------------------
feature_names_path = os.path.join("models", "feature_names.pkl")
feature_names = joblib.load(feature_names_path)

st.markdown("---")
st.subheader("📊 Enter Feature Values")

with st.expander("🔍 Click to Enter All 210 Features", expanded=False):
    features_input = []

    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        col = cols[i % 3]
        value = col.number_input(
            label=feature,
            value=0.0,
            format="%.5f"
        )
        features_input.append(value)

# ----------------------------
# AUTO-GENERATED INPUTS
# ----------------------------
features_input = []

cols = st.columns(3)

for i, feature in enumerate(feature_names):
    col = cols[i % 3]
    value = col.number_input(
    label=feature,
    value=0.0,
    format="%.5f",
    key=f"input_{i}"
)
    features_input.append(value)

st.markdown("---")

# ----------------------------
# PREDICT BUTTON
# ----------------------------
if st.button("🚀 Predict"):
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        prediction = result["prediction"]
        probability = result["probability"]

        if prediction == 1:
            st.error("⚠ High Risk of Default")
        else:
            st.success("✅ Low Risk of Default")

        st.progress(probability)
        st.metric("Default Probability", f"{probability:.4f}")

    except requests.exceptions.HTTPError:
        st.error("❌ Invalid input. Please check your feature values.")

    except requests.exceptions.ConnectionError:
        st.error("🚫 Backend server is not reachable.")

    except requests.exceptions.RequestException as e:
        st.error(f"⚠ Unexpected error: {e}")