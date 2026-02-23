import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000/api/v1"

st.set_page_config(page_title="Loan Default Risk Predictor")

st.title("🏦 Loan Default Risk Predictor")
st.write("Enter feature values and click Predict.")

# ===============================
# Fetch Features Dynamically
# ===============================
try:
    response = requests.get(f"{API_BASE}/features")
    response.raise_for_status()
    feature_names = response.json()["features"]
except Exception as e:
    st.error(f"Unable to fetch feature list: {e}")
    st.stop()

# ===============================
# Generate Dynamic Inputs
# ===============================
inputs = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0, format="%.4f")
    inputs.append(value)

# ===============================
# Prediction
# ===============================
if st.button("🚀 Predict"):
    payload = {"features": inputs}

    try:
        response = requests.post(f"{API_BASE}/predict", json=payload)
        response.raise_for_status()
        result = response.json()

        probability = result["probability"]
        prediction = result["prediction"]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠ High Risk of Default")
        else:
            st.success("✅ Low Risk of Default")

        st.metric("Default Probability", f"{probability:.2%}")

    except requests.exceptions.RequestException as e:
        st.error(f"Backend error: {e}")
    except Exception:
        st.error("Invalid input. Please check feature values.")