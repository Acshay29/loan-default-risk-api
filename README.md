# 🏦 Loan Default Risk Prediction System

An end-to-end Machine Learning system that predicts loan default probability using ensemble models, deployed with FastAPI and integrated with a dynamic Streamlit frontend.

---

## 🚀 Project Overview

This project builds a production-style ML system to predict whether a loan applicant is likely to default.

It includes:

- Data preprocessing & feature engineering
- SMOTE balancing for imbalanced data
- Hyperparameter tuning (Optuna)
- Ensemble modeling (XGBoost + LightGBM)
- Versioned FastAPI backend (`/api/v1`)
- Dynamic Streamlit frontend
- Input validation (Pydantic)
- Unit testing (Pytest)
- Dockerized deployment

---

## 🏗 System Architecture
User (Streamlit UI)
↓
FastAPI Backend (/api/v1/predict)
↓
Input Validation (Pydantic)
↓
Model Loader (XGBoost + LightGBM)
↓
Prediction Engine
↓
Probability + Risk Classification

---

## 📊 Model Details

- Algorithms:
  - XGBoost
  - LightGBM
- Imbalance Handling:
  - SMOTE
- Hyperparameter Optimization:
  - Optuna
- Evaluation Metric:
  - ROC-AUC

> Final models stored in `/models/`

---

## 🔥 API Endpoints

### Health Check

Prediction Endpoint
GET /

Response:

json
{
  "message": "Loan Default Risk API is running"
}

Request Body:

POST /api/v1/predict

{
  "features": [210 float values]
}

Response:

{
  "probability": 0.7421,
  "prediction": 1
}

	•	prediction = 1 → High Risk of Default
	•	prediction = 0 → Low Risk

Swagger Documentation available at:
    http://127.0.0.1:8000/docs

---

## 💻 Frontend (Streamlit)

The frontend:
	•	Dynamically generates 210 feature inputs
	•	Uses feature names directly from saved model metadata
	•	Displays:
	•	Risk classification
	•	Probability progress bar
	•	Clean UX with structured layout

Run frontend:
    python -m streamlit run frontend.py

---

## 🧪 Testing

Unit tests implemented using Pytest.

Run tests:
    PYTHONPATH=. pytest

All API validation and edge cases are tested.

---

## 🐳 Docker Deployment

### Build Docker image:
    docker build -t loan-default-api .

### Run container:
    docker run -p 8000:8000 loan-default-api

---

## 📁 Project Structure

project1/
│
├── app/
│   ├── api/v1/routes.py
│   ├── main.py
│   ├── model_loader.py
│   ├── schemas.py
│
├── models/
│   ├── xgb_advanced.pkl
│   ├── lgbm_model.pkl
│   ├── feature_names.pkl
│
├── tests/
│   ├── test_api.py
│
├── frontend.py
├── train_advanced.py
├── Dockerfile
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

	1.	Clone repository
	2.	Create virtual environment:
        python -m venv venv
        source venv/bin/activate
    3.	Install dependencies:
        pip install -r requirements.txt
    4.	Run backend:
        uvicorn app.main:app --reload
    5.	Run frontend:
        python -m streamlit run frontend.py

---

## 🎯 Key Engineering Highlights

	•	Versioned API design (/api/v1)
	•	Clean separation of concerns
	•	Centralized model loading
	•	Input validation & error handling
	•	Dynamic frontend UI
	•	Proper environment management
	•	Test-driven validation
	•	Dockerized for deployment

---

## 📌 Future Improvements

	•	Public cloud deployment (Render / Railway)
	•	Authentication & rate limiting
	•	Feature importance visualization
	•	Model monitoring & logging
	•	CI/CD integration

---

## 👨‍💻 Author

Developed as a production-style ML engineering project for internship readiness.

---

