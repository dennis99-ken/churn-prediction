"""
Step 4: Streamlit App — Churn Prediction Demo
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="centered"
)

st.title("📡 Customer Churn Predictor")
st.markdown("Predict whether a telecom customer is likely to cancel their subscription.")
st.markdown("---")

# ── Load Model & Scaler ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

if not os.path.exists("models/best_model.pkl"):
    st.error("⚠️ Model not found. Please run `python src/train.py` first.")
    st.stop()

model, scaler = load_artifacts()

# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Customer Details")

tenure            = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges   = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
total_charges     = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))

gender            = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen    = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner           = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents        = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
phone_service     = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines    = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service  = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security   = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup     = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support      = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv      = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies  = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract          = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method    = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

# ── Build Feature Vector ───────────────────────────────────────────────────────
def yes_no(val):
    return 1 if val in ["Yes", "Male", "No phone service"] else 0

def no_internet(val):
    return 0 if val == "No internet service" else (1 if val == "Yes" else 0)

features = {
    "gender":           1 if gender == "Male" else 0,
    "SeniorCitizen":    1 if senior_citizen == "Yes" else 0,
    "Partner":          yes_no(partner),
    "Dependents":       yes_no(dependents),
    "tenure":           tenure,
    "PhoneService":     yes_no(phone_service),
    "MultipleLines":    yes_no(multiple_lines),
    "OnlineSecurity":   no_internet(online_security),
    "OnlineBackup":     no_internet(online_backup),
    "DeviceProtection": no_internet(device_protection),
    "TechSupport":      no_internet(tech_support),
    "StreamingTV":      no_internet(streaming_tv),
    "StreamingMovies":  no_internet(streaming_movies),
    "PaperlessBilling": yes_no(paperless_billing),
    "MonthlyCharges":   monthly_charges,
    "TotalCharges":     total_charges,
    # InternetService one-hot (drop_first → DSL is baseline)
    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "InternetService_No":          1 if internet_service == "No" else 0,
    # Contract one-hot
    "Contract_One year":  1 if contract == "One year" else 0,
    "Contract_Two year":  1 if contract == "Two year" else 0,
    # PaymentMethod one-hot
    "PaymentMethod_Credit card (automatic)":  1 if payment_method == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check":         1 if payment_method == "Electronic check" else 0,
    "PaymentMethod_Mailed check":             1 if payment_method == "Mailed check" else 0,
}

input_df = pd.DataFrame([features])

# Scale numeric columns
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# ── Predict ────────────────────────────────────────────────────────────────────
st.subheader("Prediction")

if st.button("🔍 Predict Churn", use_container_width=True):
    prob  = model.predict_proba(input_df)[0][1]
    label = "⚠️ Likely to Churn" if prob >= 0.5 else "✅ Likely to Stay"
    color = "red" if prob >= 0.5 else "green"

    st.markdown(f"### {label}")
    st.metric("Churn Probability", f"{prob:.1%}")
    st.progress(float(prob))

    if prob >= 0.5:
        st.warning("This customer is at risk. Consider offering a retention discount or contract upgrade.")
    else:
        st.success("This customer appears loyal. Keep up the good service!")

st.markdown("---")
st.caption("Built with Scikit-learn & Streamlit | Dataset: Telco Customer Churn (Kaggle)")
