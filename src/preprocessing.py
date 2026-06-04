"""
Step 2: Data Preprocessing
Cleans and prepares the raw Telco Churn dataset for modeling.
Returns train/test splits ready for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_clean(filepath="data/telco_churn.csv"):
    """Load raw CSV and fix known data issues."""
    df = pd.read_csv(filepath)

    # Drop customerID — not useful for modeling
    df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges: convert from string to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with nulls (only ~11 rows affected)
    df.dropna(inplace=True)

    # Encode target: Yes → 1, No → 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print(f"✅ Data loaded and cleaned. Shape: {df.shape}")
    return df


def encode_features(df):
    """Encode categorical columns using one-hot encoding."""
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]

    # Binary Yes/No → 1/0
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0,
                                    "No phone service": 0, "No internet service": 0})

    # Multi-class columns → one-hot encode
    multi_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    print(f"✅ Features encoded. Shape: {df.shape}")
    return df


def scale_and_split(df, test_size=0.2, random_state=42):
    """Scale numeric features and split into train/test sets."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Save scaler for use in the Streamlit app
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Scaler saved to models/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"✅ Split done. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def preprocess(filepath="data/telco_churn.csv"):
    """Full preprocessing pipeline."""
    df = load_and_clean(filepath)
    df = encode_features(df)
    return scale_and_split(df)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = preprocess()
    print(f"\nFeatures used ({len(feature_names)}):\n{feature_names}")
