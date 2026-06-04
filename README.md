# 📡 Customer Churn Prediction

An end-to-end machine learning pipeline that predicts whether a telecom customer will cancel their subscription. Built with Python, Scikit-learn, and Streamlit.

---

## 🎯 Problem Statement

Customer churn is one of the most costly problems for subscription-based businesses. This project builds a binary classification model to identify at-risk customers, enabling proactive retention strategies.

---

## 📊 Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` — whether a customer left in the last month (Yes/No)
- **Key features:** tenure, contract type, monthly charges, internet service, payment method

---

## 🏗️ Project Structure

```
churn-prediction/
├── data/
│   └── telco_churn.csv        # Raw dataset (download from Kaggle)
├── models/
│   ├── best_model.pkl         # Saved best-performing model
│   └── scaler.pkl             # Saved StandardScaler
├── src/
│   ├── eda.py                 # Step 1: Exploratory Data Analysis
│   ├── preprocessing.py       # Step 2: Cleaning & feature engineering
│   └── train.py               # Step 3: Training & evaluation
├── app.py                     # Step 4: Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction.git
cd churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), rename it to `telco_churn.csv`, and place it in the `data/` folder.

### 4. Run the pipeline

```bash
# Explore the data
python src/eda.py

# Train and evaluate models
python src/train.py

# Launch the app
streamlit run app.py
```

---

## 🤖 Models Compared

| Model               | ROC-AUC |
|---------------------|---------|
| Logistic Regression | ~0.84   |
| Random Forest       | ~0.83   |
| XGBoost             | ~0.85   |

> Best model is saved automatically and used in the Streamlit app.

---

## 📈 Results

- **Best ROC-AUC:** ~0.85 (XGBoost)
- **Top churn predictors:** Contract type, tenure, monthly charges, internet service
- Customers on **month-to-month contracts** with **high monthly charges** and **short tenure** are most likely to churn

---

## 🖥️ Streamlit App

The interactive web app lets you input customer details and get a real-time churn probability score.

![App Screenshot](screenshot.png)

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn, xgboost
- **Visualization** — matplotlib, seaborn
- **Deployment** — Streamlit

---

## 📬 Contact

Made by [Your Name](https://github.com/YOUR_USERNAME) · Feel free to ⭐ the repo if you found it helpful!
