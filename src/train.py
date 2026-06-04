"""
Step 3: Model Training & Evaluation
Trains Logistic Regression, Random Forest, and XGBoost.
Compares performance and saves the best model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import preprocess

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not installed. Run: pip install xgboost")
    XGBOOST_AVAILABLE = False


# ── Load & Preprocess Data ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, feature_names = preprocess()

# ── Define Models ──────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42
    )

# ── Train & Evaluate ───────────────────────────────────────────────────────────
results = {}
print("=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    results[name] = {"model": model, "y_pred": y_pred, "y_proba": y_proba, "auc": auc}

    print(f"\n📊 {name}  (ROC-AUC: {auc:.4f})")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ── Pick Best Model ────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
print(f"\n🏆 Best model: {best_name} (AUC: {results[best_name]['auc']:.4f})")
print("✅ Saved to models/best_model.pkl")

# ── ROC Curve Comparison ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

# ROC curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", label="Random")
axes[0].set_title("ROC Curves")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Confusion matrix for best model
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title(f"Confusion Matrix — {best_name}")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plot saved as model_evaluation.png")

# ── Feature Importance (Random Forest) ────────────────────────────────────────
if "Random Forest" in results:
    rf = results["Random Forest"]["model"]
    importances = pd.Series(rf.feature_importances_, index=feature_names).nlargest(10)

    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind="barh", color="steelblue")
    plt.title("Top 10 Feature Importances — Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Feature importance plot saved as feature_importance.png")
