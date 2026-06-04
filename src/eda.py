"""
Step 1: Exploratory Data Analysis (EDA)
Run this script to explore the Telco Churn dataset before modeling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/telco_churn.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ── Fix Known Issue: TotalCharges is string, should be float ───────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"\nMissing values after TotalCharges fix:\n{df.isnull().sum()}")

# ── Target Distribution ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("CHURN DISTRIBUTION")
print("=" * 50)
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Telco Churn - EDA", fontsize=14, fontweight="bold")

# Plot 1: Churn count
df["Churn"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue", "tomato"], edgecolor="white")
axes[0].set_title("Churn Distribution")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)

# Plot 2: Tenure by Churn
df.boxplot(column="tenure", by="Churn", ax=axes[1], patch_artist=True)
axes[1].set_title("Tenure by Churn")
axes[1].set_xlabel("Churn")
axes[1].set_ylabel("Tenure (months)")
plt.sca(axes[1])
plt.title("Tenure by Churn")

# Plot 3: Monthly Charges by Churn
df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[2], patch_artist=True)
axes[2].set_title("Monthly Charges by Churn")
axes[2].set_xlabel("Churn")
axes[2].set_ylabel("Monthly Charges ($)")
plt.sca(axes[2])
plt.title("Monthly Charges by Churn")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as eda_overview.png")

# ── Correlation Heatmap ────────────────────────────────────────────────────────
numeric_df = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
plt.figure(figsize=(6, 4))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Correlation heatmap saved as correlation_heatmap.png")
