"""
ml/train.py
Run once to train and save the model:
    python -m ml.train
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# ── Load ───────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/accepted_2007_to_2018Q4.csv", nrows=100000, low_memory=False)
print(f"Raw shape: {df.shape}")

# ── Clean ──────────────────────────────────────────────────
# Drop columns with too many missing values
threshold = 0.4
df = df[df.columns[df.isnull().mean() < threshold]]

# Drop rows missing critical columns
df.dropna(subset=["loan_status", "annual_inc", "dti"], inplace=True)

# Fix int_rate if stored as string like "14.5%"
if df["int_rate"].dtype == object:
    df["int_rate"] = df["int_rate"].str.replace("%", "").astype(float)

if "revol_util" in df.columns and df["revol_util"].dtype == object:
    df["revol_util"] = df["revol_util"].str.replace("%", "").astype(float)

# ── Target Variable ────────────────────────────────────────
default_statuses = ["Charged Off", "Default", "Late (31-120 days)"]
df["is_default"] = df["loan_status"].isin(default_statuses).astype(int)
print(f"Default rate: {df['is_default'].mean():.2%}")

# ── Feature Selection ──────────────────────────────────────
# Core financial features — highly predictive for credit risk
NUMERIC_FEATURES = [
    "loan_amnt",          # loan amount requested
    "funded_amnt",        # amount funded
    "int_rate",           # interest rate
    "installment",        # monthly payment
    "annual_inc",         # annual income
    "dti",                # debt-to-income ratio
    "delinq_2yrs",        # delinquencies in last 2 years
    "inq_last_6mths",     # credit inquiries last 6 months
    "open_acc",           # number of open credit lines
    "pub_rec",            # public derogatory records
    "revol_bal",          # revolving balance
    "revol_util",         # revolving utilization rate
    "total_acc",          # total credit lines
    "fico_range_low",     # FICO score lower bound
    "fico_range_high",    # FICO score upper bound
]

CATEGORICAL_FEATURES = ["grade", "home_ownership", "purpose"]

# Keep only columns that exist in dataset
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in df.columns]
CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c in df.columns]

print(f"\nNumeric features: {NUMERIC_FEATURES}")
print(f"Categorical features: {CATEGORICAL_FEATURES}")

# ── Encode Categorical Features ────────────────────────────
encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

ENCODED_CATEGORICAL = [c + "_encoded" for c in CATEGORICAL_FEATURES]
ALL_FEATURES = NUMERIC_FEATURES + ENCODED_CATEGORICAL

# ── Final Dataset ──────────────────────────────────────────
df = df.dropna(subset=ALL_FEATURES)
X = df[ALL_FEATURES]
y = df["is_default"]

print(f"\nFinal dataset: {X.shape}")
print(f"Features used: {ALL_FEATURES}")

# ── Train/Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ── Train XGBoost ──────────────────────────────────────────
scale = (y == 0).sum() / (y == 1).sum()  # handle class imbalance

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

print("\nTraining XGBoost...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ── Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Feature importance
importance = pd.Series(model.feature_importances_, index=ALL_FEATURES)
print("\nTop 10 Feature Importances:")
print(importance.sort_values(ascending=False).head(10))

# ── Save ───────────────────────────────────────────────────
os.makedirs("ml", exist_ok=True)

joblib.dump(model, "ml/model.pkl")
joblib.dump(ALL_FEATURES, "ml/features.pkl")
joblib.dump(encoders, "ml/encoders.pkl")
joblib.dump(NUMERIC_FEATURES, "ml/numeric_features.pkl")
joblib.dump(CATEGORICAL_FEATURES, "ml/categorical_features.pkl")

print("\n✅ Saved:")
print("   ml/model.pkl")
print("   ml/features.pkl")
print("   ml/encoders.pkl")
print(f"\nTotal features: {len(ALL_FEATURES)}")