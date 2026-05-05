"""
ml/summarize_data.py
Run this ONCE after train.py to generate the RAG knowledge base:
    python -m ml.summarize_data
"""

import pandas as pd
import numpy as np
import os

print("Loading data...")
df = pd.read_csv("data/accepted_2007_to_2018Q4.csv", nrows=100000, low_memory=False)

# ── Clean (same as train.py) ───────────────────────────────
threshold = 0.4
df = df[df.columns[df.isnull().mean() < threshold]]
df.dropna(subset=["loan_status", "annual_inc", "dti"], inplace=True)

if df["int_rate"].dtype == object:
    df["int_rate"] = df["int_rate"].str.replace("%", "").astype(float)

default_statuses = ["Charged Off", "Default", "Late (31-120 days)"]
df["is_default"] = df["loan_status"].isin(default_statuses).astype(int)

# ── Generate Summary Text ──────────────────────────────────
lines = []

lines.append("# Credit Risk Analysis Report — Lending Club Dataset")
lines.append(f"Total loans analyzed: {len(df):,}")
lines.append(f"Overall default rate: {df['is_default'].mean():.2%}")
lines.append(f"Average loan amount: ${df['loan_amnt'].mean():,.2f}")
lines.append(f"Average interest rate: {df['int_rate'].mean():.2f}%")
lines.append(f"Average annual income: ${df['annual_inc'].mean():,.2f}")
lines.append(f"Average debt-to-income ratio (DTI): {df['dti'].mean():.2f}")

# VaR
returns = df["int_rate"].dropna() / 100
VaR_95 = returns.quantile(0.05)
VaR_99 = returns.quantile(0.01)
lines.append(f"\n## Value at Risk (VaR)")
lines.append(f"95% VaR (interest rate): {VaR_95:.2%}")
lines.append(f"99% VaR (interest rate): {VaR_99:.2%}")
lines.append("VaR measures the minimum loss expected in the worst 5% or 1% of scenarios.")

# Default by grade
if "grade" in df.columns:
    lines.append("\n## Default Rate by Loan Grade")
    grade_default = df.groupby("grade")["is_default"].mean().sort_index()
    for grade, rate in grade_default.items():
        lines.append(f"Grade {grade}: {rate:.2%} default rate")
    lines.append("Grade A loans are lowest risk. Grade G loans are highest risk.")

# Default by DTI bucket
lines.append("\n## Default Rate by DTI Bucket")
df["dti_bucket"] = pd.cut(
    df["dti"],
    bins=[0, 10, 20, 30, 40, 100],
    labels=["0-10", "10-20", "20-30", "30-40", "40+"]
)
dti_default = df.groupby("dti_bucket")["is_default"].mean()
for bucket, rate in dti_default.items():
    lines.append(f"DTI {bucket}: {rate:.2%} default rate")
lines.append("Higher DTI strongly correlates with higher default probability.")

# Default by interest rate bucket
lines.append("\n## Default Rate by Interest Rate Range")
df["rate_bucket"] = pd.cut(
    df["int_rate"],
    bins=[0, 8, 12, 16, 20, 100],
    labels=["<8%", "8-12%", "12-16%", "16-20%", ">20%"]
)
rate_default = df.groupby("rate_bucket")["is_default"].mean()
for bucket, rate in rate_default.items():
    lines.append(f"Interest rate {bucket}: {rate:.2%} default rate")

# Defaulter vs non-defaulter profile
lines.append("\n## Average Profile: Defaulters vs Non-Defaulters")
profile = df.groupby("is_default")[["loan_amnt", "int_rate", "annual_inc", "dti"]].mean()
lines.append("Non-defaulters:")
lines.append(f"  Avg loan amount: ${profile.loc[0, 'loan_amnt']:,.2f}")
lines.append(f"  Avg interest rate: {profile.loc[0, 'int_rate']:.2f}%")
lines.append(f"  Avg annual income: ${profile.loc[0, 'annual_inc']:,.2f}")
lines.append(f"  Avg DTI: {profile.loc[0, 'dti']:.2f}")
lines.append("Defaulters:")
lines.append(f"  Avg loan amount: ${profile.loc[1, 'loan_amnt']:,.2f}")
lines.append(f"  Avg interest rate: {profile.loc[1, 'int_rate']:.2f}%")
lines.append(f"  Avg annual income: ${profile.loc[1, 'annual_inc']:,.2f}")
lines.append(f"  Avg DTI: {profile.loc[1, 'dti']:.2f}")

# Key risk factors
lines.append("\n## Key Risk Factors for Credit Default")
lines.append("1. High DTI ratio (above 20) significantly increases default risk.")
lines.append("2. High interest rates (above 16%) correlate strongly with defaults.")
lines.append("3. Lower annual income combined with high loan amounts increases risk.")
lines.append("4. Loan grade is a strong predictor — Grade A is safest, Grade G is riskiest.")
lines.append("5. The model uses loan_amnt, int_rate, annual_inc, dti, and grade as features.")

# Model info
lines.append("\n## ML Model Information")
lines.append("Model type: XGBoost Classifier")
lines.append("Target variable: is_default (1 = default, 0 = no default)")
lines.append("Features used: loan_amnt, int_rate, annual_inc, dti, grade_encoded")
lines.append("Class imbalance handled using scale_pos_weight parameter.")
lines.append("Model evaluated using ROC-AUC score and classification report.")

# ── Save ───────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
report_path = "data/credit_risk_report.txt"

with open(report_path, "w") as f:
    f.write("\n".join(lines))

print(f"✅ Report saved to {report_path}")
print(f"Total lines: {len(lines)}")