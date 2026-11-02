"""Cleaning, feature engineering and the sklearn preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "churn"
RAW_NUMERIC = ["age", "tenure_months", "num_services", "monthly_charges", "total_charges",
               "support_calls", "late_payments"]
RAW_CATEGORICAL = ["region", "contract", "internet_service", "payment_method", "paperless_billing"]
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL

ENG_NUMERIC = ["avg_monthly_spend", "charge_ratio", "support_per_year", "late_per_year",
               "is_new_customer", "high_support", "fiber_month_to_month", "charges_per_service"]
NUMERIC = RAW_NUMERIC + ENG_NUMERIC
CATEGORICAL = RAW_CATEGORICAL + ["tenure_bucket"]


def clean(df):
    """Drop duplicates / invalid rows (used for training only)."""
    df = df.drop_duplicates().copy()
    df = df[df.tenure_months.between(0, 120) & (df.monthly_charges > 0)]
    return df.reset_index(drop=True)


def add_features(X):
    """NaN-safe feature engineering. Works on a DataFrame of raw columns."""
    X = X.copy()
    ten = X["tenure_months"].clip(lower=1)
    X["avg_monthly_spend"] = X["total_charges"] / ten
    X["charge_ratio"] = X["monthly_charges"] / (X["avg_monthly_spend"] + 1e-6)
    X["support_per_year"] = X["support_calls"] / (ten / 12)
    X["late_per_year"] = X["late_payments"] / (ten / 12)
    X["is_new_customer"] = (X["tenure_months"] <= 6).astype(int)
    X["high_support"] = (X["support_calls"] >= 3).astype(int)
    X["fiber_month_to_month"] = ((X["internet_service"] == "Fiber") & (X["contract"] == "Month-to-month")).astype(int)
    X["charges_per_service"] = X["monthly_charges"] / (X["num_services"] + 1)
    X["tenure_bucket"] = pd.cut(X["tenure_months"], [-1, 6, 12, 24, 48, 1000],
                                labels=["0-6", "7-12", "13-24", "25-48", "49+"]).astype(object)
    return X.replace([np.inf, -np.inf], np.nan)


def build_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    ct = ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
    return Pipeline([("fe", FunctionTransformer(add_features)), ("ct", ct)])
