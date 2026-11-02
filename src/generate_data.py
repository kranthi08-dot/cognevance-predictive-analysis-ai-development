"""Generate a large synthetic customer-churn dataset (business domain).

Used because no internet was available at build time. Replace data/customer_churn.csv
with any real churn dataset that has the same columns (e.g. Kaggle Telco Churn) and the
pipeline runs unchanged.
"""
import numpy as np
import pandas as pd


def generate(n=50_000, seed=7):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fiber", "DSL", "None"], n, p=[0.45, 0.35, 0.20])
    payment = rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n, p=[0.35, 0.25, 0.25, 0.15])
    region = rng.choice(["North", "South", "East", "West", "Central"], n)
    age = np.clip(rng.normal(44, 15, n), 18, 90).round()
    paperless = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
    num_services = rng.integers(0, 7, n)
    support_calls = rng.poisson(1.3, n)
    late_payments = rng.poisson(0.6, n)
    base = np.select([internet == "Fiber", internet == "DSL"], [70, 45], 20)
    monthly = np.round(base + 5 * num_services + rng.normal(0, 8, n), 2).clip(15)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.05, n), 2)

    z = (-1.2
         + 1.3 * (contract == "Month-to-month") - 1.0 * (contract == "Two year")
         + 0.5 * (internet == "Fiber")
         + 0.55 * (payment == "Electronic check")
         - 0.035 * tenure
         + 0.012 * (monthly - 60)
         + 0.28 * support_calls
         + 0.35 * late_payments
         + 0.25 * (paperless == "Yes")
         - 0.08 * num_services
         + 0.008 * (age < 30) * 20
         + rng.normal(0, 0.6, n))
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame(dict(
        customer_id=[f"C{100000 + i}" for i in range(n)], age=age, region=region,
        tenure_months=tenure, contract=contract, internet_service=internet,
        payment_method=payment, paperless_billing=paperless, num_services=num_services,
        monthly_charges=monthly, total_charges=total, support_calls=support_calls,
        late_payments=late_payments, churn=churn))
    # realistic dirt: missing values and a few duplicates
    df.loc[rng.random(n) < 0.012, "total_charges"] = np.nan
    df.loc[rng.random(n) < 0.02, "age"] = np.nan
    df.loc[rng.random(n) < 0.01, "region"] = np.nan
    df = pd.concat([df, df.sample(120, random_state=1)], ignore_index=True)
    return df


if __name__ == "__main__":
    d = generate()
    d.to_csv("data/customer_churn.csv", index=False)
    print(d.shape, "churn rate: %.3f" % d.churn.mean())
