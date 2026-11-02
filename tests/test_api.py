from app.app import app

GOOD = {"tenure_months": 5, "contract": "Month-to-month", "internet_service": "Fiber",
        "payment_method": "Electronic check", "monthly_charges": 90, "support_calls": 4}
SAFE = {"tenure_months": 60, "contract": "Two year", "internet_service": "DSL",
        "payment_method": "Credit card", "monthly_charges": 45, "support_calls": 0}


def test_health():
    assert app.test_client().get("/health").get_json()["status"] == "ok"


def test_predict_risk_ordering():
    c = app.test_client()
    hi = c.post("/predict", json=GOOD).get_json()["churn_probability"]
    lo = c.post("/predict", json=SAFE).get_json()["churn_probability"]
    assert hi > lo


def test_batch_and_errors():
    c = app.test_client()
    assert len(c.post("/predict_batch", json=[GOOD, SAFE]).get_json()["predictions"]) == 2
    assert c.post("/predict", json={"tenure_months": 3}).status_code == 400
    assert c.post("/predict_batch", json={}).status_code == 400
