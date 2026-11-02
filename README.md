# Predictive Analytics & AI Model Deployment: Customer Churn

End-to-end ML pipeline: data, preprocessing and feature engineering, multi-model training, evaluation, REST API deployment, dashboard and documentation.

## Workflow
1. `python -m src.generate_data` builds `data/customer_churn.csv` (50,120 rows, with missing values and duplicates). You can replace it with a real churn CSV that has the same columns.
2. `python -m src.train` cleans, engineers features, trains 4 models, compares them, saves `models/best_model.joblib` and charts.
3. `python -m src.dashboard` builds `outputs/dashboard.html`.
4. `python -m app.app` serves the API and web form at http://localhost:5000.

## Technologies
Python, pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer, MLP), matplotlib, seaborn, Flask, gunicorn, Docker.

## Quick start
```bash
pip install -r requirements.txt
python -m src.generate_data && python -m src.train && python -m src.dashboard
python -m app.app
pytest -q tests
```

## API
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"tenure_months":5,"contract":"Month-to-month","internet_service":"Fiber","payment_method":"Electronic check","monthly_charges":90,"support_calls":4}'
# {"churn_probability":0.9014,"risk_level":"High","will_churn":true}
```
Endpoints: `GET /` (form), `GET /health`, `POST /predict`, `POST /predict_batch` (JSON list).

## Deploy
- Docker: `docker build -t churn . && docker run -p 5000:5000 churn`
- Render / Railway / Heroku: push the repo; the `Procfile` runs `gunicorn app.app:app`.

## Results (test set, 10,000 rows)
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7342 | 0.5428 | 0.7751 |
| Neural Network (MLP) | 0.7326 | 0.5477 | 0.7715 |
| Gradient Boosting | 0.7317 | 0.5410 | 0.7707 |
| Random Forest | 0.7326 | 0.5346 | 0.7650 |

Best model: Logistic Regression. With a tuned threshold of 0.303 it reaches recall 0.77, precision 0.52, F1 0.62. See `outputs/dashboard.html` and `docs/`.

## Note on data
The dataset is synthetic (no internet was available when this was built). Linear models do well partly because the data was generated from a logistic relationship. Expect different rankings on real data.

## Structure
`src/` pipeline code, `app/` + `templates/` API, `data/`, `models/`, `outputs/` (metrics, charts, dashboard), `docs/` (architecture, report), `tests/`, `screenshots/`.
