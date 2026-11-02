"""Train, compare and save models. Run: python -m src.train"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.features import RAW_FEATURES, TARGET, build_preprocessor, clean

sns.set_theme(style="whitegrid")
OUT, CH = "outputs", "outputs/charts"


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                              alpha=1e-3, max_iter=200, random_state=42),
    }


def main(path="data/customer_churn.csv"):
    os.makedirs(CH, exist_ok=True); os.makedirs("models", exist_ok=True)
    raw = pd.read_csv(path)
    df = clean(raw)
    print(f"Rows raw={len(raw)} clean={len(df)} | churn rate={df[TARGET].mean():.3f}")
    X, y = df[RAW_FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rows, fitted, probs = [], {}, {}
    cv = StratifiedKFold(3, shuffle=True, random_state=42)
    for name, est in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        t = time.time()
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1).mean()
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append(dict(model=name, accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
                         recall=recall_score(yte, pred), f1=f1_score(yte, pred),
                         roc_auc=roc_auc_score(yte, p), cv_roc_auc=cv_auc, train_seconds=time.time() - t))
        fitted[name], probs[name] = pipe, p
        print(f"{name:22s} AUC={rows[-1]['roc_auc']:.4f} F1={rows[-1]['f1']:.4f}")
    res = pd.DataFrame(rows).set_index("model").round(4).sort_values("roc_auc", ascending=False)
    res.to_csv(f"{OUT}/model_comparison.csv")
    best = res.index[0]
    print("\n", res, "\nBest:", best)

    # pick decision threshold maximising F1 for the best model
    pb = probs[best]
    prec, rec, thr = precision_recall_curve(yte, pb)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    threshold = float(thr[np.argmax(f1s[:-1])])
    pred = (pb >= threshold).astype(int)
    final = dict(best_model=best, threshold=round(threshold, 3), accuracy=accuracy_score(yte, pred),
                 precision=precision_score(yte, pred), recall=recall_score(yte, pred),
                 f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, pb),
                 n_train=len(Xtr), n_test=len(Xte), churn_rate=float(y.mean()))
    final = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in final.items()}
    json.dump(final, open(f"{OUT}/final_metrics.json", "w"), indent=2)
    joblib.dump(fitted[best], "models/best_model.joblib")
    json.dump(dict(threshold=final["threshold"], model=best, features=RAW_FEATURES),
              open("models/model_meta.json", "w"), indent=2)

    # ---- charts
    fig, ax = plt.subplots(figsize=(7, 4.5))
    res[["accuracy", "f1", "roc_auc"]].plot.bar(ax=ax, rot=15)
    ax.set(title="Model Comparison (test set)", ylim=(0.5, 1.0)); fig.tight_layout()
    fig.savefig(f"{CH}/01_model_comparison.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for n, p in probs.items():
        fpr, tpr, _ = roc_curve(yte, p); ax.plot(fpr, tpr, label=f"{n} ({roc_auc_score(yte, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(title="ROC Curves", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{CH}/02_roc_curves.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(confusion_matrix(yte, pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    ax.set(title=f"Confusion Matrix - {best}", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(f"{CH}/03_confusion_matrix.png", dpi=140); plt.close(fig)

    samp = Xte.sample(3000, random_state=0)
    pi = permutation_importance(fitted[best], samp, yte.loc[samp.index], scoring="roc_auc",
                                n_repeats=3, random_state=0, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=RAW_FEATURES).sort_values()
    imp.to_csv(f"{OUT}/feature_importance.csv", header=["importance"])
    fig, ax = plt.subplots(figsize=(7, 4.5)); imp.plot.barh(ax=ax, color="#4C72B0")
    ax.set(title="Permutation Feature Importance (drop in AUC)"); fig.tight_layout()
    fig.savefig(f"{CH}/04_feature_importance.png", dpi=140); plt.close(fig)

    d = pd.DataFrame({"p": pb, "y": yte.values}); d["decile"] = pd.qcut(d.p.rank(method="first"), 10, labels=False) + 1
    dec = d.groupby("decile").agg(avg_pred=("p", "mean"), actual=("y", "mean"))
    fig, ax = plt.subplots(figsize=(7, 4)); dec.plot(kind="bar", ax=ax, rot=0, color=["#999", "#C44E52"])
    ax.set(title="Calibration by Risk Decile (10 = highest risk)", xlabel="Risk decile", ylabel="Churn rate")
    fig.tight_layout(); fig.savefig(f"{CH}/05_risk_deciles.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df.groupby("contract")[TARGET].mean().plot.bar(ax=axes[0], rot=0, color="#55A868")
    axes[0].set(title="Churn rate by contract", ylabel="Churn rate")
    sns.histplot(data=df, x="tenure_months", hue=TARGET, bins=36, ax=axes[1], multiple="stack")
    axes[1].set(title="Tenure distribution by churn")
    fig.tight_layout(); fig.savefig(f"{CH}/06_eda.png", dpi=140); plt.close(fig)

    # sample predictions for the dashboard
    sp = Xte.head(10).copy(); sp["churn_probability"] = pb[:10].round(3); sp["actual"] = yte.head(10).values
    sp.to_csv(f"{OUT}/sample_predictions.csv", index=False)
    print("\nFinal:", final)


if __name__ == "__main__":
    main()
