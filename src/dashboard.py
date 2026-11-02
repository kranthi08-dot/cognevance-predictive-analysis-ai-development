"""Build a self-contained HTML dashboard: python -m src.dashboard"""
import base64
import glob
import json
import os

import pandas as pd


def img(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def main():
    m = json.load(open("outputs/final_metrics.json"))
    comp = pd.read_csv("outputs/model_comparison.csv")
    sp = pd.read_csv("outputs/sample_predictions.csv")
    cards = "".join(f"<div class='card'><div class='v'>{v}</div><div class='l'>{l}</div></div>" for l, v in [
        ("Best model", m["best_model"]), ("ROC-AUC", m["roc_auc"]), ("Recall", m["recall"]),
        ("Precision", m["precision"]), ("F1", m["f1"]), ("Threshold", m["threshold"])])
    charts = "".join(f"<div class='ch'><img src='{img(p)}'></div>" for p in sorted(glob.glob("outputs/charts/*.png")))
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Churn Model Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;margin:0;background:#f9fafb;color:#111827}}
header{{background:#1e3a8a;color:#fff;padding:1.2rem 2rem}}main{{padding:1.5rem 2rem;max-width:1200px;margin:auto}}
.cards{{display:flex;gap:1rem;flex-wrap:wrap}}.card{{background:#fff;border-radius:10px;padding:1rem 1.4rem;box-shadow:0 1px 3px #0002;min-width:130px}}
.v{{font-size:1.35rem;font-weight:700}}.l{{color:#6b7280;font-size:.85rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1rem;margin-top:1rem}}
.ch{{background:#fff;border-radius:10px;padding:.6rem;box-shadow:0 1px 3px #0002}}.ch img{{width:100%}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:.85rem}}th,td{{padding:.45rem .6rem;border-bottom:1px solid #e5e7eb;text-align:right}}
th:first-child,td:first-child{{text-align:left}}h2{{margin-top:2rem}}.w{{overflow-x:auto}}</style></head><body>
<header><h1 style="margin:0">Customer Churn - Predictive Analytics Dashboard</h1>
<div>Train {m['n_train']:,} / Test {m['n_test']:,} rows - churn rate {m['churn_rate']:.1%}</div></header><main>
<div class="cards">{cards}</div>
<h2>Model comparison</h2><div class="w">{comp.round(4).to_html(index=False, border=0)}</div>
<h2>Charts</h2><div class="grid">{charts}</div>
<h2>Sample predictions (test set)</h2><div class="w">{sp.to_html(index=False, border=0)}</div>
</main></body></html>"""
    open("outputs/dashboard.html", "w").write(html)
    print("Wrote outputs/dashboard.html")


if __name__ == "__main__":
    main()
