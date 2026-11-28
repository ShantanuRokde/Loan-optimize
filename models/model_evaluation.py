"""Model evaluation: classification metrics, ranking power, importance, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from models.feature_engineering import FeatureSpec


def classification_metrics(y: pd.Series, pred: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    m = {
        "n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()) if len(y) else None,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "predicted_positive_rate": float(np.mean(pred)) if len(y) else None,
    }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, score)
        fpr, tpr, _ = roc_curve(y, score)
        m.update(roc_auc=float(auc), gini=float(2 * auc - 1), pr_auc=float(average_precision_score(y, score)),
                 ks=float(np.max(tpr - fpr)),
                 roc_curve={"fpr": [float(x) for x in fpr], "tpr": [float(x) for x in tpr]})
    else:
        m.update(roc_auc=None, gini=None, pr_auc=None, ks=None, roc_curve=None,
                 roc_note="ROC-AUC not calculable: only one class present")
    return m


def aggregate_importance(spec: FeatureSpec, columns: list[str], importances: np.ndarray) -> list[dict]:
    """Sum one-hot column importances back to their source feature."""
    agg: dict[str, float] = {}
    for col, imp in zip(columns, importances):
        src, _, _ = spec.column_source(col)
        agg[src] = agg.get(src, 0.0) + float(imp)
    total = sum(agg.values()) or 1.0
    rows = [{"feature": f, "importance": v, "importance_share": v / total} for f, v in agg.items()]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)


def permutation_importance_by_feature(model, spec: FeatureSpec, X: pd.DataFrame, y: pd.Series,
                                      random_state: int = 42, n_repeats: int = 10) -> list[dict]:
    """Permutation importance (ROC-AUC drop) on the held-out set, grouped by source feature.
    One-hot columns of the same field are permuted together."""
    if y.nunique() < 2 or len(X) == 0:
        return []
    rng = np.random.default_rng(random_state)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    groups: dict[str, list[str]] = {}
    for c in X.columns:
        groups.setdefault(spec.column_source(c)[0], []).append(c)
    rows = []
    for f, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            idx = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].values[idx]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature": f, "auc_drop_mean": float(np.mean(drops)), "auc_drop_std": float(np.std(drops))})
    return sorted(rows, key=lambda r: r["auc_drop_mean"], reverse=True)


def calibration_table(y: pd.Series, pd_hat: pd.Series, bins: int = 5) -> list[dict]:
    """Actual vs predicted default rate by predicted-PD band."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(pd_hat)})
    if df.empty:
        return []
    try:
        df["band"] = pd.qcut(df["p"].rank(method="first"), q=min(bins, len(df)), labels=False)
    except ValueError:
        df["band"] = 0
    g = df.groupby("band")
    t = pd.DataFrame({"loans": g.size(), "predicted_pd": g["p"].mean(), "actual_rate": g["y"].mean(),
                      "defaults": g["y"].sum(), "min_pd": g["p"].min(), "max_pd": g["p"].max()}).reset_index()
    return t.to_dict(orient="records")
