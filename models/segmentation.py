"""Decision-tree segmentation.

Walks the fitted tree to produce:
* a node list (for the interactive tree visual) with *unweighted* train and
  validation counts obtained from `decision_path` – not from tree_.value,
  which is distorted by class weights;
* one **segment per leaf**, with its full decision path decoded into the
  rules-engine condition format and simplified;
* segment profiles (loans, defaults, rate, averages, exposure, mean PD) on the
  training set, the held-out set and all scored loans.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from models.feature_engineering import (
    FeatureSpec,
    condition_text,
    conditions_text,
    simplify_conditions,
    split_to_conditions,
)
from utils.constants import INCOME_BAND_ORDER


def _node_counts(tree: DecisionTreeClassifier, X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = tree.tree_.node_count
    if len(X) == 0:
        return np.zeros(n_nodes, dtype=int), np.zeros(n_nodes, dtype=int)
    ind = tree.decision_path(X)                       # (n_samples, n_nodes) sparse indicator
    counts = np.asarray(ind.sum(axis=0)).ravel().astype(int)
    defaults = np.asarray(ind.T @ np.asarray(y).astype(int)).ravel().astype(int)
    return counts, defaults


def extract_tree(tree: DecisionTreeClassifier, spec: FeatureSpec, X_train: pd.DataFrame, y_train: pd.Series,
                 X_valid: pd.DataFrame, y_valid: pd.Series) -> tuple[list[dict], list[dict]]:
    t = tree.tree_
    cols = list(X_train.columns)
    n_tr, d_tr = _node_counts(tree, X_train, y_train)
    n_va, d_va = _node_counts(tree, X_valid, y_valid)
    nodes: list[dict] = []
    segments: list[dict] = []
    seg_counter = 0

    def rate(d, n):
        return float(d / n) if n else None

    def walk(node: int, parent: int | None, depth: int, conds: list[dict], edge: str | None):
        nonlocal seg_counter
        is_leaf = t.children_left[node] == t.children_right[node]
        simplified = simplify_conditions(conds, spec)
        rec = {
            "node_id": int(node), "parent_id": parent, "depth": depth, "is_leaf": bool(is_leaf),
            "edge_label": edge,
            "n_train": int(n_tr[node]), "defaults_train": int(d_tr[node]), "rate_train": rate(d_tr[node], n_tr[node]),
            "n_valid": int(n_va[node]), "defaults_valid": int(d_va[node]), "rate_valid": rate(d_va[node], n_va[node]),
            "path_conditions": simplified, "path_text": conditions_text(simplified),
        }
        if is_leaf:
            seg_counter += 1
            rec["segment_id"] = seg_counter
            segments.append({"segment_id": seg_counter, "leaf_node_id": int(node), "conditions": simplified,
                             "path_text": conditions_text(simplified), "raw_conditions": conds})
            nodes.append(rec)
            return
        col = cols[t.feature[node]]
        thr = float(t.threshold[node])
        left_c, right_c = split_to_conditions(spec, col, thr)
        rec.update(split_feature=spec.column_source(col)[0], split_column=col, threshold=thr,
                   left_condition=left_c, right_condition=right_c,
                   split_text=f"{condition_text(left_c)}  |  {condition_text(right_c)}")
        nodes.append(rec)
        walk(int(t.children_left[node]), node, depth + 1, conds + [left_c], condition_text(left_c))
        walk(int(t.children_right[node]), node, depth + 1, conds + [right_c], condition_text(right_c))

    walk(0, None, 0, [], None)
    return nodes, segments


def assign_segments(leaf_nodes: pd.Series, segments: list[dict]) -> pd.Series:
    m = {s["leaf_node_id"]: s["segment_id"] for s in segments}
    return leaf_nodes.map(m).astype("Int64")


def _profile(g: pd.DataFrame) -> dict:
    n = len(g)
    y = g["target"]
    obs = y.notna()
    defaults = int((y == 1).sum())
    bands = g["income_band"].value_counts().reindex(INCOME_BAND_ORDER).fillna(0).astype(int).to_dict()
    return {
        "loans": int(n),
        "observed_loans": int(obs.sum()),
        "defaults": defaults,
        "default_rate": float(defaults / obs.sum()) if obs.sum() else None,
        "avg_credit_score": float(g["credit_score"].mean()) if n else None,
        "avg_income_band_rank": float(g["income_band"].map({b: i + 1 for i, b in enumerate(INCOME_BAND_ORDER)}).mean()) if n else None,
        "income_band_mix": bands,
        "modal_income_band": max(bands, key=bands.get) if n else None,
        "avg_enquiries": float(g["enquiries_6m"].mean()) if n else None,
        "avg_existing_loans": float(g["existing_loans"].mean()) if n else None,
        "avg_age": float(g["age"].mean()) if n else None,
        "avg_interest_rate": float(g["interest_rate"].mean()) if n else None,
        "total_loan_amount": float(g["loan_amount"].sum()),
        "default_exposure": float(g.loc[y == 1, "loan_amount"].sum()),
        "avg_predicted_pd": float(g["probability_of_default"].mean()) if n else None,
        "avg_risk_score": float(g["risk_score"].mean()) if n else None,
    }


def segment_profiles(scored: pd.DataFrame, segments: list[dict], overall_train_rate: float) -> list[dict]:
    """scored: loan-level frame with segment_id, split, target and feature columns."""
    out = []
    for s in segments:
        sid = s["segment_id"]
        g = scored[scored["segment_id"] == sid]
        tr, va = g[g["split"] == "train"], g[g["split"] == "validation"]
        ptr, pva, pall = _profile(tr), _profile(va), _profile(g)
        out.append({
            **{k: v for k, v in s.items() if k != "raw_conditions"},
            "train": ptr, "validation": pva, "all_loans": pall,
            "lift_train": (ptr["default_rate"] / overall_train_rate) if ptr["default_rate"] is not None and overall_train_rate else None,
            "lift_valid": (pva["default_rate"] / overall_train_rate) if pva["default_rate"] is not None and overall_train_rate else None,
        })
    return out
