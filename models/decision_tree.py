"""Decision-tree training with a strictly chronological split.

* Only loans from **mature** quarters of the selected cohort are used (targets
  are fully observed there).
* Earlier mature quarters  -> training
* Latest mature quarter    -> held-out validation (never seen in fitting)
* No random shuffling across time; `assert_chronological` enforces that every
  training disbursal precedes every validation disbursal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from models.feature_engineering import FeatureSpec, build_feature_spec

FEATURE_SOURCE_COLUMNS = ["credit_score", "enquiries_6m", "existing_loans", "age", "requested_amount", "loan_amount",
                          "interest_rate", "loan_tenure_months", "income_band", "customer_segment", "occupation",
                          "loan_type", "region"]


@dataclass
class ModelingData:
    cohort: str
    train: pd.DataFrame
    valid: pd.DataFrame
    unobserved: pd.DataFrame        # loans whose target is not (yet) observable – scored but never trained on
    train_quarters: list[str]
    valid_quarters: list[str]


def build_modeling_data(lm: pd.DataFrame, cohort_frame: pd.DataFrame, mature_quarters: list[str],
                        holdout_quarters: int = 1) -> ModelingData:
    if len(mature_quarters) < holdout_quarters + 1:
        raise ValueError(f"Need at least {holdout_quarters + 1} mature quarters for a chronological split; "
                         f"got {mature_quarters}")
    mq = sorted(mature_quarters)
    train_q, valid_q = mq[:-holdout_quarters], mq[-holdout_quarters:]
    cols = ["loan_account_no", "branch", "state", "product_code"] + FEATURE_SOURCE_COLUMNS
    feats = lm[[c for c in cols if c in lm.columns]]
    df = cohort_frame.merge(feats, on="loan_account_no", how="left", suffixes=("", "_lm"))
    df = df.drop(columns=[c for c in df.columns if c.endswith("_lm")])
    observed = df["target"].notna()
    train = df[observed & df["disbursal_quarter"].isin(train_q)].copy()
    valid = df[observed & df["disbursal_quarter"].isin(valid_q)].copy()
    rest = df[~df.index.isin(train.index) & ~df.index.isin(valid.index)].copy()
    train["split"], valid["split"] = "train", "validation"
    # complete-window loans in not-yet-mature quarters vs loans whose window is still open
    rest["split"] = np.where(rest["target"].notna(), "partial_quarter", "unobserved")
    assert_chronological(train, valid)
    return ModelingData(cohort_frame["cohort"].iloc[0], train, valid, rest, train_q, valid_q)


def assert_chronological(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    if len(train) and len(valid) and train["disbursal_date"].max() >= valid["disbursal_date"].min():
        raise AssertionError("Chronological split violated: training data overlaps the validation period")
    if set(train["loan_account_no"]) & set(valid["loan_account_no"]):
        raise AssertionError("Train / validation overlap in loan accounts")


@dataclass
class TrainedModels:
    spec: FeatureSpec
    tree: DecisionTreeClassifier
    challenger: RandomForestClassifier | None
    params: dict


def train_models(md: ModelingData, max_depth: int = 4, min_samples_leaf: int = 50, class_weight: str = "balanced",
                 random_state: int = 42, train_challenger: bool = True, features: list[str] | None = None) -> TrainedModels:
    y = md.train["target"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Training data contains a single class – cannot fit a classifier")
    spec = build_feature_spec(md.train, features)
    X = spec.transform(md.train)
    cw = None if class_weight == "none" else class_weight
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, class_weight=cw,
                                  random_state=random_state)
    tree.fit(X, y)
    rf = None
    if train_challenger:
        rf = RandomForestClassifier(n_estimators=300, max_depth=max(max_depth + 2, 6), min_samples_leaf=max(10, min_samples_leaf // 2),
                                    class_weight="balanced_subsample" if cw else None, random_state=random_state, n_jobs=-1)
        rf.fit(X, y)
    params = {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "class_weight": class_weight,
              "random_state": random_state, "train_challenger": train_challenger, "features": spec.source_features}
    return TrainedModels(spec=spec, tree=tree, challenger=rf, params=params)


def leaf_default_rates(tree: DecisionTreeClassifier, X_train: pd.DataFrame, y_train: pd.Series) -> dict[int, float]:
    """Unweighted training default rate per leaf – the calibrated PD used downstream
    (class_weight='balanced' inflates predict_proba, so it is reported only as a risk score)."""
    leaves = tree.apply(X_train)
    s = pd.DataFrame({"leaf": leaves, "y": y_train.values}).groupby("leaf")["y"].mean()
    return s.to_dict()


def score(tm: TrainedModels, df: pd.DataFrame, leaf_pd: dict[int, float]) -> pd.DataFrame:
    X = tm.spec.transform(df)
    leaves = tm.tree.apply(X)
    out = pd.DataFrame(index=df.index)
    out["leaf_node"] = leaves
    out["risk_score"] = tm.tree.predict_proba(X)[:, 1]
    out["probability_of_default"] = pd.Series(leaves, index=df.index).map(leaf_pd).astype(float)
    out["predicted_class"] = tm.tree.predict(X).astype(int)
    if tm.challenger is not None:
        out["challenger_score"] = tm.challenger.predict_proba(X)[:, 1]
    return out
