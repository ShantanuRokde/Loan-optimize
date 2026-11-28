"""Day 3 tests: features, leakage guard, chronological split, tree, segments, IV/WoE, registry, Agent 4."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.decision_tree import assert_chronological, build_modeling_data, leaf_default_rates, score, train_models
from models.feature_engineering import (
    assert_no_leakage,
    build_feature_spec,
    simplify_conditions,
    split_to_conditions,
)
from models.information_value import information_values, woe_table
from models.segmentation import assign_segments, extract_tree, segment_profiles
from policy.conditions import condition_mask, conditions_mask


# ----------------------------------------------------------- synthetic book
def synthetic_book(n_per_q: int = 600, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three disbursal quarters; planted rule: enquiries_6m >= 6 AND income < 10L -> high default."""
    rng = np.random.default_rng(seed)
    rows = []
    for qi, q in enumerate(["2024Q1", "2024Q2", "2024Q3"]):
        start = pd.Period(q, "Q").start_time
        for i in range(n_per_q):
            enq = int(rng.integers(0, 12))
            band = rng.choice(["<5L", "5L-10L", "10L-25L", "25L+"])
            risky = enq >= 6 and band in ("<5L", "5L-10L")
            y = int(rng.random() < (0.45 if risky else 0.03))
            rows.append({
                "loan_account_no": f"S{qi}{i:04d}", "disbursal_date": start + pd.Timedelta(days=int(rng.integers(0, 89))),
                "disbursal_quarter": q, "credit_score": int(rng.integers(600, 850)), "enquiries_6m": enq,
                "existing_loans": int(rng.integers(0, 5)), "age": int(rng.integers(22, 60)), "income_band": band,
                "customer_segment": rng.choice(["Retail", "HNI"]), "occupation": rng.choice(["Salaried", "Professional"]),
                "loan_type": rng.choice(["Auto Loan", "Home Loan"]), "region": rng.choice(["West", "East"]),
                "branch": "B", "state": "S", "product_code": "P",
                "requested_amount": float(rng.integers(1, 50)) * 1e5, "loan_amount": float(rng.integers(1, 50)) * 1e5,
                "interest_rate": float(rng.uniform(8, 16)), "loan_tenure_months": int(rng.choice([36, 60])),
                "target": float(y), "breach_mob": 3.0 if y else np.nan, "cohort": "30DPD12",
            })
    df = pd.DataFrame(rows)
    lm = df.drop(columns=["target", "breach_mob", "cohort"])
    cf = df[["loan_account_no", "disbursal_date", "disbursal_quarter", "loan_amount", "loan_tenure_months",
             "target", "breach_mob", "cohort"]]
    return lm, cf


@pytest.fixture(scope="module")
def trained():
    lm, cf = synthetic_book()
    md = build_modeling_data(lm, cf, ["2024Q1", "2024Q2", "2024Q3"])
    tm = train_models(md, max_depth=3, min_samples_leaf=30, train_challenger=False)
    return md, tm


# ----------------------------------------------------------- leakage guard
def test_leakage_guard_rejects_outcome_and_id_fields():
    assert_no_leakage(["credit_score", "enquiries_6m", "income_band", "loan_type"])
    for bad in (["loan_account_no"], ["max_dpd"], ["current_outstanding"], ["target"], ["first_mob_30dpd"], ["n_missed"]):
        with pytest.raises(ValueError):
            assert_no_leakage(bad)


def test_model_features_exclude_forbidden(trained):
    md, tm = trained
    assert "loan_account_no" not in tm.spec.source_features
    assert not any("dpd" in c or "target" in c for c in tm.spec.columns)


# --------------------------------------------------------- split / encode
def test_chronological_split(trained):
    md, _ = trained
    assert md.train_quarters == ["2024Q1", "2024Q2"] and md.valid_quarters == ["2024Q3"]
    assert md.train["disbursal_date"].max() < md.valid["disbursal_date"].min()
    assert not set(md.train.loan_account_no) & set(md.valid.loan_account_no)
    with pytest.raises(AssertionError):
        assert_chronological(md.valid, md.train)


def test_split_requires_two_mature_quarters():
    lm, cf = synthetic_book(50)
    with pytest.raises(ValueError):
        build_modeling_data(lm, cf, ["2024Q1"])


def test_unobserved_targets_never_trained_on():
    lm, cf = synthetic_book(100)
    cf = cf.copy()
    cf.loc[cf.index[:20], "target"] = np.nan          # window not complete
    md = build_modeling_data(lm, cf, ["2024Q1", "2024Q2", "2024Q3"])
    assert md.train["target"].notna().all() and md.valid["target"].notna().all()
    assert (md.unobserved["split"] == "unobserved").sum() == 20


def test_split_condition_decoding():
    lm, _ = synthetic_book(50)
    spec = build_feature_spec(lm)
    assert split_to_conditions(spec, "enquiries_6m", 5.5) == (
        {"field": "enquiries_6m", "op": "<", "value": 6}, {"field": "enquiries_6m", "op": ">=", "value": 6})
    l, r = split_to_conditions(spec, "income_band__ord", 2.5)
    assert l["value"] == ["<5L", "5L-10L"] and r["value"] == ["10L-25L", "25L+"]
    l, r = split_to_conditions(spec, "cat__occupation__Salaried", 0.5)
    assert l == {"field": "occupation", "op": "not_in", "value": ["Salaried"]}
    assert r == {"field": "occupation", "op": "in", "value": ["Salaried"]}
    l, r = split_to_conditions(spec, "interest_rate", 12.3456789)
    assert l["op"] == "<=" and r["op"] == ">" and l["value"] == 12.3457


def test_simplify_conditions():
    lm, _ = synthetic_book(50)
    spec = build_feature_spec(lm)
    conds = [{"field": "credit_score", "op": ">=", "value": 650}, {"field": "credit_score", "op": ">=", "value": 700},
             {"field": "credit_score", "op": "<", "value": 800},
             {"field": "occupation", "op": "not_in", "value": ["Salaried"]},
             {"field": "income_band", "op": "in", "value": ["<5L", "5L-10L", "10L-25L"]},
             {"field": "income_band", "op": "in", "value": ["5L-10L", "10L-25L", "25L+"]}]
    s = simplify_conditions(conds, spec)
    assert {"field": "credit_score", "op": ">=", "value": 700} in s
    assert {"field": "credit_score", "op": "<", "value": 800} in s
    assert {"field": "occupation", "op": "in", "value": ["Professional"]} in s
    assert {"field": "income_band", "op": "in", "value": ["5L-10L", "10L-25L"]} in s


# ----------------------------------------------------------- tree + segments
def test_tree_finds_planted_pattern(trained):
    md, tm = trained
    from models.model_evaluation import aggregate_importance
    imp = aggregate_importance(tm.spec, tm.spec.columns, tm.tree.feature_importances_)
    assert {imp[0]["feature"], imp[1]["feature"]} == {"enquiries_6m", "income_band"}


def test_segments_reconcile_and_paths_match_tree(trained):
    md, tm = trained
    Xtr, Xva = tm.spec.transform(md.train), tm.spec.transform(md.valid)
    ytr, yva = md.train["target"].astype(int), md.valid["target"].astype(int)
    nodes, segments = extract_tree(tm.tree, tm.spec, Xtr, ytr, Xva, yva)
    leaf_pd = leaf_default_rates(tm.tree, Xtr, ytr)
    allf = pd.concat([md.train, md.valid])
    allf = allf.join(score(tm, allf, leaf_pd))
    allf["segment_id"] = assign_segments(allf["leaf_node"], segments)
    prof = segment_profiles(allf, segments, float(ytr.mean()))
    # sum of segment loan counts == loans assigned to leaves
    assert sum(p["train"]["loans"] for p in prof) == len(md.train)
    assert sum(p["validation"]["loans"] for p in prof) == len(md.valid)
    assert sum(p["train"]["defaults"] for p in prof) == int(ytr.sum())
    # every loan satisfies exactly its own segment's decoded path (and no other)
    masks = {s["segment_id"]: conditions_mask(allf, s["conditions"]) for s in segments}
    for sid, m in masks.items():
        assert (m == (allf["segment_id"] == sid)).all(), f"segment {sid} path disagrees with tree.apply"
    # node counts use unweighted data
    root = next(n for n in nodes if n["node_id"] == 0)
    assert root["n_train"] == len(md.train) and root["defaults_train"] == int(ytr.sum())
    # leaf PD equals the leaf's actual training default rate
    for p in prof:
        assert p["train"]["avg_predicted_pd"] == pytest.approx(p["train"]["default_rate"])


def test_condition_evaluator_missing_values_fail_closed():
    df = pd.DataFrame({"x": [1.0, None, 5.0], "c": ["a", None, "b"]})
    assert condition_mask(df, {"field": "x", "op": "<", "value": 3}).tolist() == [True, False, False]
    assert condition_mask(df, {"field": "c", "op": "not_in", "value": ["a"]}).tolist() == [False, False, True]
    with pytest.raises(ValueError):
        condition_mask(df, {"field": "x", "op": "~", "value": 1})


# ----------------------------------------------------------------- IV / WoE
def test_woe_and_iv():
    x = pd.Series(["a"] * 100 + ["b"] * 100, name="f")
    y = pd.Series([0] * 90 + [1] * 10 + [0] * 50 + [1] * 50)
    t = woe_table(x, y).set_index("bin")
    assert t.loc["a", "woe"] > 0 > t.loc["b", "woe"]           # 'a' safer than average
    iv = information_values(pd.DataFrame({"f": x, "t": y}), "t", ["f"])[0]
    pg = np.array([90.5, 50.5]) / 141; pb = np.array([10.5, 50.5]) / 61
    assert iv["iv"] == pytest.approx(float(((pg - pb) * np.log(pg / pb)).sum()))
    noise = information_values(pd.DataFrame({"f": ["a", "b"] * 100, "t": [0, 0, 1, 1] * 50}), "t", ["f"])[0]
    assert noise["iv"] < 0.02


# ---------------------------------------------------------- registry
def test_model_signature_changes_with_inputs():
    from models.model_registry import model_signature
    base = model_signature("h", "30DPD12", ["Q1"], ["Q2"], {"max_depth": 4})
    assert base == model_signature("h", "30DPD12", ["Q1"], ["Q2"], {"max_depth": 4})
    assert base != model_signature("h2", "30DPD12", ["Q1"], ["Q2"], {"max_depth": 4})
    assert base != model_signature("h", "30DPD12", ["Q1"], ["Q2"], {"max_depth": 5})
    assert base != model_signature("h", "60DPD12", ["Q1"], ["Q2"], {"max_depth": 4})


# ------------------------------------------------------------- real data
def test_real_risk_discovery_agent(real_env):
    from agents.cohort_agent import CohortAnalysisAgent
    from agents.data_agent import DataIntelligenceAgent
    from agents.risk_discovery_agent import RiskDiscoveryAgent, load_scores, segment_loans
    from database import repository as repo
    from database.models import ModelRegistry, RiskSegment

    ctx: dict = {}
    assert DataIntelligenceAgent().run(ctx, force=True).ok
    assert CohortAnalysisAgent().run(ctx).ok
    r = RiskDiscoveryAgent().run(ctx)
    assert r.status == "SUCCESS", r.error
    o = r.output
    assert o["cohort"] == "30DPD12" and o["retrained"]
    assert o["train_quarters"] == ["2024Q3", "2024Q4", "2025Q1"] and o["validation_quarters"] == ["2025Q2"]
    assert (o["train_loans"], o["validation_loans"]) == (1918, 604)
    assert o["validation_auc"] > 0.65
    assert repo.count_rows(ModelRegistry) == 1
    n_seg = repo.count_rows(RiskSegment)
    assert n_seg == o["n_segments"]
    s = load_scores()
    assert len(s) == 5000 and s["segment_id"].notna().all()
    assert s.loc[s.split == "train", "segment_id"].value_counts().sum() == 1918
    assert len(segment_loans(1, split="train")) > 0
    # rerun: same signature -> no retrain, no duplicate registry rows
    r2 = RiskDiscoveryAgent().run(ctx)
    assert r2.status == "SUCCESS" and not r2.output["retrained"]
    assert repo.count_rows(ModelRegistry) == 1 and repo.count_rows(RiskSegment) == n_seg
    # changed hyper-parameter -> new model
    r3 = RiskDiscoveryAgent().run(ctx, max_depth=3)
    assert r3.output["retrained"] and r3.output["model_id"] != o["model_id"]
