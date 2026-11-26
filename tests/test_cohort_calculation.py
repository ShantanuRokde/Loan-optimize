"""Day 2 tests: cohort maturity, DPD classification, statistics, portfolio KPIs, Agents 2-3."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analytics.cohort_calculation import cohort_quarter_table, compute_all_cohorts, loan_cohort_frame, parse_cohort
from analytics.delinquency import delinquency_kpis, monthly_trends
from analytics.portfolio_metrics import breakdown, portfolio_kpis
from analytics.statistical_analysis import control_limits, rank_cohorts, summarise_cohort, wilson_interval


# --------------------------------------------------------------- fixtures
def make_book(specs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build loan_master + repayment_history from compact specs.

    spec: id, disb (YYYY-MM-DD), n (instalments observed), dpd (dict mob->dpd), amount, eligible
    """
    loans, rows = [], []
    for s in specs:
        disb = pd.Timestamp(s["disb"])
        dpd = s.get("dpd", {})
        hist = []
        for mob in range(1, s["n"] + 1):
            d = float(dpd.get(mob, 0))
            hist.append(d)
            rows.append({"loan_account_no": s["id"], "due_month": disb + pd.DateOffset(months=mob), "mob": mob,
                         "days_past_due": d, "paid_flag": "N" if d else "Y", "emi_amount": 1000.0,
                         "outstanding_balance": s.get("amount", 100000.0) - 1000 * mob, "is_analytic_eligible": True,
                         "dpd_bucket": "Current" if d == 0 else ("1-30" if d <= 30 else ("31-60" if d <= 60 else ("61-90" if d <= 90 else "90+"))),
                         "due_period": (disb + pd.DateOffset(months=mob)).strftime("%Y-%m")})
        first = {t: next((m for m, d in enumerate(hist, 1) if d >= t), None) for t in (30, 60, 90)}
        loans.append({
            "loan_account_no": s["id"], "disbursal_date": disb, "disbursal_quarter": str(disb.to_period("Q")),
            "disbursal_month": disb.strftime("%Y-%m"),
            "loan_amount": s.get("amount", 100000.0), "sanction_amount": s.get("amount", 100000.0),
            "requested_amount": s.get("amount", 100000.0),
            "loan_tenure_months": s.get("tenure", 60), "interest_rate": 10.0, "credit_score": s.get("score", 700),
            "max_mob_observed": s["n"] if s["n"] else np.nan,
            "current_outstanding": (s.get("amount", 100000.0) - 1000 * s["n"]) if s["n"] else np.nan,
            "max_dpd": max(hist) if hist else np.nan, "current_dpd": hist[-1] if hist else np.nan,
            "current_dpd_bucket": None, "is_analytic_eligible": s.get("eligible", True),
            "loan_type": s.get("type", "Auto Loan"), "region": "West", "state": "X", "branch": "B",
            "customer_segment": "Retail", "occupation": "Salaried", "income_band": "5L-10L", "product_code": "P1",
            **{f"first_mob_{t}dpd": first[t] for t in (30, 60, 90)},
            **{f"ever_{t}dpd": first[t] is not None for t in (30, 60, 90)},
        })
    return pd.DataFrame(loans), pd.DataFrame(rows)


# ------------------------------------------------------------ cohort codes
def test_parse_cohort():
    assert parse_cohort("30DPD12") == (30, 12)
    assert parse_cohort("90dpd06") == (90, 6)
    with pytest.raises(ValueError):
        parse_cohort("DPD30")


# ------------------------------------------------- DPD classification
def test_breach_boundaries_threshold_and_window():
    lm, rh = make_book([
        {"id": "A", "disb": "2024-01-15", "n": 14, "dpd": {12: 30}},   # exactly 30 DPD at MOB 12 -> bad
        {"id": "B", "disb": "2024-01-15", "n": 14, "dpd": {13: 60}},   # breach at MOB 13 -> good for W12
        {"id": "C", "disb": "2024-01-15", "n": 14, "dpd": {5: 29}},    # 29 DPD < 30 -> good
        {"id": "D", "disb": "2024-01-15", "n": 14, "dpd": {3: 30, 4: 60, 5: 90}},
    ])
    lf = loan_cohort_frame(lm, rh, "30DPD12").set_index("loan_account_no")
    assert lf.loc["A", "target"] == 1 and lf.loc["A", "breach_mob"] == 12
    assert lf.loc["B", "target"] == 0
    assert lf.loc["C", "target"] == 0
    assert lf.loc["D", "target"] == 1 and lf.loc["D", "breach_mob"] == 3
    lf90 = loan_cohort_frame(lm, rh, "90DPD06").set_index("loan_account_no")
    assert lf90.loc["D", "target"] == 1 and lf90.loc["A", "target"] == 0
    assert lf.loc["D", "max_dpd_in_window"] == 90


# ------------------------------------------------------------- maturity
def test_incomplete_window_never_counted_as_good():
    lm, rh = make_book([
        {"id": "E", "disb": "2025-01-10", "n": 11},                    # 11 months observed, clean
        {"id": "F", "disb": "2025-01-10", "n": 11, "dpd": {4: 30}},    # early default, incomplete
        {"id": "G", "disb": "2025-01-10", "n": 12},                    # complete, good
    ])
    lf = loan_cohort_frame(lm, rh, "30DPD12").set_index("loan_account_no")
    assert math.isnan(lf.loc["E", "target"]) and not lf.loc["E", "window_complete"]
    assert math.isnan(lf.loc["F", "target"]) and lf.loc["F", "early_default"]
    assert lf.loc["G", "target"] == 0
    t = cohort_quarter_table(lf.reset_index(), "30DPD12", pd.Timestamp("2026-01-10"))
    row = t.iloc[0]
    assert row["eligible_loans"] == 1 and row["defaults"] == 0 and row["early_defaults"] == 1
    assert not row["is_mature"] and row["maturity_pct"] == pytest.approx(1 / 3)


def test_quarter_mature_only_when_all_loans_complete():
    lm, rh = make_book([
        {"id": "H", "disb": "2024-07-05", "n": 13},
        {"id": "I", "disb": "2024-09-28", "n": 12},
        {"id": "J", "disb": "2024-10-01", "n": 12},
        {"id": "K", "disb": "2024-12-30", "n": 11},                    # Q4 has one immature loan
    ])
    tables, _ = compute_all_cohorts(lm, rh, None, ["30DPD12"])
    t = tables["30DPD12"].set_index("quarter")
    assert t.loc["2024Q3", "is_mature"]
    assert not t.loc["2024Q4", "is_mature"]


def test_short_tenure_loan_requires_only_its_tenure():
    lm, rh = make_book([{"id": "S", "disb": "2024-01-01", "n": 6, "tenure": 6}])
    lf = loan_cohort_frame(lm, rh, "30DPD12")
    assert lf["window_complete"].iloc[0] and lf["target"].iloc[0] == 0


def test_ineligible_loans_excluded_from_cohorts():
    lm, rh = make_book([{"id": "X", "disb": "2024-01-01", "n": 14, "dpd": {2: 30}, "eligible": False},
                        {"id": "Y", "disb": "2024-01-01", "n": 14}])
    lf = loan_cohort_frame(lm, rh, "30DPD12")
    assert lf["loan_account_no"].tolist() == ["Y"]


# ------------------------------------------------------------ statistics
def test_wilson_interval_known_value():
    lo, hi = wilson_interval(205, 2522)
    assert lo == pytest.approx(0.0712, abs=5e-4) and hi == pytest.approx(0.0926, abs=5e-4)
    assert wilson_interval(0, 0) == (None, None)


def test_control_limits_and_breaches():
    t = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4", "Q5"],
                      "default_rate": [0.05, 0.05, 0.05, 0.05, 0.20],
                      "eligible_loans": [100] * 5, "defaults": [5, 5, 5, 5, 20],
                      "is_mature": [True, True, True, True, True],
                      "defaulted_amount": [0.0] * 5, "total_disbursed": [1.0] * 5})
    lim = control_limits(t)
    rates = np.array([0.05, 0.05, 0.05, 0.05, 0.20])
    assert lim["mean"] == pytest.approx(rates.mean())
    assert lim["sd"] == pytest.approx(rates.std(ddof=1))
    assert lim["ucl_2sd"] == pytest.approx(rates.mean() + 2 * rates.std(ddof=1))
    s = summarise_cohort("TEST", t, 3, 5)
    tt = s["_table"]
    assert tt.loc[4, "breach_1sd"] and not tt.loc[0, "breach_1sd"]
    assert s["weakest_quarter"] == "Q5"


def test_ranking_requires_sufficiency():
    def tbl(rates, n=500, mature=True):
        return pd.DataFrame({"quarter": [f"Q{i}" for i in range(len(rates))], "default_rate": rates,
                             "eligible_loans": [n] * len(rates), "defaults": [int(r * n) for r in rates],
                             "is_mature": [mature] * len(rates), "defaulted_amount": [0.0] * len(rates),
                             "total_disbursed": [1.0] * len(rates)})
    risky_short = summarise_cohort("R2", tbl([0.30, 0.30]), 3, 20)       # very risky, only 2 quarters
    strong = summarise_cohort("S4", tbl([0.08, 0.09, 0.08, 0.10]), 3, 20)
    weak_evidence = summarise_cohort("W3", tbl([0.12, 0.12, 0.12], n=20), 3, 20)  # 7 defaults < 20
    ranked = rank_cohorts([risky_short, strong, weak_evidence])
    assert ranked[0]["cohort"] == "S4" and ranked[0]["rank"] == 1
    assert {r["cohort"] for r in ranked if r["rank"] is None} == {"R2", "W3"}


# -------------------------------------------------------- portfolio KPIs
def test_portfolio_kpis_and_breakdowns():
    lm, rh = make_book([
        {"id": "P1", "disb": "2024-01-01", "n": 10, "amount": 200000, "dpd": {9: 30, 10: 60}, "type": "Home Loan"},
        {"id": "P2", "disb": "2024-01-01", "n": 10, "amount": 100000, "dpd": {6: 30, 7: 60, 8: 90, 9: 120, 10: 150}},
        {"id": "P3", "disb": "2024-01-01", "n": 10, "amount": 300000},
        {"id": "P4", "disb": "2024-01-01", "n": 10, "amount": 400000, "eligible": False},
    ])
    k = portfolio_kpis(lm, npa_dpd=90)
    assert k["total_loans"] == 3 and k["excluded_loans"] == 1
    assert k["total_disbursed"] == 600000
    assert k["total_outstanding"] == (200000 - 10000) + (100000 - 10000) + (300000 - 10000)
    assert k["npa_accounts"] == 1 and k["defaulted_accounts"] == 1 and k["overdue_accounts"] == 2
    assert k["npa_rate_balance"] == pytest.approx(90000 / 570000)
    d = delinquency_kpis(lm, 90)
    assert d["dpd_30plus_accounts"] == 2 and d["dpd_60plus_accounts"] == 2 and d["dpd_90plus_accounts"] == 1
    b = breakdown(lm, "loan_type")
    assert b["loans"].sum() == 3 and b["disbursed"].sum() == 600000
    tr = monthly_trends(rh).set_index("month")
    assert tr["active_accounts"].iloc[0] == 4   # repayment rows in fixture all flagged eligible


# --------------------------------------------------------- real data
def test_real_cohorts_and_agents(real_env):
    from agents.cohort_agent import CohortAnalysisAgent
    from agents.data_agent import DataIntelligenceAgent
    from agents.portfolio_agent import PortfolioMonitoringAgent
    from database import repository as repo
    from database.models import CohortResult

    ctx: dict = {}
    assert DataIntelligenceAgent().run(ctx, force=True).ok
    p = PortfolioMonitoringAgent().run(ctx)
    assert p.status == "SUCCESS", p.error
    assert p.output["total_loans"] == 5000
    c = CohortAnalysisAgent().run(ctx)
    assert c.status == "SUCCESS", c.error
    assert c.output["selected_cohort"] == "30DPD12"
    assert c.output["mature_quarters"] == ["2024Q3", "2024Q4", "2025Q1", "2025Q2"]
    assert c.output["pooled_default_rate"] == pytest.approx(205 / 2522)
    assert "30DPD18" not in c.output["eligible_cohorts"]
    n = repo.count_rows(CohortResult)
    assert n == 9 * 8
    CohortAnalysisAgent().run(ctx)
    assert repo.count_rows(CohortResult) == n     # idempotent per data hash
    # independent recomputation of one mature quarter straight from repayment rows
    rh = pd.read_parquet(real_env.processed_dir / "repayment_history.parquet")
    lm = pd.read_parquet(real_env.processed_dir / "loan_master.parquet")
    q = lm[lm.disbursal_quarter == "2024Q4"]
    bad = rh[rh.loan_account_no.isin(q.loan_account_no) & (rh.mob <= 12) & (rh.days_past_due >= 30)].loan_account_no.nunique()
    from agents.base import load_report
    t = pd.DataFrame(load_report("cohort_analysis")["quarter_tables"]["30DPD12"]).set_index("quarter")
    assert t.loc["2024Q4", "defaults"] == bad and t.loc["2024Q4", "eligible_loans"] == len(q)
