"""Delinquency analytics: DPD buckets, delinquency / NPA rates and monthly trends.

Current-state metrics use each loan's latest observed repayment row.
Monthly trends use the repayment rows due in each calendar month (one row
per active loan per month) – i.e. a month-end-style portfolio snapshot.
"""
from __future__ import annotations

import pandas as pd

from analytics.portfolio_metrics import eligible
from utils.constants import DPD_BUCKETS
from utils.helpers import safe_div

BUCKET_ORDER = [b[0] for b in DPD_BUCKETS]


def bucket_distribution(lm: pd.DataFrame) -> pd.DataFrame:
    """Current DPD bucket distribution by accounts AND outstanding balance."""
    df = eligible(lm)
    df = df[df["current_dpd_bucket"].notna()]
    g = df.groupby("current_dpd_bucket")
    out = pd.DataFrame({
        "accounts": g.size(),
        "outstanding": g["current_outstanding"].sum(),
        "loan_amount": g["loan_amount"].sum(),
    }).reindex(BUCKET_ORDER).fillna(0).reset_index().rename(columns={"current_dpd_bucket": "bucket", "index": "bucket"})
    out["account_share"] = out["accounts"] / out["accounts"].sum()
    out["outstanding_share"] = out["outstanding"] / out["outstanding"].sum()
    return out


def delinquency_kpis(lm: pd.DataFrame, npa_dpd: int = 90) -> dict:
    df = eligible(lm)
    n = len(df)
    cur = df["current_dpd"].fillna(0)
    os_ = df["current_outstanding"].fillna(0)
    total_os = float(os_.sum())
    k: dict = {"accounts": n, "outstanding": total_os}
    for label, lo, hi in (("1_30", 1, 30), ("30plus", 30, None), ("60plus", 60, None), ("90plus", 90, None)):
        m = (cur >= lo) if hi is None else ((cur >= lo) & (cur <= hi))
        k[f"dpd_{label}_accounts"] = int(m.sum())
        k[f"dpd_{label}_outstanding"] = float(os_[m].sum())
        k[f"dpd_{label}_loan_amount"] = float(df.loc[m, "loan_amount"].sum())
        k[f"dpd_{label}_rate_count"] = safe_div(m.sum(), n)
        k[f"dpd_{label}_rate_balance"] = safe_div(os_[m].sum(), total_os)
    for t in (30, 60, 90):
        e = df[f"ever_{t}dpd"].fillna(False).astype(bool)
        k[f"ever_{t}dpd_accounts"] = int(e.sum())
        k[f"ever_{t}dpd_rate"] = safe_div(e.sum(), n)
    delinquent = cur >= 1
    k["delinquency_rate_count"] = safe_div(delinquent.sum(), n)
    k["delinquency_rate_balance"] = safe_div(os_[delinquent].sum(), total_os)
    npa = cur >= npa_dpd
    k["npa_accounts"] = int(npa.sum())
    k["npa_outstanding"] = float(os_[npa].sum())
    k["npa_rate_count"] = safe_div(npa.sum(), n)
    k["npa_rate_balance"] = safe_div(os_[npa].sum(), total_os)
    dflt = df["max_dpd"].fillna(0) >= npa_dpd
    k["default_accounts"] = int(dflt.sum())
    k["default_rate"] = safe_div(dflt.sum(), n)
    k["npa_dpd_threshold"] = npa_dpd
    k["definitions"] = {
        "delinquency_rate": "accounts currently >= 1 DPD / all accounts (count) or their outstanding / total outstanding (balance)",
        "npa_proxy": f"accounts currently >= {npa_dpd} DPD",
        "default_rate": f"accounts that EVER reached {npa_dpd}+ DPD / all accounts",
    }
    return k


def monthly_trends(rh: pd.DataFrame, npa_dpd: int = 90) -> pd.DataFrame:
    """Per due month: active accounts, 30/60/90+ and NPA counts and balances, collections."""
    r = rh[rh["is_analytic_eligible"].astype(bool)].copy()
    dpd = r["days_past_due"].fillna(0)
    os_ = r["outstanding_balance"].fillna(0)
    r["_os"] = os_
    for t in (1, 30, 60, 90):
        r[f"_c{t}"] = (dpd >= t).astype(int)
        r[f"_b{t}"] = os_.where(dpd >= t, 0.0)
    r["_npa"] = (dpd >= npa_dpd).astype(int)
    r["_npab"] = os_.where(dpd >= npa_dpd, 0.0)
    r["_paid_amt"] = r["emi_amount"].where(r["paid_flag"] == "Y", 0.0)
    r["_paid"] = (r["paid_flag"] == "Y").astype(int)
    g = r.groupby("due_period")
    out = pd.DataFrame({
        "active_accounts": g.size(),
        "outstanding": g["_os"].sum(),
        "dpd1_accounts": g["_c1"].sum(), "dpd30_accounts": g["_c30"].sum(),
        "dpd60_accounts": g["_c60"].sum(), "dpd90_accounts": g["_c90"].sum(),
        "dpd30_outstanding": g["_b30"].sum(), "dpd60_outstanding": g["_b60"].sum(), "dpd90_outstanding": g["_b90"].sum(),
        "npa_accounts": g["_npa"].sum(), "npa_outstanding": g["_npab"].sum(),
        "emi_due": g["emi_amount"].sum(), "emi_collected": g["_paid_amt"].sum(),
        "instalments_due": g.size(), "instalments_paid": g["_paid"].sum(),
    }).reset_index().rename(columns={"due_period": "month"}).sort_values("month")
    n = out["active_accounts"].where(out["active_accounts"] > 0)
    out["delinquency_rate"] = out["dpd1_accounts"] / n
    out["dpd30_rate"] = out["dpd30_accounts"] / n
    out["dpd60_rate"] = out["dpd60_accounts"] / n
    out["dpd90_rate"] = out["dpd90_accounts"] / n
    out["npa_rate_count"] = out["npa_accounts"] / n
    out["npa_rate_balance"] = out["npa_outstanding"] / out["outstanding"].where(out["outstanding"] > 0)
    out["collection_efficiency"] = out["emi_collected"] / out["emi_due"].where(out["emi_due"] > 0)
    return out.reset_index(drop=True)


def dpd_distribution(rh: pd.DataFrame) -> pd.DataFrame:
    """Distribution of DPD values over all eligible repayment rows."""
    r = rh[rh["is_analytic_eligible"].astype(bool)]
    vc = r["dpd_bucket"].value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)
    return pd.DataFrame({"bucket": vc.index, "rows": vc.values})


def roll_rates(rh: pd.DataFrame) -> pd.DataFrame:
    """Bucket-to-bucket transition matrix between consecutive instalments."""
    r = rh[rh["is_analytic_eligible"].astype(bool)].sort_values(["loan_account_no", "due_month"])
    r = r.assign(next_bucket=r.groupby("loan_account_no")["dpd_bucket"].shift(-1)).dropna(subset=["next_bucket"])
    m = pd.crosstab(r["dpd_bucket"], r["next_bucket"], normalize="index").reindex(index=BUCKET_ORDER, columns=BUCKET_ORDER).fillna(0)
    return m
