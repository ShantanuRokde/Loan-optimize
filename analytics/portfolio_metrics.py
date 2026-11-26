"""Portfolio-level KPIs and business breakdowns.

Conventions (kept explicit because they are easy to confuse):
  * *count* metrics count loan accounts
  * *disbursed* = loan_amount at booking (₹)
  * *sanctioned* = sanction_amount (₹)
  * *outstanding* = outstanding_balance on each loan's latest observed repayment row (₹)
  * *default* = loan has EVER reached the NPA-proxy DPD threshold (default 90)
  * *NPA proxy* = loan is CURRENTLY at/above the NPA-proxy DPD threshold
  * *overdue* = loan currently has DPD >= overdue threshold (default 1)

The data contain no customer identifier, so "customers" equals loan accounts;
this limitation is surfaced in the output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.helpers import safe_div, to_jsonable

CREDIT_SCORE_BINS = [0, 600, 650, 700, 750, 800, 900]
CREDIT_SCORE_LABELS = ["<600", "600-649", "650-699", "700-749", "750-799", "800+"]
AMOUNT_BINS_LAKH = [0, 5, 10, 25, 50, 100, np.inf]
AMOUNT_LABELS = ["<5L", "5L-10L", "10L-25L", "25L-50L", "50L-1Cr", "1Cr+"]


def eligible(lm: pd.DataFrame) -> pd.DataFrame:
    return lm[lm["is_analytic_eligible"].astype(bool)].copy()


def add_bands(lm: pd.DataFrame) -> pd.DataFrame:
    out = lm.copy()
    out["credit_score_band"] = pd.cut(out["credit_score"], CREDIT_SCORE_BINS, right=False, labels=CREDIT_SCORE_LABELS).astype(str)
    out["loan_amount_band"] = pd.cut(out["loan_amount"] / 1e5, AMOUNT_BINS_LAKH, right=False, labels=AMOUNT_LABELS).astype(str)
    return out


def _flags(lm: pd.DataFrame, npa_dpd: int, overdue_dpd: int) -> pd.DataFrame:
    out = lm.copy()
    cur = out["current_dpd"].fillna(0)
    mx = out["max_dpd"].fillna(0)
    out["_overdue"] = cur >= overdue_dpd
    out["_npa"] = cur >= npa_dpd
    out["_default"] = mx >= npa_dpd
    out["_dpd30"] = cur >= 30
    out["_dpd60"] = cur >= 60
    out["_dpd90"] = cur >= 90
    out["_os"] = out["current_outstanding"].fillna(0)
    return out


def portfolio_kpis(lm: pd.DataFrame, npa_dpd: int = 90, overdue_dpd: int = 1) -> dict:
    df = _flags(eligible(lm), npa_dpd, overdue_dpd)
    n = len(df)
    os_total = float(df["_os"].sum())
    k = {
        "total_loans": n,
        "total_customers": int(df["loan_account_no"].nunique()),
        "customers_note": "No customer identifier in source data – customers counted as unique loan accounts.",
        "total_disbursed": float(df["loan_amount"].sum()),
        "total_sanctioned": float(df["sanction_amount"].sum()),
        "total_requested": float(df["requested_amount"].sum()),
        "total_outstanding": os_total,
        "avg_loan_amount": float(df["loan_amount"].mean()) if n else None,
        "avg_interest_rate": float(df["interest_rate"].mean()) if n else None,
        "weighted_avg_interest_rate": safe_div((df["interest_rate"] * df["loan_amount"]).sum(), df["loan_amount"].sum()),
        "avg_tenure_months": float(df["loan_tenure_months"].mean()) if n else None,
        "avg_credit_score": float(df["credit_score"].mean()) if n else None,
        "overdue_accounts": int(df["_overdue"].sum()),
        "overdue_outstanding": float(df.loc[df["_overdue"], "_os"].sum()),
        "defaulted_accounts": int(df["_default"].sum()),
        "defaulted_loan_amount": float(df.loc[df["_default"], "loan_amount"].sum()),
        "default_rate": safe_div(df["_default"].sum(), n),
        "npa_accounts": int(df["_npa"].sum()),
        "npa_outstanding": float(df.loc[df["_npa"], "_os"].sum()),
        "npa_rate_count": safe_div(df["_npa"].sum(), n),
        "npa_rate_balance": safe_div(df.loc[df["_npa"], "_os"].sum(), os_total),
        "npa_dpd_threshold": npa_dpd,
        "overdue_dpd_threshold": overdue_dpd,
        "excluded_loans": int(len(lm) - n),
    }
    return k


def breakdown(lm: pd.DataFrame, dim: str, npa_dpd: int = 90, overdue_dpd: int = 1) -> pd.DataFrame:
    df = _flags(add_bands(eligible(lm)), npa_dpd, overdue_dpd)
    g = df.groupby(dim, observed=True, dropna=False)
    out = pd.DataFrame({
        "loans": g.size(),
        "disbursed": g["loan_amount"].sum(),
        "outstanding": g["_os"].sum(),
        "avg_loan_amount": g["loan_amount"].mean(),
        "avg_interest_rate": g["interest_rate"].mean(),
        "avg_credit_score": g["credit_score"].mean(),
        "overdue_accounts": g["_overdue"].sum(),
        "dpd30_accounts": g["_dpd30"].sum(),
        "npa_accounts": g["_npa"].sum(),
        "defaulted_accounts": g["_default"].sum(),
        "ever_30dpd_accounts": g["ever_30dpd"].apply(lambda s: int(s.fillna(False).astype(bool).sum())),
    }).reset_index()
    out["default_rate"] = out["defaulted_accounts"] / out["loans"]
    out["delinquency_rate"] = out["overdue_accounts"] / out["loans"]
    out["ever_30dpd_rate"] = out["ever_30dpd_accounts"] / out["loans"]
    out["npa_balance"] = df.groupby(dim, observed=True, dropna=False).apply(
        lambda x: x.loc[x["_npa"], "_os"].sum(), include_groups=False).values
    out["npa_rate_balance"] = out["npa_balance"] / out["outstanding"].replace(0, np.nan)
    out["share_of_loans"] = out["loans"] / out["loans"].sum()
    out["share_of_disbursed"] = out["disbursed"] / out["disbursed"].sum()
    order = {"credit_score_band": CREDIT_SCORE_LABELS, "loan_amount_band": AMOUNT_LABELS}
    if dim in order:
        out[dim] = pd.Categorical(out[dim], order[dim], ordered=True)
        out = out.sort_values(dim)
        out[dim] = out[dim].astype(str)
    elif dim == "income_band":
        from utils.constants import INCOME_BAND_ORDER
        out[dim] = pd.Categorical(out[dim], INCOME_BAND_ORDER, ordered=True)
        out = out.sort_values(dim)
        out[dim] = out[dim].astype(str)
    else:
        out = out.sort_values("disbursed", ascending=False)
    return out.rename(columns={dim: "group"}).reset_index(drop=True)


BREAKDOWN_DIMENSIONS = ["loan_type", "region", "state", "branch", "customer_segment", "occupation",
                        "income_band", "credit_score_band", "loan_amount_band", "product_code"]


def all_breakdowns(lm: pd.DataFrame, npa_dpd: int = 90, overdue_dpd: int = 1) -> dict[str, list[dict]]:
    return {d: to_jsonable(breakdown(lm, d, npa_dpd, overdue_dpd)) for d in BREAKDOWN_DIMENSIONS}


def disbursement_trend(lm: pd.DataFrame) -> pd.DataFrame:
    df = eligible(lm)
    g = df.groupby("disbursal_month")
    return pd.DataFrame({
        "loans": g.size(), "disbursed": g["loan_amount"].sum(), "sanctioned": g["sanction_amount"].sum(),
        "avg_interest_rate": g["interest_rate"].mean(),
    }).reset_index().rename(columns={"disbursal_month": "month"}).sort_values("month")


def quarterly_disbursement(lm: pd.DataFrame) -> pd.DataFrame:
    df = eligible(lm)
    g = df.groupby("disbursal_quarter")
    return pd.DataFrame({"loans": g.size(), "disbursed": g["loan_amount"].sum()}).reset_index().rename(
        columns={"disbursal_quarter": "quarter"})


def histogram(values: pd.Series, bins: list | int) -> list[dict]:
    v = values.dropna()
    counts, edges = np.histogram(v, bins=bins)
    return [{"from": float(edges[i]), "to": float(edges[i + 1]), "count": int(counts[i])} for i in range(len(counts))]


def distributions(lm: pd.DataFrame) -> dict:
    df = eligible(lm)
    return {
        "credit_score": histogram(df["credit_score"], list(range(550, 901, 25))),
        "loan_amount_lakh": histogram(df["loan_amount"] / 1e5, 20),
        "outstanding_lakh": histogram(df["current_outstanding"] / 1e5, 20),
        "interest_rate": histogram(df["interest_rate"], 16),
        "max_dpd": histogram(df["max_dpd"], [0, 1, 31, 61, 91, 181, 361, 10_000]),
    }


def geography_consistency(lm: pd.DataFrame) -> dict:
    """Observation only: how branch / state / region relate in the source data."""
    df = eligible(lm)
    return {
        "states_per_branch": df.groupby("branch")["state"].nunique().to_dict(),
        "regions_per_state": df.groupby("state")["region"].nunique().to_dict(),
    }
