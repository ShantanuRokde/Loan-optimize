"""DPD cohort calculation with strict maturity logic.

Cohort code  ``{T}DPD{WW}``: a loan is *bad* when it reaches >= T days past due
on any instalment with month-on-book (MOB) <= W.

Observation-window rule (the critical part)
-------------------------------------------
A loan is **window-complete** for W only when its repayment history actually
contains the instalment at MOB W (i.e. ``max_mob_observed >= W``) — the
observation cutoff is derived from the repayment data, never the system date.

* window-complete & bad      -> target 1
* window-complete & not bad  -> target 0
* NOT window-complete        -> target undefined (NaN). Such a loan is NEVER
                                counted as good. If it has already breached it
                                is reported separately as an *early default*
                                but excluded from the cohort rate so that
                                immature quarters are not biased.

A disbursement quarter is **mature** for W only when *every* analytic-eligible
loan in it is window-complete.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from analytics.portfolio_metrics import eligible
from utils.constants import COHORTS

_COHORT_RE = re.compile(r"^(\d+)DPD(\d+)$")


def parse_cohort(code: str) -> tuple[int, int]:
    m = _COHORT_RE.match(code.strip().upper())
    if not m:
        raise ValueError(f"Invalid cohort code '{code}' (expected e.g. 30DPD12)")
    return int(m.group(1)), int(m.group(2))


def cohort_definition(code: str) -> dict:
    t, w = parse_cohort(code)
    return {"cohort": code, "dpd_threshold": t, "window_months": w,
            "definition": f"{t}+ DPD within the first {w} months after disbursement"}


def loan_cohort_frame(lm: pd.DataFrame, rh: pd.DataFrame, code: str) -> pd.DataFrame:
    """Per-loan cohort flags for one cohort definition (analytic-eligible loans only)."""
    t, w = parse_cohort(code)
    df = eligible(lm)
    r = rh[rh["is_analytic_eligible"].astype(bool) & (rh["mob"] <= w)]
    in_win = r.groupby("loan_account_no").agg(
        max_dpd_in_window=("days_past_due", "max"),
    )
    breach = r[r["days_past_due"] >= t].sort_values("mob").drop_duplicates("loan_account_no")
    breach = breach.set_index("loan_account_no")[["mob", "outstanding_balance", "due_month"]].rename(
        columns={"mob": "breach_mob", "outstanding_balance": "outstanding_at_breach", "due_month": "breach_date"})
    out = df[["loan_account_no", "disbursal_date", "disbursal_quarter", "loan_amount", "sanction_amount",
              "loan_tenure_months", "max_mob_observed", "current_outstanding"]].copy()
    out = out.join(in_win, on="loan_account_no").join(breach, on="loan_account_no")
    required = np.minimum(w, out["loan_tenure_months"].fillna(w))   # a loan cannot be observed beyond its tenure
    out["required_mob"] = required.astype(int)
    out["window_complete"] = out["max_mob_observed"].fillna(-1).astype(float) >= required
    breached = out["breach_mob"].notna()
    out["target"] = np.where(out["window_complete"], breached.astype(float), np.nan)
    out["early_default"] = breached & ~out["window_complete"]
    out["cohort"] = code
    return out


def cohort_quarter_table(lf: pd.DataFrame, code: str, cutoff: pd.Timestamp | None) -> pd.DataFrame:
    """Quarter-level statistics for one cohort from its loan frame."""
    t, w = parse_cohort(code)
    rows = []
    for q, g in lf.groupby("disbursal_quarter"):
        comp = g[g["window_complete"]]
        bad = comp[comp["target"] == 1]
        n_tot, n_el = len(g), len(comp)
        q_end = pd.Period(q, freq="Q").end_time.normalize()
        rows.append({
            "cohort": code, "quarter": q,
            "total_loans": n_tot,
            "eligible_loans": n_el,
            "immature_loans": n_tot - n_el,
            "defaults": int(len(bad)),
            "default_rate": (len(bad) / n_el) if n_el else None,
            "early_defaults": int(g["early_default"].sum()),
            "total_disbursed": float(comp["loan_amount"].sum()),
            "defaulted_amount": float(bad["loan_amount"].sum()),
            "defaulted_outstanding_at_breach": float(bad["outstanding_at_breach"].sum()),
            "exposure_rate": (bad["loan_amount"].sum() / comp["loan_amount"].sum()) if n_el else None,
            "avg_max_dpd": float(comp["max_dpd_in_window"].mean()) if n_el else None,
            "avg_breach_mob": float(bad["breach_mob"].mean()) if len(bad) else None,
            "is_mature": bool(n_tot > 0 and n_el == n_tot),
            "maturity_pct": n_el / n_tot if n_tot else 0.0,
            "observation_window_months": w,
            "dpd_threshold": t,
            "quarter_start": pd.Period(q, freq="Q").start_time.date().isoformat(),
            "quarter_end": q_end.date().isoformat(),
            "full_maturity_date": (q_end + pd.DateOffset(months=w)).date().isoformat(),
            "observation_end": None if cutoff is None else pd.Timestamp(cutoff).date().isoformat(),
        })
    return pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)


def compute_all_cohorts(lm: pd.DataFrame, rh: pd.DataFrame, cutoff: pd.Timestamp | None,
                        cohorts: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Return ({cohort: quarter_table}, {cohort: loan_frame}) for all nine definitions."""
    tables, frames = {}, {}
    for code in cohorts or COHORTS:
        lf = loan_cohort_frame(lm, rh, code)
        frames[code] = lf
        tables[code] = cohort_quarter_table(lf, code, cutoff)
    return tables, frames


def heatmap_matrix(tables: dict[str, pd.DataFrame]) -> dict:
    quarters = sorted({q for t in tables.values() for q in t["quarter"]})
    z, mature, text = [], [], []
    for code, t in tables.items():
        tt = t.set_index("quarter")
        z.append([None if q not in tt.index or tt.loc[q, "default_rate"] is None or pd.isna(tt.loc[q, "default_rate"])
                  else float(tt.loc[q, "default_rate"]) for q in quarters])
        mature.append([bool(tt.loc[q, "is_mature"]) if q in tt.index else False for q in quarters])
        text.append([f"{int(tt.loc[q, 'defaults'])}/{int(tt.loc[q, 'eligible_loans'])}" if q in tt.index else "" for q in quarters])
    return {"cohorts": list(tables), "quarters": quarters, "default_rate": z, "is_mature": mature, "counts": text}
