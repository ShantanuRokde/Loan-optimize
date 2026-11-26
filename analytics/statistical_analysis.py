"""Statistics for cohort monitoring: control limits, breaches, Wilson intervals, ranking.

Ranking method (documented, deterministic)
------------------------------------------
1. A cohort is *eligible* for weakest-cohort selection only if it has at least
   ``min_mature_quarters`` mature quarters and at least ``min_defaults`` defaults
   across them (data sufficiency).
2. Eligible cohorts are ranked by the **Wilson 95% lower bound** of their pooled
   default rate over mature quarters. The lower bound rises with the default
   rate *and* with the number of observations, so it rewards "strong risk
   with sufficient evidence" without arbitrary weights.
3. Ties are broken by number of defaults, then number of mature quarters.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def control_limits(table: pd.DataFrame) -> dict:
    """Mean ± 1/2 SD limits over mature-quarter default rates, plus binomial p-chart limits."""
    m = table[table["is_mature"] & table["default_rate"].notna()]
    rates = m["default_rate"].astype(float)
    if len(rates) < 2:
        return {"n_quarters": int(len(rates)), "mean": float(rates.mean()) if len(rates) else None, "sd": None,
                "ucl_1sd": None, "lcl_1sd": None, "ucl_2sd": None, "lcl_2sd": None,
                "pooled_rate": None}
    mean, sd = float(rates.mean()), float(rates.std(ddof=1))
    pooled = float(m["defaults"].sum() / m["eligible_loans"].sum())
    return {
        "n_quarters": int(len(rates)), "mean": mean, "sd": sd, "pooled_rate": pooled,
        "ucl_1sd": mean + sd, "lcl_1sd": max(0.0, mean - sd),
        "ucl_2sd": mean + 2 * sd, "lcl_2sd": max(0.0, mean - 2 * sd),
    }


def annotate_breaches(table: pd.DataFrame, limits: dict) -> pd.DataFrame:
    """Add SD-based and p-chart breach flags to every quarter (mature quarters only are judged)."""
    t = table.copy()
    t["breach_1sd"] = False
    t["breach_2sd"] = False
    t["pchart_ucl_2sd"] = np.nan
    t["pchart_breach_2sd"] = False
    if limits.get("sd") is None:
        return t
    mature = t["is_mature"] & t["default_rate"].notna()
    t.loc[mature, "breach_1sd"] = t.loc[mature, "default_rate"] > limits["ucl_1sd"]
    t.loc[mature, "breach_2sd"] = t.loc[mature, "default_rate"] > limits["ucl_2sd"]
    p = limits["pooled_rate"]
    n = t["eligible_loans"].where(t["eligible_loans"] > 0)
    t["pchart_ucl_2sd"] = p + 2 * np.sqrt(p * (1 - p) / n)
    t.loc[mature, "pchart_breach_2sd"] = t.loc[mature, "default_rate"] > t.loc[mature, "pchart_ucl_2sd"]
    t["z_score"] = (t["default_rate"] - limits["mean"]) / limits["sd"] if limits["sd"] else np.nan
    return t


def _trend_slope(rates: pd.Series) -> float | None:
    if len(rates) < 2:
        return None
    x = np.arange(len(rates), dtype=float)
    return float(np.polyfit(x, rates.astype(float).values, 1)[0])


def summarise_cohort(code: str, table: pd.DataFrame, min_mature_quarters: int, min_defaults: int) -> dict:
    limits = control_limits(table)
    t = annotate_breaches(table, limits)
    m = t[t["is_mature"]]
    n, k = int(m["eligible_loans"].sum()), int(m["defaults"].sum())
    lo, hi = wilson_interval(k, n)
    weakest = m.loc[m["default_rate"].idxmax()] if len(m) else None
    reasons = []
    if len(m) < min_mature_quarters:
        reasons.append(f"only {len(m)} mature quarter(s); {min_mature_quarters} required")
    if k < min_defaults:
        reasons.append(f"only {k} defaults in mature quarters; {min_defaults} required")
    return {
        "cohort": code,
        "mature_quarters": m["quarter"].tolist(),
        "n_mature_quarters": int(len(m)),
        "immature_quarters": t.loc[~t["is_mature"], "quarter"].tolist(),
        "mature_loans": n,
        "mature_defaults": k,
        "pooled_default_rate": (k / n) if n else None,
        "wilson_lower": lo, "wilson_upper": hi,
        "mean_quarter_rate": limits.get("mean"), "sd_quarter_rate": limits.get("sd"),
        "weakest_quarter": None if weakest is None else weakest["quarter"],
        "weakest_quarter_rate": None if weakest is None else float(weakest["default_rate"]),
        "latest_mature_quarter": m["quarter"].iloc[-1] if len(m) else None,
        "latest_mature_rate": float(m["default_rate"].iloc[-1]) if len(m) else None,
        "trend_slope_per_quarter": _trend_slope(m["default_rate"]),
        "breaches_1sd": int(t["breach_1sd"].sum()),
        "breaches_2sd": int(t["breach_2sd"].sum()),
        "pchart_breaches_2sd": int(t["pchart_breach_2sd"].sum()),
        "defaulted_amount_mature": float(m["defaulted_amount"].sum()),
        "disbursed_mature": float(m["total_disbursed"].sum()),
        "eligible_for_selection": not reasons,
        "ineligibility_reasons": reasons,
        "limits": limits,
        "_table": t,
    }


def rank_cohorts(summaries: list[dict]) -> list[dict]:
    elig = [s for s in summaries if s["eligible_for_selection"]]
    inel = [s for s in summaries if not s["eligible_for_selection"]]
    elig.sort(key=lambda s: (s["wilson_lower"] or 0, s["mature_defaults"], s["n_mature_quarters"]), reverse=True)
    inel.sort(key=lambda s: (s["pooled_default_rate"] or 0), reverse=True)
    ranked = []
    for i, s in enumerate(elig, 1):
        ranked.append({**s, "rank": i})
    for s in inel:
        ranked.append({**s, "rank": None})
    return ranked
