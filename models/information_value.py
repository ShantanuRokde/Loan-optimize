"""Weight of Evidence (WoE) and Information Value (IV).

WoE_bin = ln( %good_bin / %bad_bin )      (positive = safer than average)
IV      = Σ (%good_bin − %bad_bin) × WoE_bin

A 0.5 count adjustment avoids infinite WoE in bins with zero goods or bads.
Computed on TRAINING data only.
Strength convention: <0.02 not predictive · 0.02–0.1 weak · 0.1–0.3 medium ·
0.3–0.5 strong · >0.5 suspiciously strong (check for leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.constants import INCOME_BAND_ORDER


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "Not predictive"
    if iv < 0.1:
        return "Weak"
    if iv < 0.3:
        return "Medium"
    if iv < 0.5:
        return "Strong"
    return "Very strong (>0.5 – confirm not leakage)"


def _bins(x: pd.Series, max_bins: int) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(x) or x.nunique() <= max_bins:
        return x.astype(str)
    q = pd.qcut(x, q=max_bins, duplicates="drop")
    return q.astype(str)


def woe_table(x: pd.Series, y: pd.Series, max_bins: int = 8) -> pd.DataFrame:
    b = _bins(x, max_bins)
    df = pd.DataFrame({"bin": b.values, "y": np.asarray(y).astype(int)})
    g = df.groupby("bin", sort=False)
    t = pd.DataFrame({"loans": g.size(), "bads": g["y"].sum()})
    t["goods"] = t["loans"] - t["bads"]
    tot_g, tot_b = t["goods"].sum(), t["bads"].sum()
    pg = (t["goods"] + 0.5) / (tot_g + 0.5 * len(t))
    pb = (t["bads"] + 0.5) / (tot_b + 0.5 * len(t))
    t["pct_good"], t["pct_bad"] = pg, pb
    t["bad_rate"] = t["bads"] / t["loans"]
    t["woe"] = np.log(pg / pb)
    t["iv_contribution"] = (pg - pb) * t["woe"]
    t = t.reset_index()
    # natural ordering for display
    if x.name == "income_band":
        t["bin"] = pd.Categorical(t["bin"], INCOME_BAND_ORDER, ordered=True)
        t = t.sort_values("bin")
        t["bin"] = t["bin"].astype(str)
    elif pd.api.types.is_numeric_dtype(x) and x.nunique() > max_bins:
        t["_left"] = t["bin"].str.extract(r"\(?\[?(-?[\d.e+]+)", expand=False).astype(float)
        t = t.sort_values("_left").drop(columns="_left")
    else:
        try:
            t = t.assign(_k=pd.to_numeric(t["bin"])).sort_values("_k").drop(columns="_k")
        except (ValueError, TypeError):
            t = t.sort_values("bin")
    return t.reset_index(drop=True)


def information_values(df: pd.DataFrame, target: str, features: list[str], max_bins: int = 8) -> list[dict]:
    rows = []
    for f in features:
        t = woe_table(df[f], df[target], max_bins)
        iv = float(t["iv_contribution"].sum())
        rows.append({"feature": f, "iv": iv, "strength": iv_strength(iv), "bins": t.to_dict(orient="records")})
    return sorted(rows, key=lambda r: r["iv"], reverse=True)
