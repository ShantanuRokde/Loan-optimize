"""Build the analytical datasets from validated source frames.

* loan_master        – one row per account (application ⟗ disbursement, 1:1)
                        + repayment metrics aggregated per loan *before* joining
                        (never a raw 3-way join, which would multiply amounts).
* repayment_history  – one row per source repayment row, enriched with
                        months-on-book (MOB), DPD bucket and DQ flags.

Records that failed validation are **kept** and flagged
`is_analytic_eligible = False`; nothing is silently removed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.data_validation import ValidationResult
from utils.constants import (
    DATASET_APPLICATION as APP,
    DATASET_DISBURSEMENT as DISB,
    DATASET_REPAYMENT as REP,
    DPD_BUCKETS,
    DPD_THRESHOLDS,
    EXPECTED_SCHEMAS,
    INCOME_BAND_ORDINAL,
    INCOME_BANDS_GE_10L,
    KEY,
    SEV_ERROR,
    SEV_WARNING,
)
from utils.helpers import months_between


@dataclass
class AnalyticalDatasets:
    loan_master: pd.DataFrame
    repayment_history: pd.DataFrame
    observation_cutoff: pd.Timestamp | None
    stats: dict


def dpd_bucket(dpd: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=dpd.index, dtype="object")
    for label, lo, hi in DPD_BUCKETS:
        m = dpd >= lo if hi is None else (dpd >= lo) & (dpd <= hi)
        out[m & dpd.notna()] = label
    return out


def _issues_by_key(exc: pd.DataFrame) -> pd.DataFrame:
    if exc.empty:
        return pd.DataFrame(columns=["dq_error_count", "dq_warning_count", "dq_issues"])
    e = exc[exc.loan_account_no.notna() & (exc.loan_account_no != "")]
    g = e.groupby("loan_account_no")
    return pd.DataFrame({
        "dq_error_count": g.apply(lambda x: int((x.severity == SEV_ERROR).sum()), include_groups=False),
        "dq_warning_count": g.apply(lambda x: int((x.severity == SEV_WARNING).sum()), include_groups=False),
        "dq_issues": g["check_name"].agg(lambda s: ",".join(sorted(set(s)))),
    })


def build_analytical_datasets(vr: ValidationResult, npa_dpd: int = 90) -> AnalyticalDatasets:
    if vr.status == "FAIL":
        raise ValueError(f"Cannot build analytical datasets: {vr.fatal_errors}")
    exc = vr.exceptions
    app, disb, rep = vr.frames[APP].copy(), vr.frames[DISB].copy(), vr.frames[REP].copy()

    app_err, disb_err, rep_err = vr.error_rows(APP), vr.error_rows(DISB), vr.error_rows(REP)
    app["_err"] = app["_row_number"].isin(app_err)
    disb["_err"] = disb["_row_number"].isin(disb_err)
    rep["_err"] = rep["_row_number"].isin(rep_err)

    # Duplicate keys are flagged in the exception table; the first occurrence represents the loan.
    app1 = app.drop_duplicates(KEY, keep="first")
    disb1 = disb.drop_duplicates(KEY, keep="first")

    app_cols = [c for c in EXPECTED_SCHEMAS[APP] if c != KEY]
    disb_cols = [c for c in EXPECTED_SCHEMAS[DISB] if c != KEY]
    a = app1[[KEY, "raw_loan_account_no", "_err"] + app_cols].rename(columns={"_err": "_app_err"})
    d = disb1[[KEY, "_err", "disbursal_quarter_derived"] + disb_cols].rename(
        columns={"_err": "_disb_err", "raw_loan_account_no": "raw_key_d"})
    loans = a.merge(d, on=KEY, how="outer", indicator=True, validate="one_to_one")
    loans["match_status"] = loans["_merge"].map(
        {"both": "matched", "left_only": "application_only", "right_only": "disbursement_only"}).astype(str)
    loans = loans.drop(columns="_merge")
    # Quarter: derived from the date (source column is validated against it)
    loans["disbursal_quarter"] = loans["disbursal_quarter_derived"].fillna(loans["disbursal_quarter"])
    loans = loans.drop(columns="disbursal_quarter_derived")
    loans["disbursal_month"] = loans["disbursal_date"].dt.to_period("M").astype(str).where(loans["disbursal_date"].notna())

    # Income band mapping (documented in utils.constants; no rupee midpoints assumed)
    loans["income_band_ordinal"] = loans["income_band"].map(INCOME_BAND_ORDINAL)
    loans["income_ge_10l"] = loans["income_band"].isin(INCOME_BANDS_GE_10L).where(loans["income_band"].notna())

    # ---------------- repayment history (row level) ----------------
    d_dates = disb1[[KEY, "disbursal_date", "disbursal_quarter_derived"]]
    rh = rep.merge(d_dates, on=KEY, how="left")
    rh["has_loan_record"] = rh["disbursal_date"].notna()
    rh["mob"] = months_between(rh["disbursal_date"], rh["due_month"]).astype("Int64")
    rh["due_period"] = rh["due_month"].dt.to_period("M").astype(str).where(rh["due_month"].notna())
    rh["dpd_bucket"] = dpd_bucket(rh["days_past_due"])
    rh = rh.rename(columns={"disbursal_quarter_derived": "disbursal_quarter", "_row_number": "source_row_number"})

    worst = (exc[exc.dataset == REP].assign(_rank=lambda x: x.severity.map({"ERROR": 0, "WARNING": 1, "INFO": 2}))
             .sort_values("_rank").drop_duplicates("row_number"))
    rh = rh.merge(worst[["row_number", "check_name"]].rename(columns={"row_number": "source_row_number", "check_name": "dq_flag"}),
                  on="source_row_number", how="left")
    rh["is_analytic_eligible"] = (~rh["_err"]) & rh["has_loan_record"]

    good = rh[rh["is_analytic_eligible"]].sort_values([KEY, "due_month"])
    cutoff = good["due_month"].max() if not good.empty else None

    # ---------------- repayment aggregates (per loan) ---------------
    g = good.groupby(KEY)
    agg = pd.DataFrame({
        "n_installments": g.size(),
        "n_paid": g["paid_flag"].apply(lambda s: int((s == "Y").sum())),
        "n_missed": g["paid_flag"].apply(lambda s: int((s == "N").sum())),
        "first_due_date": g["due_month"].min(),
        "last_due_date": g["due_month"].max(),
        "max_mob_observed": g["mob"].max(),
        "max_dpd": g["days_past_due"].max(),
        "current_dpd": g["days_past_due"].last(),
        "current_outstanding": g["outstanding_balance"].last(),
        "total_emi_due": g["emi_amount"].sum(),
        "total_emi_paid": good.assign(_p=np.where(good["paid_flag"] == "Y", good["emi_amount"], 0.0)).groupby(KEY)["_p"].sum(),
        "_mob_n": g["mob"].nunique(),
        "_mob_min": g["mob"].min(),
    })
    for t in DPD_THRESHOLDS:
        hit = good[good["days_past_due"] >= t]
        agg[f"first_mob_{t}dpd"] = hit.groupby(KEY)["mob"].min()
    agg["has_repayment_gaps"] = (agg["max_mob_observed"] - agg["_mob_min"] + 1) != agg["_mob_n"]
    agg = agg.drop(columns=["_mob_n", "_mob_min"])

    loans = loans.merge(agg, left_on=KEY, right_index=True, how="left")
    for t in DPD_THRESHOLDS:
        loans[f"ever_{t}dpd"] = loans[f"first_mob_{t}dpd"].notna().where(loans["n_installments"].notna())
        loans[f"first_mob_{t}dpd"] = loans[f"first_mob_{t}dpd"].astype("Int64")
    loans["current_dpd_bucket"] = dpd_bucket(loans["current_dpd"])
    loans["is_npa"] = (loans["current_dpd"] >= npa_dpd).where(loans["current_dpd"].notna())
    loans["months_to_cutoff"] = (months_between(loans["disbursal_date"], cutoff).astype("Int64")
                                 if cutoff is not None else pd.NA)
    for c in ("n_installments", "n_paid", "n_missed", "max_mob_observed"):
        loans[c] = loans[c].astype("Int64")

    # ---------------- DQ flags & eligibility -----------------------
    issues = _issues_by_key(exc)
    loans = loans.merge(issues, left_on=KEY, right_index=True, how="left")
    loans["dq_error_count"] = loans["dq_error_count"].fillna(0).astype(int)
    loans["dq_warning_count"] = loans["dq_warning_count"].fillna(0).astype(int)
    rep_err_keys = set(rh.loc[rh["_err"], KEY])
    loans["is_analytic_eligible"] = (
        (loans["match_status"] == "matched")
        & ~loans["_app_err"].fillna(False).astype(bool)
        & ~loans["_disb_err"].fillna(False).astype(bool)
        & ~loans[KEY].isin(rep_err_keys)
    )
    loans = loans.drop(columns=["_app_err", "_disb_err", "raw_key_d"], errors="ignore")
    loans = loans.sort_values(KEY).reset_index(drop=True)

    rh = rh.drop(columns=["_err", "disbursal_date"] + [c for c in rh.columns if c.startswith("raw_")], errors="ignore")
    rh = rh.sort_values([KEY, "due_month"]).reset_index(drop=True)

    elig = loans[loans["is_analytic_eligible"]]
    stats = {
        "loans_total": int(len(loans)),
        "loans_matched": int((loans.match_status == "matched").sum()),
        "loans_application_only": int((loans.match_status == "application_only").sum()),
        "loans_disbursement_only": int((loans.match_status == "disbursement_only").sum()),
        "loans_analytic_eligible": int(len(elig)),
        "loans_excluded": int((~loans["is_analytic_eligible"]).sum()),
        "repayment_rows": int(len(rh)),
        "repayment_rows_eligible": int(rh["is_analytic_eligible"].sum()),
        "repayment_rows_orphan": int((~rh["has_loan_record"]).sum()),
        "loans_without_repayment": int(loans.loc[loans.match_status == "matched", "n_installments"].isna().sum()),
        "loans_with_repayment_gaps": int(loans["has_repayment_gaps"].fillna(False).astype(bool).sum()),
        "observation_cutoff": None if cutoff is None else cutoff.date().isoformat(),
        "disbursal_min": None if elig.empty else elig["disbursal_date"].min().date().isoformat(),
        "disbursal_max": None if elig.empty else elig["disbursal_date"].max().date().isoformat(),
        "quarters": sorted(elig["disbursal_quarter"].dropna().unique().tolist()),
        # reconciliation: amounts in loan_master equal source sums (no multiplication by the join)
        "loan_amount_sum_master": float(loans["loan_amount"].sum()),
        "loan_amount_sum_source": float(disb1["loan_amount"].sum()),
    }
    return AnalyticalDatasets(loan_master=loans, repayment_history=rh, observation_cutoff=cutoff, stats=stats)


def save_processed(ds: AnalyticalDatasets, out_dir) -> dict[str, str]:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in (("loan_master", ds.loan_master), ("repayment_history", ds.repayment_history)):
        p = out / f"{name}.parquet"
        df.to_parquet(p, index=False)
        df.to_csv(out / f"{name}.csv", index=False)
        paths[name] = str(p)
    return paths


def load_processed(out_dir) -> AnalyticalDatasets | None:
    """Fast reload of the last build (used by downstream agents)."""
    from pathlib import Path

    out = Path(out_dir)
    lm, rh = out / "loan_master.parquet", out / "repayment_history.parquet"
    if not (lm.exists() and rh.exists()):
        return None
    loans = pd.read_parquet(lm)
    rep = pd.read_parquet(rh)
    good = rep[rep["is_analytic_eligible"]]
    cutoff = good["due_month"].max() if not good.empty else None
    return AnalyticalDatasets(loan_master=loans, repayment_history=rep, observation_cutoff=cutoff, stats={})
