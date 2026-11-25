"""Data validation for the three source datasets.

Every check emits rows into one *exception table* with columns:
    dataset, check_name, severity, row_number, loan_account_no,
    column_name, raw_value, message, action_taken

Severity semantics
------------------
ERROR    the record cannot be trusted for analytics. It is KEPT in the
         processed tables but flagged `is_analytic_eligible = False`.
WARNING  the record is usable; the issue was repaired (e.g. key normalised)
         or is recorded for review.
INFO     observation only.

No record is ever deleted, and the raw CSV files are never modified.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analytics.data_loader import LoadedDataset
from utils.constants import (
    DATASET_APPLICATION as APP,
    DATASET_DISBURSEMENT as DISB,
    DATASET_REPAYMENT as REP,
    EXPECTED_SCHEMAS,
    KEY,
    NATURAL_KEYS,
    NUMERIC_RANGES,
    PAID_FLAG_VALUES,
    SEV_ERROR,
    SEV_INFO,
    SEV_WARNING,
)
from utils.helpers import months_between

EXC_COLUMNS = ["dataset", "check_name", "severity", "row_number", "loan_account_no",
               "column_name", "raw_value", "message", "action_taken"]

# Columns whose absence / invalidity makes a record unusable for analytics.
CRITICAL_COLUMNS = {
    APP: {"loan_account_no", "application_date", "credit_score", "income_band"},
    DISB: {"loan_account_no", "disbursal_date", "loan_amount", "loan_tenure_months"},
    REP: {"loan_account_no", "due_month", "days_past_due", "paid_flag"},
}
KEY_PATTERN = re.compile(r"^LN\d+$")
DPD_PER_MISSED_INSTALMENT = 30   # observed convention in the source data (DPD rises in steps of 30)
DPD_TOLERANCE_DAYS = 1           # allowance for 31-day months


@dataclass
class ValidationResult:
    frames: dict[str, pd.DataFrame]                     # typed frames (raw text preserved in raw_* cols)
    exceptions: pd.DataFrame
    schema_report: dict[str, dict]
    status: str                                         # PASS | PASS_WITH_WARNINGS | FAIL
    fatal_errors: list[str] = field(default_factory=list)

    def error_rows(self, dataset: str) -> set[int]:
        e = self.exceptions
        m = (e.dataset == dataset) & (e.severity == SEV_ERROR) & e.row_number.notna()
        return set(e.loc[m, "row_number"].astype(int))

    def summary(self) -> dict:
        e = self.exceptions
        by_check = (
            e.groupby(["dataset", "check_name", "severity"]).size().reset_index(name="count")
            .sort_values(["dataset", "severity", "count"], ascending=[True, True, False])
            if not e.empty else pd.DataFrame(columns=["dataset", "check_name", "severity", "count"])
        )
        return {
            "status": self.status,
            "fatal_errors": self.fatal_errors,
            "exceptions_total": int(len(e)),
            "by_severity": e.severity.value_counts().to_dict() if not e.empty else {},
            "by_check": by_check.to_dict(orient="records"),
            "schema": self.schema_report,
        }


class _Collector:
    def __init__(self) -> None:
        self.parts: list[pd.DataFrame] = []

    def add(self, dataset: str, check: str, severity: str, rows: pd.DataFrame | None = None, *,
            column: str | None = None, raw_col: str | None = None, message: str | pd.Series = "",
            action: str = "flagged", key_col: str = KEY) -> None:
        if rows is None:  # dataset-level exception
            self.parts.append(pd.DataFrame([{
                "dataset": dataset, "check_name": check, "severity": severity, "row_number": None,
                "loan_account_no": None, "column_name": column, "raw_value": None,
                "message": message, "action_taken": action}]))
            return
        if rows.empty:
            return
        out = pd.DataFrame({
            "dataset": dataset,
            "check_name": check,
            "severity": severity,
            "row_number": rows["_row_number"].values if "_row_number" in rows else None,
            "loan_account_no": rows[key_col].values if key_col in rows else None,
            "column_name": column,
            "raw_value": rows[raw_col].astype(str).values if raw_col and raw_col in rows else None,
            "message": message.values if isinstance(message, pd.Series) else message,
            "action_taken": action,
        })
        self.parts.append(out)

    def frame(self) -> pd.DataFrame:
        if not self.parts:
            return pd.DataFrame(columns=EXC_COLUMNS)
        df = pd.concat(self.parts, ignore_index=True)
        df["row_number"] = df["row_number"].astype("Int64")
        return df[EXC_COLUMNS]


# --------------------------------------------------------------- helpers
def normalise_key(raw: pd.Series) -> pd.Series:
    """Remove stray BOM characters and literal quotes introduced by file concatenation."""
    return raw.fillna("").str.replace("﻿", "", regex=False).str.strip().str.strip('"').str.strip()


def _cast(df: pd.DataFrame, dataset: str, col: str, typ: str, c: _Collector) -> None:
    raw = df[col]
    df[f"raw_{col}"] = raw
    present = raw.notna() & (raw != "")
    sev = SEV_ERROR if col in CRITICAL_COLUMNS[dataset] else SEV_WARNING
    if typ == "date":
        parsed = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
        bad = present & parsed.isna()
        c.add(dataset, "INVALID_DATE", sev, df[bad], column=col, raw_col=f"raw_{col}",
              message=f"'{col}' is not a valid YYYY-MM-DD date", action="excluded from analytics" if sev == SEV_ERROR else "set to null")
        df[col] = parsed
    elif typ in ("int", "float"):
        parsed = pd.to_numeric(raw, errors="coerce")
        bad = present & parsed.isna()
        c.add(dataset, "INVALID_NUMBER", sev, df[bad], column=col, raw_col=f"raw_{col}",
              message=f"'{col}' is not numeric", action="excluded from analytics" if sev == SEV_ERROR else "set to null")
        if typ == "int":
            frac = parsed.notna() & (parsed != np.floor(parsed))
            c.add(dataset, "NON_INTEGER", SEV_WARNING, df[frac], column=col, raw_col=f"raw_{col}",
                  message=f"'{col}' expected integer, has fractional part", action="kept as decimal")
        df[col] = parsed.astype(float)
    elif typ == "flag":
        up = raw.str.upper()
        bad = present & ~up.isin(PAID_FLAG_VALUES)
        c.add(dataset, "INVALID_FLAG", SEV_ERROR, df[bad], column=col, raw_col=f"raw_{col}",
              message=f"'{col}' must be one of {sorted(PAID_FLAG_VALUES)}", action="excluded from analytics")
        df[col] = up.where(~bad)


# ------------------------------------------------------------ per-dataset
def _validate_single(ds: LoadedDataset, c: _Collector) -> tuple[pd.DataFrame | None, dict, list[str]]:
    name = ds.name
    expected = EXPECTED_SCHEMAS[name]
    fatal: list[str] = []
    if not ds.ok:
        c.add(name, "FILE_LOAD_FAILED", SEV_ERROR, message=ds.message or "load failed", action="dataset unavailable")
        return None, {"status": "FAILED", "message": ds.message}, [f"{name}: {ds.message}"]

    df = ds.frame.copy()
    actual = [col for col in df.columns if col != "_row_number"]
    missing = [col for col in expected if col not in actual]
    extra = [col for col in actual if col not in expected]
    for col in missing:
        sev = SEV_ERROR
        c.add(name, "MISSING_COLUMN", sev, column=col, message=f"Expected column '{col}' not found", action="dataset rejected")
    for col in extra:
        c.add(name, "UNEXPECTED_COLUMN", SEV_INFO, column=col, message=f"Column '{col}' not in expected schema", action="ignored")
    schema = {
        "status": "OK" if not missing else "MISSING_COLUMNS",
        "expected_columns": list(expected), "actual_columns": actual,
        "missing_columns": missing, "unexpected_columns": extra, "rows": int(len(df)),
    }
    if missing:
        fatal.append(f"{name}: missing columns {missing}")
        return None, schema, fatal

    # --- key normalisation (BOM / quote artefacts) -------------------
    df["raw_loan_account_no"] = df[KEY]
    df[KEY] = normalise_key(df[KEY])
    changed = df[KEY] != df["raw_loan_account_no"].fillna("")
    c.add(name, "KEY_NORMALISED", SEV_WARNING, df[changed], column=KEY, raw_col="raw_loan_account_no",
          message="Account key contained an embedded BOM / literal quotes (file-concatenation artefact)",
          action="normalised key (raw value preserved)")
    empty_key = df[KEY] == ""
    c.add(name, "MISSING_KEY", SEV_ERROR, df[empty_key], column=KEY, raw_col="raw_loan_account_no",
          message="loan_account_no is empty", action="excluded from analytics")
    bad_fmt = ~empty_key & ~df[KEY].str.match(KEY_PATTERN)
    c.add(name, "KEY_FORMAT", SEV_WARNING, df[bad_fmt], column=KEY, raw_col="raw_loan_account_no",
          message="loan_account_no does not match pattern LN<digits>", action="flagged")

    # --- missing values ---------------------------------------------
    for col in expected:
        if col == KEY:
            continue
        miss = df[col].isna() | (df[col] == "")
        if miss.any():
            sev = SEV_ERROR if col in CRITICAL_COLUMNS[name] else SEV_WARNING
            c.add(name, "MISSING_VALUE", sev, df[miss], column=col,
                  message=f"'{col}' is missing", action="excluded from analytics" if sev == SEV_ERROR else "kept as null")
    schema["missing_values"] = {col: int((df[col].isna() | (df[col] == "")).sum()) for col in expected}

    # --- type casting -----------------------------------------------
    for col, typ in expected.items():
        if col == KEY or typ == "str":
            if typ == "str" and col != KEY:
                df[col] = df[col].replace("", np.nan)
            continue
        _cast(df, name, col, typ, c)

    # --- ranges -----------------------------------------------------
    for col, (lo, hi) in NUMERIC_RANGES.get(name, {}).items():
        v = df[col]
        bad = v.notna() & (((v < lo) if lo is not None else False) | ((v > hi) if hi is not None else False))
        if col == "days_past_due":
            neg = v.notna() & (v < 0)
            c.add(name, "INVALID_DPD", SEV_ERROR, df[neg], column=col, raw_col=f"raw_{col}",
                  message="days_past_due is negative", action="excluded from analytics")
            continue
        c.add(name, "OUT_OF_RANGE", SEV_WARNING, df[bad], column=col, raw_col=f"raw_{col}",
              message=f"'{col}' outside plausible range [{lo}, {hi}]", action="flagged")

    # --- duplicates -------------------------------------------------
    value_cols = list(expected)
    exact = df.duplicated(subset=value_cols, keep="first")
    c.add(name, "DUPLICATE_ROW", SEV_ERROR, df[exact], message="Exact duplicate of an earlier row",
          action="excluded from analytics (first occurrence kept)")
    nk = NATURAL_KEYS[name]
    keydup = df.duplicated(subset=nk, keep="first") & ~exact
    c.add(name, "DUPLICATE_KEY", SEV_ERROR, df[keydup],
          message=f"Duplicate natural key {nk} with different values", action="excluded from analytics (first occurrence kept)")
    schema["duplicate_rows"] = int(exact.sum())
    schema["duplicate_keys"] = int(keydup.sum())

    # --- dataset-specific -------------------------------------------
    if name == DISB:
        q = df["disbursal_date"].dt.to_period("Q").astype(str)
        mism = df["disbursal_date"].notna() & (q != df["disbursal_quarter"])
        c.add(name, "QUARTER_MISMATCH", SEV_WARNING, df[mism], column="disbursal_quarter", raw_col="disbursal_quarter",
              message="disbursal_quarter disagrees with disbursal_date", action="quarter derived from disbursal_date")
        df["disbursal_quarter_derived"] = q.where(df["disbursal_date"].notna())
        sanc = df["sanction_amount"].notna() & df["loan_amount"].notna() & (df["loan_amount"] < df["sanction_amount"] * 0.5)
        c.add(name, "AMOUNT_INCONSISTENT", SEV_INFO, df[sanc], column="loan_amount", raw_col="raw_loan_amount",
              message="loan_amount less than 50% of sanction_amount", action="flagged")
    if name == REP:
        n_no_y = (df["paid_flag"] == "N") & (df["days_past_due"] == 0)
        c.add(name, "FLAG_DPD_INCONSISTENT", SEV_WARNING, df[n_no_y], column="paid_flag", raw_col="raw_paid_flag",
              message="paid_flag = N but days_past_due = 0", action="flagged")
        y_dpd = (df["paid_flag"] == "Y") & (df["days_past_due"] > 0)
        c.add(name, "FLAG_DPD_INCONSISTENT", SEV_INFO, df[y_dpd], column="paid_flag", raw_col="raw_paid_flag",
              message="paid_flag = Y while days_past_due > 0 (late payment)", action="flagged")

    if name == APP:
        cats = {}
        for col in ("customer_segment", "occupation", "income_band", "region", "state", "branch", "loan_type", "product_code"):
            cats[col] = {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}
        schema["categorical_values"] = cats

    schema["dtypes"] = {col: expected[col] for col in expected}
    return df, schema, fatal


# ----------------------------------------------------------- cross-dataset
def _validate_cross(frames: dict[str, pd.DataFrame], c: _Collector) -> None:
    app, disb, rep = frames[APP], frames[DISB], frames[REP]
    a_keys, d_keys, r_keys = set(app[KEY]), set(disb[KEY]), set(rep[KEY])

    # referential integrity in every direction
    c.add(APP, "UNMATCHED_APPLICATION", SEV_WARNING, app[~app[KEY].isin(d_keys)],
          message="Application has no disbursement record (not a booked loan)", action="excluded from loan analytics")
    c.add(DISB, "UNMATCHED_DISBURSEMENT", SEV_ERROR, disb[~disb[KEY].isin(a_keys)],
          message="Disbursement has no application record (no customer attributes)", action="excluded from analytics")
    c.add(REP, "ORPHAN_REPAYMENT", SEV_ERROR, rep[~rep[KEY].isin(d_keys)],
          message="Repayment row has no disbursement record", action="excluded from analytics")
    no_rep = disb[~disb[KEY].isin(r_keys)]
    c.add(DISB, "NO_REPAYMENT_HISTORY", SEV_WARNING, no_rep,
          message="Disbursed loan has no repayment rows", action="kept; repayment metrics null")

    # date sequence: application -> disbursal -> first due
    ad = app[[KEY, "_row_number", "application_date"]].merge(disb[[KEY, "disbursal_date"]], on=KEY, how="inner")
    seq = ad["disbursal_date"] < ad["application_date"]
    c.add(DISB, "DATE_SEQUENCE", SEV_WARNING, ad[seq], column="disbursal_date",
          message="disbursal_date precedes application_date", action="flagged")

    d_small = disb[[KEY, "disbursal_date", "emi_amount", "loan_amount"]].rename(
        columns={"emi_amount": "_emi_d", "loan_amount": "_loan_d"})
    r = rep.merge(d_small, on=KEY, how="inner")
    r = r.sort_values([KEY, "due_month"])
    r["_mob"] = months_between(r["disbursal_date"], r["due_month"])

    before = r["due_month"].notna() & r["disbursal_date"].notna() & (r["due_month"] <= r["disbursal_date"])
    c.add(REP, "DUE_BEFORE_DISBURSAL", SEV_ERROR, r[before], column="due_month", raw_col="raw_due_month",
          message="EMI due date is on/before disbursal date", action="excluded from analytics")

    # Invalid DPD: more days past due than instalments elapsed allow
    max_possible = r["_mob"].clip(lower=0) * (DPD_PER_MISSED_INSTALMENT + DPD_TOLERANCE_DAYS)
    impossible = r["days_past_due"].notna() & (r["days_past_due"] > max_possible)
    c.add(REP, "INVALID_DPD", SEV_ERROR, r[impossible], column="days_past_due", raw_col="raw_days_past_due",
          message=("days_past_due exceeds the maximum possible for months on book (mob="
                   + r.loc[impossible, "_mob"].astype(str) + ")"),
          action="excluded from analytics")

    # DPD progression: can rise by at most 30 per elapsed month between consecutive dues
    r["_prev_dpd"] = r.groupby(KEY)["days_past_due"].shift()
    r["_prev_due"] = r.groupby(KEY)["due_month"].shift()
    gap_m = months_between(r["_prev_due"], r["due_month"])
    jump = r["_prev_dpd"].notna() & (r["days_past_due"] - r["_prev_dpd"] > gap_m * (DPD_PER_MISSED_INSTALMENT + DPD_TOLERANCE_DAYS))
    c.add(REP, "DPD_PROGRESSION", SEV_WARNING, r[jump], column="days_past_due", raw_col="raw_days_past_due",
          message=("DPD rose from " + r.loc[jump, "_prev_dpd"].astype(int).astype(str) + " to "
                   + r.loc[jump, "days_past_due"].astype(int).astype(str) + " in one instalment"),
          action="flagged (value kept)")

    # repayment gaps: missing instalment months inside a loan's history
    g = r.dropna(subset=["_mob"]).groupby(KEY)["_mob"].agg(["min", "max", "nunique"])
    gaps = g[(g["max"] - g["min"] + 1) != g["nunique"]]
    if not gaps.empty:
        rows = disb[disb[KEY].isin(gaps.index)]
        c.add(DISB, "REPAYMENT_GAP", SEV_WARNING, rows,
              message="Repayment history has missing instalment months", action="flagged; DPD max still used")

    # EMI consistency between disbursement and repayment
    emi_m = r["emi_amount"].notna() & r["_emi_d"].notna() & (r["emi_amount"].round(0) != r["_emi_d"].round(0))
    c.add(REP, "EMI_MISMATCH", SEV_INFO, r[emi_m], column="emi_amount", raw_col="raw_emi_amount",
          message="EMI differs from the disbursement EMI", action="flagged")

    # outstanding above original principal
    ob = r["outstanding_balance"].notna() & (r["outstanding_balance"] > r["_loan_d"] * 1.05)
    c.add(REP, "BALANCE_ABOVE_PRINCIPAL", SEV_WARNING, r[ob], column="outstanding_balance",
          raw_col="raw_outstanding_balance", message="Outstanding balance > 105% of loan_amount", action="flagged")


# ----------------------------------------------------------------- entry
def validate_datasets(datasets: dict[str, LoadedDataset]) -> ValidationResult:
    c = _Collector()
    frames: dict[str, pd.DataFrame] = {}
    schema: dict[str, dict] = {}
    fatal: list[str] = []
    for name in (APP, DISB, REP):
        df, sch, fat = _validate_single(datasets[name], c)
        schema[name] = sch
        fatal += fat
        if df is not None:
            frames[name] = df
    if not fatal:
        _validate_cross(frames, c)
    exc = c.frame()
    if fatal:
        status = "FAIL"
    elif (exc.severity == SEV_ERROR).any() or (exc.severity == SEV_WARNING).any():
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"
    return ValidationResult(frames=frames, exceptions=exc, schema_report=schema, status=status, fatal_errors=fatal)
