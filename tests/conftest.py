"""Shared fixtures: isolated config + temp SQLite DB + synthetic CSVs with planted defects."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import AppConfig, reset_config, set_config  # noqa: E402
from backend.database import dispose_engine, init_db  # noqa: E402

APP_HDR = ('"loan_account_no","application_date","customer_segment","occupation","income_band","credit_score",'
           '"enquiries_6m","existing_loans","age","region","state","branch","loan_type","product_code","requested_amount"')
DISB_HDR = ('"loan_account_no","disbursal_date","disbursal_quarter","sanction_amount","loan_amount","emi_amount",'
            '"loan_tenure_months","interest_rate"')
REP_HDR = '"loan_account_no","due_month","emi_amount","paid_flag","days_past_due","outstanding_balance"'


def _q(*vals) -> str:
    return ",".join(f'"{v}"' for v in vals)


def write_synthetic(raw: Path) -> dict:
    """Create three small CSVs.

    Planted defects:
      * LN3 application row has a BOM + quoted key (normalised, WARNING)
      * LN4 application exists without disbursement           -> UNMATCHED_APPLICATION
      * LN9 disbursement exists without application           -> UNMATCHED_DISBURSEMENT (ERROR)
      * LN1 repayment row duplicated exactly                 -> DUPLICATE_ROW (ERROR)
      * LN2 repayment DPD 120 at month-on-book 1              -> INVALID_DPD (ERROR)
      * LN7 repayment row without any loan                    -> ORPHAN_REPAYMENT (ERROR)
      * LN5 application has a non-numeric credit score        -> INVALID_NUMBER (ERROR)
    """
    raw.mkdir(parents=True, exist_ok=True)
    app = [APP_HDR,
           _q("LN1", "2025-01-05", "Retail", "Salaried", "5L-10L", "  700", "1", "0", "30", "West", "Maharashtra", "Pune", "Auto Loan", "PRD1", "  500000"),
           _q("LN2", "2025-01-10", "HNI", "Professional", "25L+", "780", "0", "1", "45", "North", "Delhi", "Delhi", "Home Loan", "PRD2", "2000000"),
           '﻿' + _q("LN3", "2025-02-01", "Priority", "Business Owner", "<5L", "640", "6", "2", "38", "South", "Karnataka", "Bengaluru", "Personal Loan", "PRD3", "300000"),
           _q("LN4", "2025-02-02", "Retail", "Salaried", "10L-25L", "720", "2", "1", "29", "East", "West Bengal", "Kolkata", "Gold Loan", "PRD4", "100000"),
           _q("LN5", "2025-02-03", "Retail", "Salaried", "10L-25L", "abc", "2", "1", "29", "East", "West Bengal", "Kolkata", "Gold Loan", "PRD4", "100000")]
    disb = [DISB_HDR,
            _q("LN1", "2025-01-15", "2025Q1", "480000", "490000", "16000", "36", "10.5"),
            _q("LN2", "2025-01-20", "2025Q1", "1900000", "1950000", "25000", "120", "9.1"),
            _q("LN3", "2025-02-10", "2025Q1", "280000", "290000", "10000", "36", "15.2"),
            _q("LN5", "2025-02-12", "2025Q1", "90000", "95000", "5000", "24", "12.0"),
            _q("LN9", "2025-03-01", "2025Q1", "90000", "95000", "5000", "24", "12.0")]
    rep = [REP_HDR,
           _q("LN1", "2025-02-15", "16000", "Y", "0", "480000"),
           _q("LN1", "2025-03-15", "16000", "N", "30", "480000"),
           _q("LN1", "2025-03-15", "16000", "N", "30", "480000"),           # exact duplicate
           _q("LN1", "2025-04-15", "16000", "N", "60", "480000"),
           _q("LN2", "2025-02-20", "25000", "N", "120", "1940000"),         # impossible DPD at MOB 1
           _q("LN3", "2025-03-10", "10000", "Y", "0", "285000"),
           _q("LN3", "2025-04-10", "10000", "Y", "0", "280000"),
           _q("LN5", "2025-03-12", "5000", "Y", "0", "90000"),
           _q("LN7", "2025-03-12", "5000", "Y", "0", "90000"),              # orphan
           _q("LN9", "2025-04-01", "5000", "Y", "0", "90000")]
    files = {"application": "app.csv", "disbursement": "disb.csv", "repayment": "rep.csv"}
    for key, lines in (("application", app), ("disbursement", disb), ("repayment", rep)):
        (raw / files[key]).write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    return files


@pytest.fixture()
def synthetic_env(tmp_path):
    raw = tmp_path / "raw"
    files = write_synthetic(raw)
    cfg = AppConfig()
    cfg.datasets.raw_dir = str(raw)
    cfg.datasets.processed_dir = str(tmp_path / "processed")
    cfg.datasets.reports_dir = str(tmp_path / "reports")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.datasets.application_file = files["application"]
    cfg.datasets.disbursement_file = files["disbursement"]
    cfg.datasets.repayment_file = files["repayment"]
    cfg.database.url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    dispose_engine()
    set_config(cfg)
    init_db()
    yield cfg
    dispose_engine()
    reset_config()


@pytest.fixture()
def real_env(tmp_path):
    """Real CSVs from data/raw, isolated outputs + DB."""
    cfg = AppConfig()
    if not all(p.exists() for p in cfg.source_files().values()):
        pytest.skip("real source CSVs not present in data/raw")
    cfg.datasets.processed_dir = str(tmp_path / "processed")
    cfg.datasets.reports_dir = str(tmp_path / "reports")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.database.url = f"sqlite:///{(tmp_path / 'real.db').as_posix()}"
    dispose_engine()
    set_config(cfg)
    init_db()
    yield cfg
    dispose_engine()
    reset_config()
