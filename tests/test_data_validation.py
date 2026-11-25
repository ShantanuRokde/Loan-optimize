"""Day 1 tests: ingestion, validation, joins, duplicate detection, Agent 1."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from agents.data_agent import DataIntelligenceAgent
from analytics.data_loader import load_csv, load_sources
from analytics.data_preprocessing import build_analytical_datasets
from analytics.data_validation import normalise_key, validate_datasets
from database import repository as repo
from database.models import ALL_TABLES, DataQualityLog, LoanMaster, RepaymentHistory


def _checks(vr, dataset=None):
    e = vr.exceptions
    if dataset:
        e = e[e.dataset == dataset]
    return e


# ----------------------------------------------------------- ingestion
def test_ingestion_reads_text_and_strips_padding(synthetic_env):
    ds = load_csv("application", synthetic_env.source_files()["application"])
    assert ds.ok and len(ds.frame) == 5
    assert ds.frame.loc[0, "credit_score"] == "700"          # padding stripped, still text
    assert ds.frame["_row_number"].tolist() == [1, 2, 3, 4, 5]
    assert "loan_account_no" in ds.columns                    # header BOM removed


def test_missing_file_is_reported_not_raised(tmp_path):
    ds = load_csv("application", tmp_path / "nope.csv")
    assert not ds.ok and ds.status == "FAILED" and "not found" in ds.message


def test_raw_files_are_not_modified(synthetic_env):
    files = synthetic_env.source_files()
    before = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}
    DataIntelligenceAgent().run({}, force=True)
    after = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}
    assert before == after


# ---------------------------------------------------------- validation
def test_key_normalisation():
    s = pd.Series(['﻿"LN1"', '"LN2"', " LN3 ", "LN4"])
    assert normalise_key(s).tolist() == ["LN1", "LN2", "LN3", "LN4"]


def test_validation_detects_planted_defects(synthetic_env):
    vr = validate_datasets(load_sources(synthetic_env.source_files()))
    e = vr.exceptions
    got = set(zip(e.dataset, e.check_name, e.loan_account_no))
    assert ("application", "KEY_NORMALISED", "LN3") in got
    assert ("application", "UNMATCHED_APPLICATION", "LN4") in got
    assert ("disbursement", "UNMATCHED_DISBURSEMENT", "LN9") in got
    assert ("repayment", "DUPLICATE_ROW", "LN1") in got
    assert ("repayment", "INVALID_DPD", "LN2") in got
    assert ("repayment", "ORPHAN_REPAYMENT", "LN7") in got
    assert ("application", "INVALID_NUMBER", "LN5") in got
    assert vr.status == "PASS_WITH_WARNINGS"
    # severities
    sev = dict(zip(zip(e.check_name, e.loan_account_no), e.severity))
    assert sev[("DUPLICATE_ROW", "LN1")] == "ERROR"
    assert sev[("KEY_NORMALISED", "LN3")] == "WARNING"


def test_duplicate_detection_flags_second_occurrence_only(synthetic_env):
    vr = validate_datasets(load_sources(synthetic_env.source_files()))
    d = _checks(vr, "repayment")
    dup = d[d.check_name == "DUPLICATE_ROW"]
    assert len(dup) == 1 and int(dup.row_number.iloc[0]) == 3


def test_missing_column_fails_validation(synthetic_env):
    p = synthetic_env.source_files()["disbursement"]
    df = pd.read_csv(p, dtype=str, encoding="utf-8-sig").drop(columns=["interest_rate"])
    df.to_csv(p, index=False)
    vr = validate_datasets(load_sources(synthetic_env.source_files()))
    assert vr.status == "FAIL"
    assert any("interest_rate" in f for f in vr.fatal_errors)
    with pytest.raises(ValueError):
        build_analytical_datasets(vr)


# ---------------------------------------------------------------- joins
def test_join_is_one_to_one_and_amounts_not_multiplied(synthetic_env):
    vr = validate_datasets(load_sources(synthetic_env.source_files()))
    ads = build_analytical_datasets(vr)
    lm = ads.loan_master
    assert lm.loan_account_no.is_unique
    # LN1 has 4 repayment rows – loan amount must still appear once
    assert lm.loc[lm.loan_account_no == "LN1", "loan_amount"].item() == 490000
    assert ads.stats["loan_amount_sum_master"] == ads.stats["loan_amount_sum_source"]
    status = dict(zip(lm.loan_account_no, lm.match_status))
    assert status["LN4"] == "application_only" and status["LN9"] == "disbursement_only"
    assert status["LN3"] == "matched"   # matched after key normalisation


def test_repayment_aggregation_and_eligibility(synthetic_env):
    vr = validate_datasets(load_sources(synthetic_env.source_files()))
    ads = build_analytical_datasets(vr)
    lm = ads.loan_master.set_index("loan_account_no")
    # LN1: duplicate row excluded -> 3 instalments, max DPD 60, 30+ first at MOB 2
    assert int(lm.loc["LN1", "n_installments"]) == 3
    assert lm.loc["LN1", "max_dpd"] == 60
    assert int(lm.loc["LN1", "first_mob_30dpd"]) == 2
    assert bool(lm.loc["LN1", "ever_60dpd"]) and not bool(lm.loc["LN1", "ever_90dpd"])
    # Loans with an ERROR anywhere are kept but not eligible
    assert not lm.loc["LN1", "is_analytic_eligible"]       # duplicate repayment row
    assert not lm.loc["LN2", "is_analytic_eligible"]       # invalid DPD
    assert not lm.loc["LN5", "is_analytic_eligible"]       # invalid credit score
    assert lm.loc["LN3", "is_analytic_eligible"]
    assert len(ads.loan_master) == 6                       # nothing deleted (LN1-5 + LN9)
    rh = ads.repayment_history
    assert len(rh) == 10                                   # every source repayment row retained
    assert not rh.loc[rh.loan_account_no == "LN7", "has_loan_record"].item()
    assert ads.observation_cutoff == pd.Timestamp("2025-04-15")
    assert rh.loc[(rh.loan_account_no == "LN1"), "mob"].tolist() == [1, 2, 2, 3]


# -------------------------------------------------------------- agent 1
def test_data_agent_persists_and_is_idempotent(synthetic_env):
    ag = DataIntelligenceAgent()
    r1 = ag.run({}, force=True)
    assert r1.status == "SUCCESS", r1.error
    assert r1.output["duplicate_records"] == 1
    assert r1.output["invalid_records"] >= 4
    assert repo.count_rows(LoanMaster) == 6
    assert repo.count_rows(RepaymentHistory) == 10
    n_dq = repo.count_rows(DataQualityLog)
    assert n_dq == r1.output["exceptions_total"]
    r2 = ag.run({})
    assert r2.status == "SUCCESS" and r2.output["reused"] is True
    assert repo.count_rows(DataQualityLog) == n_dq          # no duplicate exception rows
    logs = repo.list_agent_logs(5, agent_name="data_intelligence")
    assert len(logs) == 2 and all(l["status"] == "SUCCESS" for l in logs)


def test_data_agent_fails_cleanly_on_missing_file(synthetic_env):
    synthetic_env.source_files()["repayment"].unlink()
    r = DataIntelligenceAgent().run({}, force=True)
    assert r.status == "FAILED" and "repayment" in r.error
    assert repo.list_agent_logs(1)[0]["status"] == "FAILED"


def test_all_15_tables_created(synthetic_env):
    from sqlalchemy import inspect

    from backend.database import get_engine
    names = set(inspect(get_engine()).get_table_names())
    assert len(ALL_TABLES) == 15 and set(ALL_TABLES) <= names


# ------------------------------------------------------------ real data
def test_real_data_pipeline(real_env):
    r = DataIntelligenceAgent().run({}, force=True)
    assert r.status == "SUCCESS", r.error
    o = r.output
    assert (o["application_records"], o["disbursement_records"], o["repayment_records"]) == (5000, 5000, 62851)
    assert o["matched_accounts"] == 5000
    assert o["keys_normalised"] == 37
    assert o["observation_cutoff"] == "2026-07-28"
    lm = repo.read_table("loan_master")
    src = pd.read_csv(real_env.source_files()["disbursement"], dtype=str, encoding="utf-8-sig")
    assert abs(lm.loan_amount.sum() - pd.to_numeric(src.loan_amount.str.strip()).sum()) < 1
