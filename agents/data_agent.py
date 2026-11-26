"""Agent 1 – Data Intelligence Agent.

Loads the three CSVs, validates them, builds the unified analytical datasets
and publishes a data-quality report. Idempotent: when the source-file hashes
match the last successful build, the persisted outputs are reused.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from agents.base import AgentError, AgentResult, BaseAgent
from analytics.data_loader import combined_hash, load_sources
from analytics.data_preprocessing import build_analytical_datasets, load_processed, save_processed
from analytics.data_validation import validate_datasets
from backend.config import AppConfig, get_config
from database import repository as repo
from database.models import LoanMaster
from utils.constants import (
    DATASET_APPLICATION as APP,
    DATASET_DISBURSEMENT as DISB,
    DATASET_REPAYMENT as REP,
    EXPECTED_SCHEMAS,
    SEV_ERROR,
    SEV_INFO,
    SEV_WARNING,
)
from utils.helpers import to_jsonable, utcnow

BUILD_META = "build_meta.json"


class DataIntelligenceAgent(BaseAgent):
    name = "data_intelligence"
    title = "Data Intelligence Agent"
    description = "Loads, validates and unifies the application, disbursement and repayment datasets."
    writes_state = True
    retry_safe = True  # rebuild is deterministic and replaces its own outputs

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__()
        self._config = config

    @property
    def config(self) -> AppConfig:
        return self._config or get_config()

    # ------------------------------------------------------------------
    def execute(self, ctx: dict, force: bool = False, **_) -> AgentResult:
        cfg = self.config
        t0 = time.perf_counter()
        steps: list[dict] = []
        run_id = ctx["run_id"]

        self.step("Loading source CSV files (read-only)", steps, t0)
        files = cfg.source_files()
        loaded = load_sources(files)
        failed = [d for d in loaded.values() if not d.ok]
        data_hash = combined_hash(loaded) if not failed else None
        repo.record_source_datasets(run_id, [d.meta() for d in loaded.values()])

        meta_path = cfg.processed_dir / BUILD_META
        prev = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if (not force and data_hash and prev and prev.get("data_hash") == data_hash
                and prev.get("status") != "FAIL" and load_processed(cfg.processed_dir) is not None
                and repo.count_rows(LoanMaster) > 0):
            self.step("Source data unchanged since last build – reusing processed datasets", steps, t0)
            out = dict(prev["output"])
            out.update(reused=True, run_id=run_id, processing_log=steps, previous_build_run_id=prev.get("run_id"))
            ctx.update(data_hash=data_hash, observation_cutoff=out.get("observation_cutoff"))
            return AgentResult(status="SUCCESS", output=out,
                               findings=["Source files unchanged (same SHA-256); previous validated build reused."] + prev.get("findings", []))

        self.step("Validating schema, types, ranges, duplicates and referential integrity", steps, t0)
        vr = validate_datasets(loaded)
        exc = vr.exceptions
        n_dq = repo.replace_dq_logs(run_id, exc)
        self.step(f"Recorded {n_dq} data-quality exceptions", steps, t0)

        if vr.status == "FAIL":
            report = self._report(run_id, data_hash, loaded, vr, None, steps)
            self._write_report(report, exc, run_id)
            raise AgentError("Validation failed: " + "; ".join(vr.fatal_errors))

        self.step("Joining application ⟷ disbursement (1:1) and aggregating repayments per loan", steps, t0)
        ads = build_analytical_datasets(vr, npa_dpd=cfg.delinquency.npa_dpd_threshold)
        if abs(ads.stats["loan_amount_sum_master"] - ads.stats["loan_amount_sum_source"]) > 0.5:
            raise AgentError("Reconciliation failed: loan_amount total in loan_master differs from source")
        ads.loan_master["build_run_id"] = run_id

        self.step("Persisting processed datasets (parquet/csv + SQLite)", steps, t0)
        paths = save_processed(ads, cfg.processed_dir)
        repo.replace_loan_master(ads.loan_master)
        repo.replace_repayment_history(ads.repayment_history)

        report = self._report(run_id, data_hash, loaded, vr, ads, steps)
        report_paths = self._write_report(report, exc, run_id)
        findings = self._findings(report)
        self.step("Data intelligence complete", steps, t0)

        output = {**report["kpis"], "status": vr.status, "data_hash": data_hash,
                  "observation_cutoff": ads.stats["observation_cutoff"], "processed_files": paths,
                  "report_files": report_paths, "processing_log": steps, "reused": False, "run_id": run_id}
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(to_jsonable({"data_hash": data_hash, "run_id": run_id, "status": vr.status,
                                                     "built_at": utcnow(), "output": output, "findings": findings}), indent=2))
        repo.write_audit("DATA_BUILD", ctx.get("actor", "system"), "dataset", data_hash[:16],
                         {"status": vr.status, "loans": ads.stats["loans_total"], "exceptions": len(exc)}, run_id)
        ctx.update(data_hash=data_hash, observation_cutoff=ads.stats["observation_cutoff"])
        return AgentResult(status="SUCCESS", output=output, findings=findings)

    # ------------------------------------------------------------------
    @staticmethod
    def _report(run_id, data_hash, loaded, vr, ads, steps) -> dict:
        exc = vr.exceptions

        def cnt(check=None, sev=None, ds=None):
            m = pd.Series(True, index=exc.index)
            if check is not None:
                m &= exc.check_name.isin([check] if isinstance(check, str) else check)
            if sev is not None:
                m &= exc.severity == sev
            if ds is not None:
                m &= exc.dataset == ds
            return int(m.sum())

        missing_total = sum(sum(v.get("missing_values", {}).values()) for v in vr.schema_report.values())
        stats = ads.stats if ads else {}
        kpis = {
            "application_records": len(loaded[APP].frame) if loaded[APP].ok else 0,
            "disbursement_records": len(loaded[DISB].frame) if loaded[DISB].ok else 0,
            "repayment_records": len(loaded[REP].frame) if loaded[REP].ok else 0,
            "matched_accounts": stats.get("loans_matched", 0),
            "unmatched_applications": cnt("UNMATCHED_APPLICATION"),
            "unmatched_disbursements": cnt("UNMATCHED_DISBURSEMENT"),
            "orphan_repayment_rows": cnt("ORPHAN_REPAYMENT"),
            "loans_without_repayment": cnt("NO_REPAYMENT_HISTORY"),
            "duplicate_records": cnt(["DUPLICATE_ROW", "DUPLICATE_KEY"]),
            "invalid_records": cnt(sev=SEV_ERROR),
            "invalid_dpd_records": cnt("INVALID_DPD"),
            "missing_values": missing_total,
            "keys_normalised": cnt("KEY_NORMALISED"),
            "warnings": cnt(sev=SEV_WARNING),
            "info": cnt(sev=SEV_INFO),
            "exceptions_total": int(len(exc)),
            "analytic_eligible_loans": stats.get("loans_analytic_eligible", 0),
            "excluded_loans": stats.get("loans_excluded", 0),
            "validation_status": vr.status,
        }
        return {
            "run_id": run_id, "generated_at": utcnow(), "data_hash": data_hash, "validation": vr.summary(),
            "kpis": kpis, "dataset_stats": stats,
            "files": {n: d.meta() for n, d in loaded.items()},
            "expected_schemas": EXPECTED_SCHEMAS, "processing_log": steps,
        }

    def _write_report(self, report: dict, exc: pd.DataFrame, run_id: str) -> dict:
        d = self.config.reports_dir
        d.mkdir(parents=True, exist_ok=True)
        jp, cp = d / "data_quality_report.json", d / "data_quality_exceptions.csv"
        jp.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
        exc.to_csv(cp, index=False)
        return {"report": str(jp), "exceptions": str(cp)}

    @staticmethod
    def _findings(report: dict) -> list[str]:
        k, s = report["kpis"], report["dataset_stats"]
        f = [
            f"Loaded {k['application_records']:,} application, {k['disbursement_records']:,} disbursement and "
            f"{k['repayment_records']:,} repayment records.",
            f"{k['matched_accounts']:,} accounts matched across application and disbursement; "
            f"{k['analytic_eligible_loans']:,} loans eligible for analytics, {k['excluded_loans']:,} excluded (flagged, not deleted).",
            f"Repayment observation cutoff derived from data: {s.get('observation_cutoff')} "
            f"(disbursals {s.get('disbursal_min')} → {s.get('disbursal_max')}).",
        ]
        if k["keys_normalised"]:
            f.append(f"{k['keys_normalised']} account keys contained embedded BOM/quote characters (file-concatenation "
                     "artefact) and were normalised; raw values preserved in the exception table.")
        if k["duplicate_records"]:
            f.append(f"{k['duplicate_records']} duplicate records flagged.")
        if k["invalid_records"]:
            f.append(f"{k['invalid_records']} records failed validation (ERROR) and are excluded from analytics.")
        unmatched = k["unmatched_applications"] + k["unmatched_disbursements"] + k["orphan_repayment_rows"]
        f.append(f"Referential integrity: {unmatched} unmatched records after key normalisation.")
        f.append(f"Validation status: {k['validation_status']}.")
        return f


def current_data_hash(config: AppConfig | None = None) -> str | None:
    """Hash of the source data behind the current processed build."""
    cfg = config or get_config()
    p = cfg.processed_dir / BUILD_META
    return json.loads(p.read_text()).get("data_hash") if p.exists() else None


def get_processed_datasets(config: AppConfig | None = None):
    """Accessor used by downstream agents. Raises if Agent 1 has not run."""
    cfg = config or get_config()
    ads = load_processed(cfg.processed_dir)
    if ads is None:
        raise AgentError("Processed datasets not found – run the Data Intelligence Agent first.")
    return ads
