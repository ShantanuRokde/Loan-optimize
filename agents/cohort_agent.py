"""Agent 3 – Cohort Analysis Agent.

Calculates all nine DPD cohorts by disbursement quarter with strict maturity,
control limits and breach detection, ranks cohorts by risk *and* data
sufficiency, and selects the weakest eligible cohort (or validates a manual
selection). Results are persisted to cohort_results (idempotent per data hash)
and data/reports/cohort_analysis.json.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import delete

from agents.base import AgentError, AgentResult, BaseAgent, save_report
from agents.data_agent import current_data_hash, get_processed_datasets
from analytics.cohort_calculation import (
    cohort_definition,
    compute_all_cohorts,
    heatmap_matrix,
    parse_cohort,
)
from analytics.statistical_analysis import rank_cohorts, summarise_cohort
from backend.config import get_config
from backend.database import session_scope
from database.models import CohortResult
from utils.constants import COHORTS
from utils.helpers import to_jsonable, utcnow

REPORT_NAME = "cohort_analysis"


def run_cohort_analysis(lm: pd.DataFrame, rh: pd.DataFrame, cutoff, min_mature_quarters: int,
                        min_defaults: int, selected: str | None = None, sd_level: int = 2) -> dict:
    """Pure computation (no I/O) – used by the agent and by tests."""
    tables, frames = compute_all_cohorts(lm, rh, cutoff)
    summaries = [summarise_cohort(c, tables[c], min_mature_quarters, min_defaults) for c in COHORTS]
    ranked = rank_cohorts(summaries)
    by_code = {s["cohort"]: s for s in ranked}

    if selected:
        selected = selected.strip().upper()
        parse_cohort(selected)
        if selected not in by_code:
            raise AgentError(f"Unknown cohort {selected}")
        mode = "manual"
        sel = by_code[selected]
        warn = None if sel["eligible_for_selection"] else (
            f"Manually selected cohort {selected} is not statistically eligible: " + "; ".join(sel["ineligibility_reasons"]))
    else:
        eligible = [s for s in ranked if s["eligible_for_selection"]]
        if not eligible:
            raise AgentError("No cohort has sufficient mature quarters / defaults for selection "
                             f"(min {min_mature_quarters} mature quarters, {min_defaults} defaults).")
        sel, mode, warn = eligible[0], "auto", None

    sd_key = "breach_2sd" if sd_level == 2 else "breach_1sd"
    quarter_tables = {c: s["_table"] for c, s in by_code.items()}
    sel_t = quarter_tables[sel["cohort"]]
    exposure = [{"cohort": c, "quarter": r["quarter"], "is_mature": r["is_mature"],
                 "total_disbursed": r["total_disbursed"], "defaulted_amount": r["defaulted_amount"],
                 "defaulted_outstanding_at_breach": r["defaulted_outstanding_at_breach"],
                 "exposure_rate": r["exposure_rate"]}
                for c, t in quarter_tables.items() for r in t.to_dict(orient="records")]

    return {
        "definitions": [cohort_definition(c) for c in COHORTS],
        "selection": {
            "cohort": sel["cohort"], "mode": mode, "warning": warn,
            "definition": cohort_definition(sel["cohort"])["definition"],
            "mature_quarters": sel["mature_quarters"],
            "weakest_quarter": sel["weakest_quarter"], "weakest_quarter_rate": sel["weakest_quarter_rate"],
            "pooled_default_rate": sel["pooled_default_rate"], "wilson_lower": sel["wilson_lower"],
            "breaches": sel_t.loc[sel_t[sd_key], "quarter"].tolist(), "sd_level": sd_level,
            "ranking_method": "Wilson 95% lower bound of pooled default rate over mature quarters "
                              f"(cohorts need >= {min_mature_quarters} mature quarters and >= {min_defaults} defaults)",
        },
        "ranking": [{k: v for k, v in s.items() if k not in ("_table",)} for s in ranked],
        "quarter_tables": {c: to_jsonable(t) for c, t in quarter_tables.items()},
        "heatmap": heatmap_matrix(tables),
        "exposure": to_jsonable(exposure),
        "_frames": frames,
        "_tables": quarter_tables,
    }


class CohortAnalysisAgent(BaseAgent):
    name = "cohort_analysis"
    title = "Cohort Analysis Agent"
    description = "Nine DPD cohorts by disbursal quarter, maturity, control limits and weakest-cohort selection."
    writes_state = True
    retry_safe = True   # replaces its own rows for the same data hash

    def execute(self, ctx: dict, cohort: str | None = None, sd_level: int | None = None, **_) -> AgentResult:
        cfg = get_config()
        cohort = cohort or ctx.get("requested_cohort") or cfg.cohort.selected_cohort
        sd_level = sd_level or cfg.cohort.control_limit_sd
        self.current_operation = "Loading processed datasets"
        ads = get_processed_datasets(cfg)
        self.current_operation = "Calculating nine cohorts with maturity logic"
        res = run_cohort_analysis(ads.loan_master, ads.repayment_history, ads.observation_cutoff,
                                  cfg.cohort.min_mature_quarters, cfg.cohort.min_cohort_defaults, cohort, sd_level)
        data_hash = ctx.get("data_hash") or current_data_hash(cfg)
        cutoff = None if ads.observation_cutoff is None else ads.observation_cutoff.date()

        self.current_operation = "Persisting cohort results"
        with session_scope() as s:
            if data_hash:
                s.execute(delete(CohortResult).where(CohortResult.data_hash == data_hash))
            for code, t in res["_tables"].items():
                for r in t.to_dict(orient="records"):
                    s.add(CohortResult(
                        run_id=ctx["run_id"], data_hash=data_hash, cohort=code, quarter=r["quarter"],
                        total_loans=r["total_loans"], eligible_loans=r["eligible_loans"], defaults=r["defaults"],
                        default_rate=None if pd.isna(r["default_rate"]) else float(r["default_rate"]),
                        total_disbursed=r["total_disbursed"], defaulted_amount=r["defaulted_amount"],
                        avg_max_dpd=None if pd.isna(r["avg_max_dpd"]) else float(r["avg_max_dpd"]),
                        is_mature=bool(r["is_mature"]), maturity_pct=float(r["maturity_pct"]),
                        observation_window_months=int(r["observation_window_months"]), observation_end=cutoff,
                        breach_1sd=bool(r["breach_1sd"]), breach_2sd=bool(r["breach_2sd"])))

        payload = {k: v for k, v in res.items() if not k.startswith("_")}
        payload.update(generated_at=utcnow(), run_id=ctx["run_id"], data_hash=data_hash,
                       observation_cutoff=None if cutoff is None else cutoff.isoformat(),
                       thresholds={"min_mature_quarters": cfg.cohort.min_mature_quarters,
                                   "min_cohort_defaults": cfg.cohort.min_cohort_defaults})
        path = save_report(REPORT_NAME, payload)

        sel = res["selection"]
        ctx.update(selected_cohort=sel["cohort"], cohort_mature_quarters=sel["mature_quarters"],
                   cohort_selection=sel, cohort_frames=res["_frames"])
        return AgentResult(status="SUCCESS", output={
            "selected_cohort": sel["cohort"], "selection_mode": sel["mode"],
            "mature_quarters": sel["mature_quarters"], "weakest_quarter": sel["weakest_quarter"],
            "weakest_quarter_rate": sel["weakest_quarter_rate"], "pooled_default_rate": sel["pooled_default_rate"],
            "breaches": sel["breaches"], "eligible_cohorts": [r["cohort"] for r in res["ranking"] if r["rank"]],
            "report_file": path,
        }, findings=self._findings(res))

    @staticmethod
    def _findings(res: dict) -> list[str]:
        sel = res["selection"]
        f = []
        if sel["warning"]:
            f.append("WARNING: " + sel["warning"])
        f.append(f"Selected cohort {sel['cohort']} ({sel['definition']}) – {sel['mode']} selection.")
        if sel["pooled_default_rate"] is not None:
            f.append(f"Pooled default rate over {len(sel['mature_quarters'])} mature quarters "
                     f"({', '.join(sel['mature_quarters'])}): {sel['pooled_default_rate']:.2%} "
                     f"(Wilson 95% lower bound {sel['wilson_lower']:.2%}).")
        if sel["weakest_quarter"]:
            f.append(f"Weakest mature quarter: {sel['weakest_quarter']} at {sel['weakest_quarter_rate']:.2%}.")
        f.append(f"{sel['sd_level']}-SD control-limit breaches: {', '.join(sel['breaches']) or 'none'}.")
        for r in res["ranking"]:
            if not r["eligible_for_selection"]:
                f.append(f"{r['cohort']} not ranked: {'; '.join(r['ineligibility_reasons'])}.")
        top = [r for r in res["ranking"] if r["rank"]][:3]
        f.append("Ranking (top 3): " + ", ".join(
            f"#{r['rank']} {r['cohort']} {r['pooled_default_rate']:.2%}" for r in top) + ".")
        return f
