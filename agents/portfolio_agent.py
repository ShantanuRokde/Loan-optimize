"""Agent 2 – Portfolio Monitoring Agent.

Computes portfolio, delinquency and business KPIs from the processed loan
master and repayment history. Read-only with respect to the database; the
full result is written to data/reports/portfolio_summary.json.
"""
from __future__ import annotations

import pandas as pd

from agents.base import AgentResult, BaseAgent, save_report
from agents.data_agent import current_data_hash, get_processed_datasets
from analytics.delinquency import bucket_distribution, delinquency_kpis, dpd_distribution, monthly_trends, roll_rates
from analytics.portfolio_metrics import (
    all_breakdowns,
    disbursement_trend,
    distributions,
    geography_consistency,
    portfolio_kpis,
    quarterly_disbursement,
)
from backend.config import get_config
from utils.helpers import to_jsonable, utcnow

REPORT_NAME = "portfolio_summary"


def build_portfolio_payload(lm: pd.DataFrame, rh: pd.DataFrame, npa_dpd: int, overdue_dpd: int) -> dict:
    trends = monthly_trends(rh, npa_dpd)
    return {
        "kpis": portfolio_kpis(lm, npa_dpd, overdue_dpd),
        "delinquency": delinquency_kpis(lm, npa_dpd),
        "dpd_buckets": to_jsonable(bucket_distribution(lm)),
        "dpd_row_distribution": to_jsonable(dpd_distribution(rh)),
        "monthly_trends": to_jsonable(trends),
        "disbursement_trend": to_jsonable(disbursement_trend(lm)),
        "quarterly_disbursement": to_jsonable(quarterly_disbursement(lm)),
        "breakdowns": all_breakdowns(lm, npa_dpd, overdue_dpd),
        "distributions": distributions(lm),
        "roll_rates": {k: to_jsonable(v) for k, v in roll_rates(rh).to_dict(orient="index").items()},
        "geography": geography_consistency(lm),
    }


class PortfolioMonitoringAgent(BaseAgent):
    name = "portfolio_monitoring"
    title = "Portfolio Monitoring Agent"
    description = "Portfolio, delinquency, NPA and business-segment KPIs with monthly trends."
    writes_state = False
    retry_safe = True

    def execute(self, ctx: dict, **_) -> AgentResult:
        cfg = get_config()
        self.current_operation = "Loading processed datasets"
        ads = get_processed_datasets(cfg)
        self.current_operation = "Computing KPIs, delinquency buckets, trends and breakdowns"
        payload = build_portfolio_payload(ads.loan_master, ads.repayment_history,
                                          cfg.delinquency.npa_dpd_threshold, cfg.delinquency.overdue_dpd_threshold)
        payload.update(generated_at=utcnow(), run_id=ctx["run_id"], data_hash=ctx.get("data_hash") or current_data_hash(cfg),
                       observation_cutoff=None if ads.observation_cutoff is None else str(ads.observation_cutoff.date()))
        path = save_report(REPORT_NAME, payload)
        findings = self._findings(payload)
        ctx["portfolio"] = payload
        k, d = payload["kpis"], payload["delinquency"]
        summary = {
            "total_loans": k["total_loans"], "total_disbursed": k["total_disbursed"],
            "total_outstanding": k["total_outstanding"], "default_rate": k["default_rate"],
            "npa_accounts": k["npa_accounts"], "npa_rate_count": k["npa_rate_count"],
            "npa_rate_balance": k["npa_rate_balance"], "dpd_30plus_accounts": d["dpd_30plus_accounts"],
            "dpd_60plus_accounts": d["dpd_60plus_accounts"], "dpd_90plus_accounts": d["dpd_90plus_accounts"],
            "report_file": path,
        }
        return AgentResult(status="SUCCESS", output=summary, findings=findings)

    @staticmethod
    def _findings(p: dict) -> list[str]:
        k, d = p["kpis"], p["delinquency"]
        f = [
            f"{k['total_loans']:,} loans, ₹{k['total_disbursed'] / 1e7:,.2f} Cr disbursed, "
            f"₹{k['total_outstanding'] / 1e7:,.2f} Cr outstanding.",
            f"Current 30+/60+/90+ DPD accounts: {d['dpd_30plus_accounts']} / {d['dpd_60plus_accounts']} / "
            f"{d['dpd_90plus_accounts']}.",
            f"NPA proxy ({k['npa_dpd_threshold']}+ DPD): {k['npa_accounts']} accounts = "
            f"{(k['npa_rate_count'] or 0):.2%} by count, {(k['npa_rate_balance'] or 0):.2%} by outstanding.",
            f"Ever-default rate ({k['npa_dpd_threshold']}+ DPD at any point): {(k['default_rate'] or 0):.2%} "
            f"({k['defaulted_accounts']} accounts).",
        ]
        for dim in ("loan_type", "region", "customer_segment", "occupation", "income_band"):
            rows = p["breakdowns"][dim]
            if rows:
                worst = max(rows, key=lambda r: r["default_rate"] or 0)
                best = min(rows, key=lambda r: r["default_rate"] or 0)
                f.append(f"By {dim.replace('_', ' ')}: highest default rate {worst['group']} "
                         f"({worst['default_rate']:.2%}), lowest {best['group']} ({best['default_rate']:.2%}).")
        tr = p["monthly_trends"]
        if len(tr) >= 7:
            last, prev = tr[-1], tr[-7]
            f.append(f"30+ DPD rate moved from {prev['dpd30_rate']:.2%} ({prev['month']}) to "
                     f"{last['dpd30_rate']:.2%} ({last['month']}).")
        geo = p["geography"]
        if any(v > 1 for v in geo["regions_per_state"].values()):
            f.append("Data observation: states map to multiple regions and branch names do not match their state – "
                     "geography fields are reported as recorded in the source.")
        return f
