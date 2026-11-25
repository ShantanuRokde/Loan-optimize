"""Repository: the only module that writes to the database.

Keeping writes here makes the audit trail and immutability rules enforceable
in one place.
"""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
from sqlalchemy import delete, func, select

from backend.database import get_engine, session_scope
from database.models import (
    AgentExecutionLog,
    AuditLog,
    DataQualityLog,
    LoanMaster,
    RepaymentHistory,
    SourceDataset,
)
from utils.helpers import to_jsonable, utcnow


# ------------------------------------------------------------------ audit
def write_audit(event_type: str, actor: str, entity_type: str | None = None, entity_id: str | None = None,
                details: dict | None = None, run_id: str | None = None) -> None:
    with session_scope() as s:
        s.add(AuditLog(event_type=event_type, actor=actor, entity_type=entity_type, entity_id=entity_id,
                       details_json=to_jsonable(details or {}), run_id=run_id))


def list_audit(limit: int = 500, event_type: str | None = None, entity_id: str | None = None) -> list[dict]:
    with session_scope() as s:
        q = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
        if event_type:
            q = q.where(AuditLog.event_type == event_type)
        if entity_id:
            q = q.where(AuditLog.entity_id == entity_id)
        return [_row(r) for r in s.scalars(q)]


# ----------------------------------------------------------- source files
def record_source_datasets(run_id: str, entries: Iterable[dict]) -> None:
    with session_scope() as s:
        for e in entries:
            s.add(SourceDataset(run_id=run_id, **e))


def latest_source_datasets() -> list[dict]:
    with session_scope() as s:
        last = s.scalar(select(SourceDataset.run_id).order_by(SourceDataset.id.desc()).limit(1))
        if not last:
            return []
        return [_row(r) for r in s.scalars(select(SourceDataset).where(SourceDataset.run_id == last))]


# -------------------------------------------------------------- DQ logs
def replace_dq_logs(run_id: str, exceptions: pd.DataFrame) -> int:
    """Store the exception table for this run (previous runs are retained)."""
    if exceptions.empty:
        return 0
    df = exceptions.copy()
    df["run_id"] = run_id
    df["created_at"] = utcnow()
    cols = [c.name for c in DataQualityLog.__table__.columns if c.name != "id"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.astype(object).where(pd.notna(df), None)
    df.to_sql(DataQualityLog.__tablename__, get_engine(), if_exists="append", index=False, chunksize=2000)
    return len(df)


def get_dq_logs(run_id: str | None = None, severity: str | None = None, dataset: str | None = None,
                check_name: str | None = None, limit: int = 5000) -> pd.DataFrame:
    with session_scope() as s:
        if run_id is None:
            run_id = s.scalar(select(DataQualityLog.run_id).order_by(DataQualityLog.id.desc()).limit(1))
        if run_id is None:
            return pd.DataFrame()
        q = select(DataQualityLog).where(DataQualityLog.run_id == run_id)
        if severity:
            q = q.where(DataQualityLog.severity == severity)
        if dataset:
            q = q.where(DataQualityLog.dataset == dataset)
        if check_name:
            q = q.where(DataQualityLog.check_name == check_name)
        rows = [_row(r) for r in s.scalars(q.order_by(DataQualityLog.id).limit(limit))]
    return pd.DataFrame(rows)


# ------------------------------------------------- analytical tables
def _prep_for_sql(df: pd.DataFrame, model) -> pd.DataFrame:
    cols = [c.name for c in model.__table__.columns]
    out = df[[c for c in cols if c in df.columns]].copy()
    for c in model.__table__.columns:
        if c.name in out.columns and c.type.python_type.__name__ == "date":
            out[c.name] = pd.to_datetime(out[c.name], errors="coerce").dt.date
    return out.astype(object).where(pd.notna(out), None)


def replace_loan_master(df: pd.DataFrame) -> int:
    engine = get_engine()
    with session_scope() as s:
        s.execute(delete(LoanMaster))
    _prep_for_sql(df, LoanMaster).to_sql(LoanMaster.__tablename__, engine, if_exists="append", index=False, chunksize=1000)
    return len(df)


def replace_repayment_history(df: pd.DataFrame) -> int:
    engine = get_engine()
    with session_scope() as s:
        s.execute(delete(RepaymentHistory))
    out = _prep_for_sql(df.drop(columns=["id"], errors="ignore"), RepaymentHistory)
    out = out.drop(columns=["id"], errors="ignore")
    out.to_sql(RepaymentHistory.__tablename__, engine, if_exists="append", index=False, chunksize=5000)
    return len(df)


def count_rows(model) -> int:
    with session_scope() as s:
        return int(s.scalar(select(func.count()).select_from(model)) or 0)


def read_table(name: str) -> pd.DataFrame:
    return pd.read_sql_table(name, get_engine())


# ------------------------------------------------------------ agent logs
def start_agent_log(run_id: str, agent_name: str, operation: str, input_json: dict | None, attempt: int = 1) -> int:
    with session_scope() as s:
        row = AgentExecutionLog(run_id=run_id, agent_name=agent_name, operation=operation, status="RUNNING",
                                attempt=attempt, input_json=to_jsonable(input_json or {}))
        s.add(row)
        s.flush()
        return row.id


def finish_agent_log(log_id: int, status: str, output_json: Any = None, findings: list | None = None,
                     error: str | None = None) -> None:
    with session_scope() as s:
        row = s.get(AgentExecutionLog, log_id)
        if row is None:
            return
        row.status = status
        row.output_json = to_jsonable(output_json)
        row.findings_json = to_jsonable(findings or [])
        row.error = error
        row.finished_at = utcnow()
        row.duration_ms = int((row.finished_at - row.started_at).total_seconds() * 1000)


def list_agent_logs(limit: int = 200, agent_name: str | None = None, run_id: str | None = None) -> list[dict]:
    with session_scope() as s:
        q = select(AgentExecutionLog).order_by(AgentExecutionLog.id.desc()).limit(limit)
        if agent_name:
            q = q.where(AgentExecutionLog.agent_name == agent_name)
        if run_id:
            q = q.where(AgentExecutionLog.run_id == run_id)
        return [_row(r) for r in s.scalars(q)]


# ---------------------------------------------------------------- utils
def _row(obj) -> dict:
    return to_jsonable({c.name: getattr(obj, c.name) for c in obj.__table__.columns})
