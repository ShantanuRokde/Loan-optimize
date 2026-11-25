"""SQLAlchemy ORM schema — the 15 persistent tables of CreditRisk AI.

Analytical tables (loan_master, repayment_history) are rebuilt by the Data
Intelligence Agent from the read-only raw CSVs. Governance tables (policy_*,
audit_logs, agent_execution_logs, refresh_runs, model_registry) are
append-mostly and persist across restarts.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from utils.helpers import utcnow


class Base(DeclarativeBase):
    pass


# 1 ----------------------------------------------------------------------
class SourceDataset(Base):
    __tablename__ = "source_datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    dataset_name: Mapped[str] = mapped_column(String(32))
    file_path: Mapped[str] = mapped_column(Text)
    file_hash: Mapped[str] = mapped_column(String(64))
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    row_count: Mapped[int] = mapped_column(Integer)
    column_count: Mapped[int] = mapped_column(Integer)
    columns_json: Mapped[list | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16))  # LOADED | FAILED
    message: Mapped[str | None] = mapped_column(Text)
    loaded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# 2 ----------------------------------------------------------------------
class DataQualityLog(Base):
    __tablename__ = "data_quality_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    dataset: Mapped[str] = mapped_column(String(32), index=True)
    check_name: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(8), index=True)
    row_number: Mapped[int | None] = mapped_column(Integer)  # 1-based data row in source file
    loan_account_no: Mapped[str | None] = mapped_column(String(32), index=True)
    column_name: Mapped[str | None] = mapped_column(String(64))
    raw_value: Mapped[str | None] = mapped_column(Text)
    message: Mapped[str] = mapped_column(Text)
    action_taken: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# 3 ----------------------------------------------------------------------
class LoanMaster(Base):
    __tablename__ = "loan_master"
    loan_account_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    raw_loan_account_no: Mapped[str | None] = mapped_column(String(64))
    match_status: Mapped[str] = mapped_column(String(24))
    # application
    application_date: Mapped[date | None] = mapped_column(Date)
    customer_segment: Mapped[str | None] = mapped_column(String(32))
    occupation: Mapped[str | None] = mapped_column(String(32))
    income_band: Mapped[str | None] = mapped_column(String(16))
    income_band_ordinal: Mapped[int | None] = mapped_column(Integer)
    income_ge_10l: Mapped[bool | None] = mapped_column(Boolean)
    credit_score: Mapped[float | None] = mapped_column(Float)
    enquiries_6m: Mapped[float | None] = mapped_column(Float)
    existing_loans: Mapped[float | None] = mapped_column(Float)
    age: Mapped[float | None] = mapped_column(Float)
    region: Mapped[str | None] = mapped_column(String(16))
    state: Mapped[str | None] = mapped_column(String(32))
    branch: Mapped[str | None] = mapped_column(String(32))
    loan_type: Mapped[str | None] = mapped_column(String(32))
    product_code: Mapped[str | None] = mapped_column(String(16))
    requested_amount: Mapped[float | None] = mapped_column(Float)
    # disbursement
    disbursal_date: Mapped[date | None] = mapped_column(Date)
    disbursal_quarter: Mapped[str | None] = mapped_column(String(8), index=True)
    disbursal_month: Mapped[str | None] = mapped_column(String(7))
    sanction_amount: Mapped[float | None] = mapped_column(Float)
    loan_amount: Mapped[float | None] = mapped_column(Float)
    emi_amount: Mapped[float | None] = mapped_column(Float)
    loan_tenure_months: Mapped[float | None] = mapped_column(Float)
    interest_rate: Mapped[float | None] = mapped_column(Float)
    # repayment aggregates
    n_installments: Mapped[int | None] = mapped_column(Integer)
    n_paid: Mapped[int | None] = mapped_column(Integer)
    n_missed: Mapped[int | None] = mapped_column(Integer)
    first_due_date: Mapped[date | None] = mapped_column(Date)
    last_due_date: Mapped[date | None] = mapped_column(Date)
    max_mob_observed: Mapped[int | None] = mapped_column(Integer)
    months_to_cutoff: Mapped[int | None] = mapped_column(Integer)
    max_dpd: Mapped[float | None] = mapped_column(Float)
    current_dpd: Mapped[float | None] = mapped_column(Float)
    current_dpd_bucket: Mapped[str | None] = mapped_column(String(12))
    current_outstanding: Mapped[float | None] = mapped_column(Float)
    total_emi_due: Mapped[float | None] = mapped_column(Float)
    total_emi_paid: Mapped[float | None] = mapped_column(Float)
    first_mob_30dpd: Mapped[int | None] = mapped_column(Integer)
    first_mob_60dpd: Mapped[int | None] = mapped_column(Integer)
    first_mob_90dpd: Mapped[int | None] = mapped_column(Integer)
    ever_30dpd: Mapped[bool | None] = mapped_column(Boolean)
    ever_60dpd: Mapped[bool | None] = mapped_column(Boolean)
    ever_90dpd: Mapped[bool | None] = mapped_column(Boolean)
    is_npa: Mapped[bool | None] = mapped_column(Boolean)
    has_repayment_gaps: Mapped[bool | None] = mapped_column(Boolean)
    # data-quality flags
    dq_error_count: Mapped[int] = mapped_column(Integer, default=0)
    dq_warning_count: Mapped[int] = mapped_column(Integer, default=0)
    dq_issues: Mapped[str | None] = mapped_column(Text)
    is_analytic_eligible: Mapped[bool] = mapped_column(Boolean, default=True)
    build_run_id: Mapped[str | None] = mapped_column(String(64))


# 4 ----------------------------------------------------------------------
class RepaymentHistory(Base):
    __tablename__ = "repayment_history"
    id: Mapped[int] = mapped_column(primary_key=True)
    loan_account_no: Mapped[str] = mapped_column(String(32), index=True)
    source_row_number: Mapped[int] = mapped_column(Integer)
    due_month: Mapped[date | None] = mapped_column(Date)
    due_period: Mapped[str | None] = mapped_column(String(7), index=True)
    mob: Mapped[int | None] = mapped_column(Integer)
    emi_amount: Mapped[float | None] = mapped_column(Float)
    paid_flag: Mapped[str | None] = mapped_column(String(2))
    days_past_due: Mapped[float | None] = mapped_column(Float)
    dpd_bucket: Mapped[str | None] = mapped_column(String(12))
    outstanding_balance: Mapped[float | None] = mapped_column(Float)
    disbursal_quarter: Mapped[str | None] = mapped_column(String(8))
    has_loan_record: Mapped[bool] = mapped_column(Boolean, default=True)
    dq_flag: Mapped[str | None] = mapped_column(String(16))
    is_analytic_eligible: Mapped[bool] = mapped_column(Boolean, default=True)


# 5 ----------------------------------------------------------------------
class CohortResult(Base):
    __tablename__ = "cohort_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    data_hash: Mapped[str | None] = mapped_column(String(64))
    cohort: Mapped[str] = mapped_column(String(12), index=True)
    quarter: Mapped[str] = mapped_column(String(8))
    total_loans: Mapped[int] = mapped_column(Integer)
    eligible_loans: Mapped[int] = mapped_column(Integer)
    defaults: Mapped[int] = mapped_column(Integer)
    default_rate: Mapped[float | None] = mapped_column(Float)
    total_disbursed: Mapped[float | None] = mapped_column(Float)
    defaulted_amount: Mapped[float | None] = mapped_column(Float)
    avg_max_dpd: Mapped[float | None] = mapped_column(Float)
    is_mature: Mapped[bool] = mapped_column(Boolean)
    maturity_pct: Mapped[float | None] = mapped_column(Float)
    observation_window_months: Mapped[int] = mapped_column(Integer)
    observation_end: Mapped[date | None] = mapped_column(Date)
    breach_1sd: Mapped[bool | None] = mapped_column(Boolean)
    breach_2sd: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# 6 ----------------------------------------------------------------------
class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[str] = mapped_column(String(64), unique=True)
    model_type: Mapped[str] = mapped_column(String(48))
    cohort: Mapped[str] = mapped_column(String(12))
    data_hash: Mapped[str] = mapped_column(String(64))
    params_json: Mapped[dict | None] = mapped_column(JSON)
    features_json: Mapped[list | None] = mapped_column(JSON)
    metrics_json: Mapped[dict | None] = mapped_column(JSON)
    train_quarters: Mapped[list | None] = mapped_column(JSON)
    validation_quarters: Mapped[list | None] = mapped_column(JSON)
    artifact_path: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    run_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# 7 ----------------------------------------------------------------------
class RiskSegment(Base):
    __tablename__ = "risk_segments"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    segment_id: Mapped[int] = mapped_column(Integer)
    leaf_node_id: Mapped[int] = mapped_column(Integer)
    path_json: Mapped[list] = mapped_column(JSON)
    path_text: Mapped[str] = mapped_column(Text)
    n_train: Mapped[int] = mapped_column(Integer)
    defaults_train: Mapped[int] = mapped_column(Integer)
    rate_train: Mapped[float | None] = mapped_column(Float)
    n_valid: Mapped[int] = mapped_column(Integer)
    defaults_valid: Mapped[int] = mapped_column(Integer)
    rate_valid: Mapped[float | None] = mapped_column(Float)
    stats_json: Mapped[dict | None] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("model_id", "segment_id"),)


# 9 (declared before 8 for FK) ---------------------------------------------
class PolicyVersion(Base):
    __tablename__ = "policy_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    version_number: Mapped[str] = mapped_column(String(16), unique=True)
    parent_version: Mapped[str | None] = mapped_column(String(16))
    label: Mapped[str | None] = mapped_column(String(64))  # baseline | proposed | challenger
    status: Mapped[str] = mapped_column(String(24), default="Draft")
    deployment_status: Mapped[str] = mapped_column(String(24), default="Not Deployed")
    is_locked: Mapped[bool] = mapped_column(Boolean, default=False)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    basis_hash: Mapped[str | None] = mapped_column(String(64), index=True)  # data hash + parent version
    profile: Mapped[str] = mapped_column(String(16), default="production")
    description: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    approved_by: Mapped[str | None] = mapped_column(String(64))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime)
    approval_comments: Mapped[str | None] = mapped_column(Text)
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime)
    source_run_id: Mapped[str | None] = mapped_column(String(64))
    evidence_json: Mapped[dict | None] = mapped_column(JSON)
    rules: Mapped[list["PolicyRule"]] = relationship(
        back_populates="version", order_by="PolicyRule.execution_order", cascade="all, delete-orphan"
    )


# 8 ----------------------------------------------------------------------
class PolicyRule(Base):
    __tablename__ = "policy_rules"
    id: Mapped[int] = mapped_column(primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("policy_versions.id"), index=True)
    rule_code: Mapped[str] = mapped_column(String(16))
    execution_order: Mapped[int] = mapped_column(Integer)
    parent_rule: Mapped[str | None] = mapped_column(String(16))
    conditions_json: Mapped[list] = mapped_column(JSON)
    treatment: Mapped[str] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(Text)
    evidence_json: Mapped[dict | None] = mapped_column(JSON)
    version: Mapped[PolicyVersion] = relationship(back_populates="rules")
    __table_args__ = (UniqueConstraint("version_id", "rule_code"),)


# 10 ---------------------------------------------------------------------
class PolicyDecision(Base):
    __tablename__ = "policy_decisions"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    version_number: Mapped[str] = mapped_column(String(16), index=True)
    loan_account_no: Mapped[str] = mapped_column(String(32), index=True)
    matched_rule: Mapped[str] = mapped_column(String(16))
    treatment: Mapped[str] = mapped_column(String(64))
    risk_segment: Mapped[int | None] = mapped_column(Integer)
    probability_of_default: Mapped[float | None] = mapped_column(Float)
    explanation: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# 11 ---------------------------------------------------------------------
class PolicyApproval(Base):
    __tablename__ = "policy_approvals"
    id: Mapped[int] = mapped_column(primary_key=True)
    version_number: Mapped[str] = mapped_column(String(16), index=True)
    action: Mapped[str] = mapped_column(String(32))
    from_status: Mapped[str | None] = mapped_column(String(24))
    to_status: Mapped[str] = mapped_column(String(24))
    actor: Mapped[str] = mapped_column(String(64))
    comments: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# 12 ---------------------------------------------------------------------
class BacktestResult(Base):
    __tablename__ = "backtest_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    baseline_version: Mapped[str] = mapped_column(String(16))
    candidate_version: Mapped[str] = mapped_column(String(16))
    scope: Mapped[str] = mapped_column(String(32))  # holdout | all_mature | quarter:2025Q2
    cohort: Mapped[str | None] = mapped_column(String(12))
    metrics_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# 13 ---------------------------------------------------------------------
class AgentExecutionLog(Base):
    __tablename__ = "agent_execution_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    agent_name: Mapped[str] = mapped_column(String(48), index=True)
    operation: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))  # RUNNING | SUCCESS | FAILED | SKIPPED
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    input_json: Mapped[dict | None] = mapped_column(JSON)
    output_json: Mapped[dict | None] = mapped_column(JSON)
    findings_json: Mapped[list | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    duration_ms: Mapped[int | None] = mapped_column(Integer)


# 14 ---------------------------------------------------------------------
class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    entity_type: Mapped[str | None] = mapped_column(String(32))
    entity_id: Mapped[str | None] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[str | None] = mapped_column(String(64))
    details_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


# 15 ---------------------------------------------------------------------
class RefreshRun(Base):
    __tablename__ = "refresh_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(16))  # RUNNING | COMPLETED | NO_CHANGE | FAILED
    triggered_by: Mapped[str] = mapped_column(String(64))
    data_hash: Mapped[str | None] = mapped_column(String(64))
    base_policy_version: Mapped[str | None] = mapped_column(String(16))
    created_policy_version: Mapped[str | None] = mapped_column(String(16))
    stages_json: Mapped[list | None] = mapped_column(JSON)
    summary_json: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


Index("ix_repay_loan_due", RepaymentHistory.loan_account_no, RepaymentHistory.due_month)

ALL_TABLES = [m.__tablename__ for m in Base.__subclasses__()]
