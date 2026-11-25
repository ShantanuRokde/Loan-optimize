"""Project-wide constants. Values here describe *schemas and vocabularies*,
never financial results — every metric is computed from the source data."""
from __future__ import annotations

APP_NAME = "CreditRisk AI – Intelligent Loan Portfolio & Policy Optimization"

# ---------------------------------------------------------------- datasets
DATASET_APPLICATION = "application"
DATASET_DISBURSEMENT = "disbursement"
DATASET_REPAYMENT = "repayment"
DATASETS = (DATASET_APPLICATION, DATASET_DISBURSEMENT, DATASET_REPAYMENT)

KEY = "loan_account_no"

# Expected schema: column -> logical type (str | date | int | float | flag)
EXPECTED_SCHEMAS: dict[str, dict[str, str]] = {
    DATASET_APPLICATION: {
        "loan_account_no": "str",
        "application_date": "date",
        "customer_segment": "str",
        "occupation": "str",
        "income_band": "str",
        "credit_score": "int",
        "enquiries_6m": "int",
        "existing_loans": "int",
        "age": "int",
        "region": "str",
        "state": "str",
        "branch": "str",
        "loan_type": "str",
        "product_code": "str",
        "requested_amount": "float",
    },
    DATASET_DISBURSEMENT: {
        "loan_account_no": "str",
        "disbursal_date": "date",
        "disbursal_quarter": "str",
        "sanction_amount": "float",
        "loan_amount": "float",
        "emi_amount": "float",
        "loan_tenure_months": "int",
        "interest_rate": "float",
    },
    DATASET_REPAYMENT: {
        "loan_account_no": "str",
        "due_month": "date",
        "emi_amount": "float",
        "paid_flag": "flag",
        "days_past_due": "int",
        "outstanding_balance": "float",
    },
}

# Natural keys used for duplicate detection
NATURAL_KEYS: dict[str, list[str]] = {
    DATASET_APPLICATION: [KEY],
    DATASET_DISBURSEMENT: [KEY],
    DATASET_REPAYMENT: [KEY, "due_month"],
}

# Plausibility ranges for numeric fields (inclusive). Breaches are flagged, not removed.
NUMERIC_RANGES: dict[str, dict[str, tuple[float | None, float | None]]] = {
    DATASET_APPLICATION: {
        "credit_score": (300, 900),
        "enquiries_6m": (0, 50),
        "existing_loans": (0, 50),
        "age": (18, 100),
        "requested_amount": (1, None),
    },
    DATASET_DISBURSEMENT: {
        "sanction_amount": (1, None),
        "loan_amount": (1, None),
        "emi_amount": (1, None),
        "loan_tenure_months": (1, 480),
        "interest_rate": (0, 60),
    },
    DATASET_REPAYMENT: {
        "emi_amount": (0, None),
        "days_past_due": (0, None),
        "outstanding_balance": (0, None),
    },
}

PAID_FLAG_VALUES = {"Y", "N"}

# Income band vocabulary observed in the data and its documented mapping.
# No rupee midpoints are assumed: the ordinal is a rank, not an amount.
INCOME_BAND_ORDER = ["<5L", "5L-10L", "10L-25L", "25L+"]
INCOME_BAND_ORDINAL = {b: i + 1 for i, b in enumerate(INCOME_BAND_ORDER)}
INCOME_BANDS_GE_10L = {"10L-25L", "25L+"}      # income >= ₹10 lakh
INCOME_BANDS_LT_10L = {"<5L", "5L-10L"}        # income <  ₹10 lakh

# ------------------------------------------------------------------ DQ
SEV_ERROR = "ERROR"      # record cannot be used for analytics (kept, flagged)
SEV_WARNING = "WARNING"  # record usable, issue recorded / repaired
SEV_INFO = "INFO"        # observation only

# ------------------------------------------------------------- delinquency
DPD_BUCKETS = [  # (label, lower inclusive, upper inclusive)
    ("Current", 0, 0),
    ("1-30", 1, 30),
    ("31-60", 31, 60),
    ("61-90", 61, 90),
    ("90+", 91, None),
]
DPD_THRESHOLDS = (30, 60, 90)
COHORT_WINDOWS = (6, 12, 18)
COHORTS = [f"{t}DPD{w:02d}" for t in DPD_THRESHOLDS for w in COHORT_WINDOWS]

# --------------------------------------------------------------- policy
TREATMENT_STANDARD = "Standard processing"
TREATMENT_MONITOR = "Standard processing with monitoring"
TREATMENT_ENHANCED = "Enhanced enquiry checks"
TREATMENT_FULL_REVIEW = "Full underwriting review"
TREATMENTS = [TREATMENT_STANDARD, TREATMENT_MONITOR, TREATMENT_ENHANCED, TREATMENT_FULL_REVIEW]
STANDARD_TRACK = {TREATMENT_STANDARD, TREATMENT_MONITOR}  # "approved" without enhanced checks

POLICY_STATUSES = ["Draft", "Review Pending", "Approved", "Deployment Ready", "Deployed", "Rejected", "Returned", "Superseded"]

AGENT_NAMES = [
    "data_intelligence",
    "portfolio_monitoring",
    "cohort_analysis",
    "risk_discovery",
    "policy_optimization",
    "policy_management",
    "policy_backtesting",
]
