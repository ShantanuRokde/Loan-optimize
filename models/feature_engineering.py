"""Feature engineering for the risk-discovery model.

Design goals
------------
* **Leakage-safe whitelist.** Only application-time / booking-time attributes
  are allowed. Account numbers, repayment-derived fields and anything computed
  from the target are rejected by `assert_no_leakage`.
* **Decodable encoding.** Each model column maps back to exactly one source
  field so tree splits can be translated into policy conditions:
    - numeric       -> used as-is                      (credit_score <= 649.5)
    - ordinal       -> income_band rank 1..4           (income_band in [...])
    - categorical   -> one-hot `cat__{field}__{level}` (occupation == ...)
  Unknown categories at scoring time encode to all-zeros (documented).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from utils.constants import INCOME_BAND_ORDER, INCOME_BAND_ORDINAL

NUMERIC_FEATURES = ["credit_score", "enquiries_6m", "existing_loans", "age", "requested_amount",
                    "loan_amount", "interest_rate", "loan_tenure_months"]
ORDINAL_FEATURES = {"income_band": INCOME_BAND_ORDINAL}
CATEGORICAL_FEATURES = ["customer_segment", "occupation", "loan_type", "region"]
ALLOWED_FEATURES = NUMERIC_FEATURES + list(ORDINAL_FEATURES) + CATEGORICAL_FEATURES

# integer-valued source fields: splits are expressed as `< ceil(t)` / `>= ceil(t)`
INTEGER_FEATURES = {"credit_score", "enquiries_6m", "existing_loans", "age", "loan_tenure_months"}

# Never allowed as model inputs (identifiers, outcome / repayment information, target-derived)
FORBIDDEN_PATTERNS = ("loan_account_no", "dpd", "paid", "missed", "outstanding", "installment", "instalment",
                      "mob", "npa", "target", "default", "breach", "due", "emi", "cutoff", "dq_", "build_run")


def assert_no_leakage(features: list[str]) -> None:
    bad = [f for f in features if f not in ALLOWED_FEATURES or any(p in f.lower() for p in FORBIDDEN_PATTERNS)]
    if bad:
        raise ValueError(f"Leakage guard: features not permitted as model inputs: {bad}")


@dataclass
class FeatureSpec:
    numeric: list[str]
    ordinal: dict[str, dict[str, int]]
    categorical: dict[str, list[str]]          # field -> levels seen in TRAINING data
    medians: dict[str, float] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        return cls(**d)

    @property
    def source_features(self) -> list[str]:
        return self.numeric + list(self.ordinal) + list(self.categorical)

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.numeric:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.medians.get(c, 0.0)).astype(float)
        for c, mapping in self.ordinal.items():
            out[f"{c}__ord"] = df[c].map(mapping).fillna(self.medians.get(f"{c}__ord", 0.0)).astype(float)
        for c, levels in self.categorical.items():
            vals = df[c].astype("string")
            for lvl in levels:
                out[f"cat__{c}__{lvl}"] = (vals == lvl).fillna(False).astype(float)
        X = pd.DataFrame(out, index=df.index)
        return X[self.columns] if self.columns else X

    # ------------------------------------------------------------------
    def column_source(self, col: str) -> tuple[str, str, str | None]:
        """model column -> (source field, kind, level)."""
        if col.startswith("cat__"):
            _, f, lvl = col.split("__", 2)
            return f, "categorical", lvl
        if col.endswith("__ord"):
            return col[:-5], "ordinal", None
        return col, "numeric", None


def build_feature_spec(train: pd.DataFrame, features: list[str] | None = None) -> FeatureSpec:
    features = features or ALLOWED_FEATURES
    assert_no_leakage(features)
    numeric = [f for f in NUMERIC_FEATURES if f in features]
    ordinal = {f: m for f, m in ORDINAL_FEATURES.items() if f in features}
    categorical = {f: sorted(train[f].dropna().astype(str).unique().tolist()) for f in CATEGORICAL_FEATURES if f in features}
    medians = {c: float(pd.to_numeric(train[c], errors="coerce").median()) for c in numeric}
    for c, m in ordinal.items():
        medians[f"{c}__ord"] = float(train[c].map(m).median())
    spec = FeatureSpec(numeric=numeric, ordinal=ordinal, categorical=categorical, medians=medians)
    spec.columns = list(spec.transform(train.head(1)).columns)
    return spec


# ------------------------------------------------- condition decoding
def split_to_conditions(spec: FeatureSpec, col: str, threshold: float) -> tuple[dict, dict]:
    """Translate a sklearn split (x <= t goes left) into (left_condition, right_condition)
    in the rules-engine format {field, op, value}."""
    src, kind, lvl = spec.column_source(col)
    if kind == "categorical":       # one-hot: <= 0.5 means "not this level"
        return ({"field": src, "op": "not_in", "value": [lvl]}, {"field": src, "op": "in", "value": [lvl]})
    if kind == "ordinal":
        mapping = spec.ordinal[src]
        order = sorted(mapping, key=mapping.get)
        left = [b for b in order if mapping[b] <= threshold]
        right = [b for b in order if mapping[b] > threshold]
        return ({"field": src, "op": "in", "value": left}, {"field": src, "op": "in", "value": right})
    if src in INTEGER_FEATURES:
        cut = int(np.floor(threshold)) + 1          # x <= 5.5  <=>  x < 6
        return ({"field": src, "op": "<", "value": cut}, {"field": src, "op": ">=", "value": cut})
    t = round(float(threshold), 4)
    return ({"field": src, "op": "<=", "value": t}, {"field": src, "op": ">", "value": t})


def simplify_conditions(conds: list[dict], spec: FeatureSpec) -> list[dict]:
    """Merge redundant conditions on the same field (tightest bounds, set intersection)."""
    out: dict[str, dict] = {}
    order: list[str] = []
    lower: dict[str, tuple[str, float]] = {}
    upper: dict[str, tuple[str, float]] = {}
    sets: dict[str, set] = {}
    for c in conds:
        f, op, v = c["field"], c["op"], c["value"]
        if f not in order:
            order.append(f)
        if op in (">", ">="):
            cur = lower.get(f)
            if cur is None or v > cur[1] or (v == cur[1] and op == ">"):
                lower[f] = (op, v)
        elif op in ("<", "<="):
            cur = upper.get(f)
            if cur is None or v < cur[1] or (v == cur[1] and op == "<"):
                upper[f] = (op, v)
        elif op in ("in", "not_in"):
            universe = set(spec.categorical.get(f) or spec.ordinal.get(f, {}).keys())
            cur = sets.get(f, set(universe))
            sets[f] = cur & set(v) if op == "in" else cur - set(v)
    result = []
    for f in order:
        if f in lower:
            result.append({"field": f, "op": lower[f][0], "value": lower[f][1]})
        if f in upper:
            result.append({"field": f, "op": upper[f][0], "value": upper[f][1]})
        if f in sets:
            levels = spec.categorical.get(f) or sorted(spec.ordinal.get(f, {}), key=lambda b: spec.ordinal[f][b])
            result.append({"field": f, "op": "in", "value": [l for l in levels if l in sets[f]]})
    return result


def condition_text(c: dict) -> str:
    v = c["value"]
    if c["op"] in ("in", "not_in"):
        word = "in" if c["op"] == "in" else "not in"
        return f"{c['field']} {word} [{', '.join(map(str, v))}]"
    if isinstance(v, float) and abs(v) >= 1000:
        v = f"{v:,.0f}"
    return f"{c['field']} {c['op']} {v}"


def conditions_text(conds: list[dict]) -> str:
    return " AND ".join(condition_text(c) for c in conds) if conds else "ALL LOANS"


def income_band_label(ordinal: float) -> str | None:
    if ordinal is None or np.isnan(ordinal):
        return None
    i = int(round(ordinal)) - 1
    return INCOME_BAND_ORDER[i] if 0 <= i < len(INCOME_BAND_ORDER) else None
