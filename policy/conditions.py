"""Single condition evaluator shared by tree segmentation, policy analysis and
the rules engine – so the same rule always produces the same decision.

Condition format: {"field": str, "op": str, "value": scalar | list}
Supported ops: <, <=, >, >=, ==, !=, in, not_in.
Missing values never satisfy a condition (fail-closed: the loan falls through
to later / default rules).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OPS = ("<", "<=", ">", ">=", "==", "!=", "in", "not_in")


def validate_condition(c: dict) -> None:
    if not isinstance(c, dict) or {"field", "op", "value"} - set(c):
        raise ValueError(f"Malformed condition: {c}")
    if c["op"] not in OPS:
        raise ValueError(f"Unsupported operator '{c['op']}' in {c}")
    if c["op"] in ("in", "not_in") and not isinstance(c["value"], (list, tuple, set)):
        raise ValueError(f"Operator {c['op']} requires a list value: {c}")


def condition_mask(df: pd.DataFrame, c: dict) -> pd.Series:
    validate_condition(c)
    if c["field"] not in df.columns:
        raise KeyError(f"Field '{c['field']}' not present in input data")
    s = df[c["field"]]
    op, v = c["op"], c["value"]
    present = s.notna()
    if op in ("in", "not_in"):
        m = s.astype("string").isin([str(x) for x in v])
        m = m if op == "in" else ~m
    else:
        num = pd.to_numeric(s, errors="coerce") if not isinstance(v, str) else s
        m = {"<": num < v, "<=": num <= v, ">": num > v, ">=": num >= v, "==": num == v, "!=": num != v}[op]
    return (m.fillna(False).astype(bool)) & present


def conditions_mask(df: pd.DataFrame, conds: list[dict]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for c in conds:
        m &= condition_mask(df, c)
    return m


def evaluate_row(row: dict, conds: list[dict]) -> bool:
    return bool(conditions_mask(pd.DataFrame([row]), conds).iloc[0])
