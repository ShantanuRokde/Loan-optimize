"""Small, dependency-free helpers shared across modules."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{utcnow():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:6]}"


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    """Deterministic hash of a JSON-serialisable object (sorted keys)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def months_between(start: pd.Series | pd.Timestamp, end: pd.Series | pd.Timestamp):
    """Whole calendar months from start to end (end.month - start.month + 12*years)."""
    if isinstance(start, pd.Series) or isinstance(end, pd.Series):
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        sy = s.dt.year if isinstance(s, pd.Series) else s.year
        sm = s.dt.month if isinstance(s, pd.Series) else s.month
        ey = e.dt.year if isinstance(e, pd.Series) else e.year
        em = e.dt.month if isinstance(e, pd.Series) else e.month
        return (ey - sy) * 12 + (em - sm)
    return (end.year - start.year) * 12 + (end.month - start.month)


def to_jsonable(obj: Any) -> Any:
    """Convert numpy / pandas objects into plain JSON-safe Python values."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return None if pd.isna(obj) else obj.isoformat()
    if obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def safe_div(num: float, den: float) -> float | None:
    return None if not den else float(num) / float(den)
