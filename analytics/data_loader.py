"""CSV ingestion.

Files are read **as text** (dtype=str) so nothing is coerced or lost at load
time; typing happens in data_validation where every failure is logged.
The raw files are opened read-only and never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from utils.helpers import file_sha256
from utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class LoadedDataset:
    name: str
    path: Path
    frame: pd.DataFrame | None          # raw text values, untouched except surrounding whitespace
    file_hash: str | None = None
    size_bytes: int = 0
    status: str = "LOADED"
    message: str | None = None
    columns: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "LOADED" and self.frame is not None

    def meta(self) -> dict:
        return {
            "dataset_name": self.name,
            "file_path": str(self.path),
            "file_hash": self.file_hash or "",
            "file_size_bytes": self.size_bytes,
            "row_count": 0 if self.frame is None else len(self.frame),
            "column_count": len(self.columns),
            "columns_json": self.columns,
            "status": self.status,
            "message": self.message,
        }


def load_csv(name: str, path: str | Path) -> LoadedDataset:
    path = Path(path)
    if not path.exists():
        msg = f"Source file not found: {path}"
        log.error(msg)
        return LoadedDataset(name, path, None, status="FAILED", message=msg)
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    except Exception as exc:  # malformed CSV, encoding error, ...
        log.exception("Failed to parse %s", path)
        return LoadedDataset(name, path, None, status="FAILED", message=f"Parse error: {exc}")
    df.columns = [c.replace("﻿", "").strip().strip('"') for c in df.columns]
    # Values are space-padded in the source (e.g. "         754"); trim only.
    for c in df.columns:
        df[c] = df[c].str.strip()
    df.insert(0, "_row_number", range(1, len(df) + 1))  # 1-based data row in the source file
    return LoadedDataset(
        name=name, path=path, frame=df, file_hash=file_sha256(path), size_bytes=path.stat().st_size,
        columns=[c for c in df.columns if c != "_row_number"],
    )


def load_sources(files: dict[str, Path]) -> dict[str, LoadedDataset]:
    return {name: load_csv(name, p) for name, p in files.items()}


def combined_hash(datasets: dict[str, LoadedDataset]) -> str:
    """Fingerprint of the full source data set (used for idempotency)."""
    import hashlib

    h = hashlib.sha256()
    for name in sorted(datasets):
        h.update(f"{name}:{datasets[name].file_hash}".encode())
    return h.hexdigest()
