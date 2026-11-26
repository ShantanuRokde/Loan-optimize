"""Application configuration.

Resolution order (later wins):
  1. defaults declared in the Pydantic models below
  2. config/settings.json (written by the Settings page)
  3. environment variables (CREDITRISK_* / OPENAI_API_KEY / OLLAMA_*), see .env.example

Production and demo qualification thresholds are kept as *separate named
profiles*. The demo profile is never applied silently: every result produced
under it carries `profile="demo"` and is labelled DEMO in the UI.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.json"

load_dotenv(PROJECT_ROOT / ".env", override=False)


class DatasetConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    reports_dir: str = "data/reports"
    application_file: str = "table_APPLICATION (1).csv"
    disbursement_file: str = "table_DISBURSEMENT (2).csv"
    repayment_file: str = "table_REPAYMENT.csv"


class DelinquencyConfig(BaseModel):
    npa_dpd_threshold: int = Field(90, ge=1, description="DPD at/above which an account is treated as NPA proxy / default")
    overdue_dpd_threshold: int = Field(1, ge=1)


class QualificationThresholds(BaseModel):
    min_loans: int = Field(50, ge=1, description="Minimum loans in the candidate segment (training)")
    min_portfolio_share: float = Field(0.05, ge=0, le=1, description="Minimum share of the training portfolio")
    min_defaults: int = Field(5, ge=0, description="Minimum defaults in the candidate segment (training)")
    min_lift: float = Field(1.5, ge=1, description="Candidate default rate / parent-rule default rate")
    require_holdout: bool = Field(True, description="Rule must remain risky on the held-out quarter")
    min_holdout_loans: int = Field(10, ge=1)
    min_holdout_lift: float = Field(1.0, ge=0, description="Held-out lift must be >= this (and > parent rate)")


class CohortConfig(BaseModel):
    selected_cohort: str | None = Field(None, description="None = auto-select weakest eligible cohort")
    min_mature_quarters: int = Field(3, ge=2)
    min_cohort_defaults: int = Field(20, ge=1, description="Minimum defaults across mature quarters for a cohort to be ranked")
    control_limit_sd: Literal[1, 2] = 2


class TreeConfig(BaseModel):
    max_depth: int = Field(4, ge=1, le=12)
    min_samples_leaf: int = Field(40, ge=1)
    class_weight: Literal["balanced", "none"] = "balanced"
    random_state: int = 42
    train_challenger: bool = True


class LLMConfig(BaseModel):
    provider: Literal["none", "openai", "ollama"] = "none"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    timeout_seconds: int = 30


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///data/creditrisk.db"


class AppConfig(BaseModel):
    datasets: DatasetConfig = DatasetConfig()
    delinquency: DelinquencyConfig = DelinquencyConfig()
    cohort: CohortConfig = CohortConfig()
    tree: TreeConfig = TreeConfig()
    active_profile: Literal["production", "demo"] = "production"
    thresholds: dict[str, QualificationThresholds] = Field(
        default_factory=lambda: {
            "production": QualificationThresholds(),
            "demo": QualificationThresholds(
                min_loans=20, min_portfolio_share=0.01, min_defaults=3, min_lift=1.3, min_holdout_loans=5
            ),
        }
    )
    llm: LLMConfig = LLMConfig()
    database: DatabaseConfig = DatabaseConfig()
    default_actor: str = "analyst"

    # ---------------------------------------------------------- helpers
    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def raw_dir(self) -> Path:
        return self.path(self.datasets.raw_dir)

    @property
    def processed_dir(self) -> Path:
        return self.path(self.datasets.processed_dir)

    @property
    def reports_dir(self) -> Path:
        return self.path(self.datasets.reports_dir)

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "artifacts" / "models"

    @property
    def artifact_reports_dir(self) -> Path:
        return PROJECT_ROOT / "artifacts" / "reports"

    def source_files(self) -> dict[str, Path]:
        d = self.datasets
        return {
            "application": self.raw_dir / d.application_file,
            "disbursement": self.raw_dir / d.disbursement_file,
            "repayment": self.raw_dir / d.repayment_file,
        }

    def active_thresholds(self) -> QualificationThresholds:
        return self.thresholds[self.active_profile]

    def database_url(self) -> str:
        url = self.database.url
        prefix = "sqlite:///"
        if url.startswith(prefix) and not url.startswith(prefix + "/") and ":memory:" not in url:
            rel = url[len(prefix):]
            if not Path(rel).is_absolute():
                return prefix + (PROJECT_ROOT / rel).as_posix()
        return url


_lock = threading.Lock()
_config: AppConfig | None = None


def _apply_env(cfg: AppConfig) -> AppConfig:
    if v := os.getenv("CREDITRISK_DB_URL"):
        cfg.database.url = v
    if v := os.getenv("CREDITRISK_RAW_DIR"):
        cfg.datasets.raw_dir = v
    if v := os.getenv("CREDITRISK_LLM_PROVIDER"):
        if v in ("none", "openai", "ollama"):
            cfg.llm.provider = v  # type: ignore[assignment]
    if v := os.getenv("OLLAMA_BASE_URL"):
        cfg.llm.ollama_base_url = v
    if v := os.getenv("OLLAMA_MODEL"):
        cfg.llm.ollama_model = v
    if v := os.getenv("OPENAI_MODEL"):
        cfg.llm.openai_model = v
    return cfg


def load_config() -> AppConfig:
    data = {}
    if SETTINGS_FILE.exists():
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    return _apply_env(AppConfig.model_validate(data))


def get_config() -> AppConfig:
    global _config
    with _lock:
        if _config is None:
            _config = load_config()
        return _config


def save_config(cfg: AppConfig) -> AppConfig:
    """Validate and persist settings (secrets are never written — they live in env)."""
    global _config
    cfg = AppConfig.model_validate(cfg.model_dump())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(cfg.model_dump(), indent=2), encoding="utf-8")
    with _lock:
        _config = cfg
    return cfg


def set_config(cfg: AppConfig) -> None:
    """Override in-memory config (used by tests)."""
    global _config
    with _lock:
        _config = cfg


def reset_config() -> None:
    global _config
    with _lock:
        _config = None
