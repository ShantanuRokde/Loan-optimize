"""Common agent contract.

Every agent:
  * has a stable `name` and human `title`
  * implements `execute(ctx, **params) -> AgentResult` with real logic
  * is invoked through `run()`, which times the call and persists an
    agent_execution_logs row (input, output summary, findings, errors).
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from database import repository as repo
from utils.helpers import new_run_id, to_jsonable
from utils.logging_config import get_logger


@dataclass
class AgentResult:
    status: str                       # SUCCESS | FAILED | SKIPPED | NO_CHANGE
    output: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0
    log_id: int | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("SUCCESS", "NO_CHANGE", "SKIPPED")

    def to_dict(self) -> dict:
        return to_jsonable({
            "status": self.status, "output": self.output, "findings": self.findings,
            "error": self.error, "duration_ms": self.duration_ms, "log_id": self.log_id,
        })


def save_report(name: str, payload: dict) -> str:
    """Persist an agent's full result payload as JSON (served to the UI / Copilot)."""
    import json

    from backend.config import get_config

    d = get_config().reports_dir
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps(to_jsonable(payload), indent=1), encoding="utf-8")
    return str(p)


def load_report(name: str) -> dict | None:
    import json

    from backend.config import get_config

    p = get_config().reports_dir / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


class AgentError(RuntimeError):
    """Raised by an agent for an expected, explainable failure."""


class BaseAgent:
    name: str = "base"
    title: str = "Base Agent"
    description: str = ""
    writes_state: bool = False          # does execute() write to DB / disk?
    retry_safe: bool = True             # may the orchestrator retry it?

    def __init__(self) -> None:
        self.log = get_logger(f"agent.{self.name}")
        self.current_operation: str | None = None

    def execute(self, ctx: dict, **params) -> AgentResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def summarise_input(self, ctx: dict, params: dict) -> dict:
        return {"params": params, "context_keys": sorted(k for k in ctx if not k.startswith("_"))}

    def run(self, ctx: dict | None = None, *, run_id: str | None = None, attempt: int = 1, **params) -> AgentResult:
        ctx = ctx if ctx is not None else {}
        run_id = run_id or ctx.get("run_id") or new_run_id(self.name)
        ctx.setdefault("run_id", run_id)
        self.current_operation = f"{self.title}: starting"
        log_id = repo.start_agent_log(run_id, self.name, self.title, self.summarise_input(ctx, params), attempt)
        t0 = time.perf_counter()
        try:
            res = self.execute(ctx, **params)
        except AgentError as exc:
            self.log.warning("%s failed: %s", self.name, exc)
            res = AgentResult(status="FAILED", error=str(exc), findings=[str(exc)])
        except Exception as exc:  # unexpected
            self.log.exception("%s crashed", self.name)
            res = AgentResult(status="FAILED", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}")
        res.duration_ms = int((time.perf_counter() - t0) * 1000)
        res.log_id = log_id
        repo.finish_agent_log(log_id, res.status, res.output, res.findings, res.error)
        self.current_operation = f"{self.title}: {res.status.lower()}"
        return res

    def step(self, message: str, steps: list[dict], t0: float) -> None:
        """Record a processing-log step (shown in the UI and the DQ report)."""
        self.current_operation = message
        steps.append({"step": message, "elapsed_ms": int((time.perf_counter() - t0) * 1000)})
        self.log.info(message)
