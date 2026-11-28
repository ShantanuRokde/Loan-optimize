"""Model registry: persistence, signatures and retrain detection.

A model's *signature* is a hash of everything that determines it:
source-data hash, cohort, train/validation quarters, hyper-parameters and
feature list. If an artifact with the same signature exists, retraining is
unnecessary (the refresh workflow reuses it); any change forces a retrain.
"""
from __future__ import annotations

from pathlib import Path

import joblib
from sqlalchemy import select, update

from backend.config import get_config
from backend.database import session_scope
from database.models import ModelRegistry, RiskSegment
from utils.helpers import stable_hash, to_jsonable


def model_signature(data_hash: str | None, cohort: str, train_q: list[str], valid_q: list[str], params: dict) -> str:
    return stable_hash({"data": data_hash, "cohort": cohort, "train": sorted(train_q), "valid": sorted(valid_q),
                        "params": params, "model": "DecisionTreeClassifier", "v": 1})


def model_id_for(cohort: str, signature: str) -> str:
    return f"dt_{cohort}_{signature[:12]}"


def find_model(model_id: str) -> dict | None:
    with session_scope() as s:
        r = s.scalar(select(ModelRegistry).where(ModelRegistry.model_id == model_id))
        if r is None:
            return None
        return {c.name: getattr(r, c.name) for c in r.__table__.columns}


def needs_retraining(model_id: str) -> tuple[bool, str]:
    rec = find_model(model_id)
    if rec is None:
        return True, "No registered model with this data/cohort/parameter signature"
    if not Path(rec["artifact_path"]).exists():
        return True, "Registered model artifact missing on disk"
    return False, f"Registered model {model_id} matches current data, cohort and parameters"


def save_bundle(model_id: str, bundle: dict) -> str:
    d = get_config().models_dir
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{model_id}.joblib"
    joblib.dump(bundle, p)
    return str(p)


def load_bundle(model_id: str) -> dict:
    rec = find_model(model_id)
    if rec is None:
        raise FileNotFoundError(f"Model {model_id} not registered")
    return joblib.load(rec["artifact_path"])


def register(model_id: str, cohort: str, data_hash: str | None, params: dict, features: list[str], metrics: dict,
             train_q: list[str], valid_q: list[str], artifact_path: str, segments: list[dict], run_id: str | None) -> None:
    """Insert registry row + segments and mark this model active (others deactivated)."""
    with session_scope() as s:
        s.execute(update(ModelRegistry).values(is_active=False))
        existing = s.scalar(select(ModelRegistry).where(ModelRegistry.model_id == model_id))
        if existing is None:
            s.add(ModelRegistry(model_id=model_id, model_type="DecisionTreeClassifier", cohort=cohort,
                                data_hash=data_hash or "", params_json=to_jsonable(params), features_json=features,
                                metrics_json=to_jsonable(metrics), train_quarters=train_q, validation_quarters=valid_q,
                                artifact_path=artifact_path, is_active=True, run_id=run_id))
            for seg in segments:
                s.add(RiskSegment(model_id=model_id, segment_id=seg["segment_id"], leaf_node_id=seg["leaf_node_id"],
                                  path_json=to_jsonable(seg["conditions"]), path_text=seg["path_text"],
                                  n_train=seg["train"]["loans"], defaults_train=seg["train"]["defaults"],
                                  rate_train=seg["train"]["default_rate"], n_valid=seg["validation"]["loans"],
                                  defaults_valid=seg["validation"]["defaults"], rate_valid=seg["validation"]["default_rate"],
                                  stats_json=to_jsonable({k: seg[k] for k in ("train", "validation", "all_loans",
                                                                              "lift_train", "lift_valid")})))
        else:
            existing.is_active = True


def active_model() -> dict | None:
    with session_scope() as s:
        r = s.scalar(select(ModelRegistry).where(ModelRegistry.is_active.is_(True)).order_by(ModelRegistry.id.desc()))
        return None if r is None else {c.name: getattr(r, c.name) for c in r.__table__.columns}


def list_models(limit: int = 50) -> list[dict]:
    with session_scope() as s:
        rows = s.scalars(select(ModelRegistry).order_by(ModelRegistry.id.desc()).limit(limit))
        return [to_jsonable({c.name: getattr(r, c.name) for c in r.__table__.columns}) for r in rows]


def segments_for(model_id: str) -> list[dict]:
    with session_scope() as s:
        rows = s.scalars(select(RiskSegment).where(RiskSegment.model_id == model_id).order_by(RiskSegment.segment_id))
        return [to_jsonable({c.name: getattr(r, c.name) for c in r.__table__.columns}) for r in rows]
