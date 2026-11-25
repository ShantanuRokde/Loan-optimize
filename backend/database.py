"""Engine / session factory. A single SQLite file by default (data/creditrisk.db)."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config import get_config
from database.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None
_engine_url: str | None = None


def _sqlite_pragmas(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.close()


def get_engine(url: str | None = None) -> Engine:
    global _engine, _SessionLocal, _engine_url
    url = url or get_config().database_url()
    if _engine is None or url != _engine_url:
        if url.startswith("sqlite:///") and ":memory:" not in url:
            Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
        _engine = create_engine(url, future=True, **kwargs)
        if url.startswith("sqlite"):
            event.listen(_engine, "connect", _sqlite_pragmas)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        _engine_url = url
    return _engine


def init_db(url: str | None = None) -> Engine:
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def session_scope() -> Iterator[Session]:
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    s: Session = _SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def dispose_engine() -> None:
    global _engine, _SessionLocal, _engine_url
    if _engine is not None:
        _engine.dispose()
    _engine = _SessionLocal = _engine_url = None
