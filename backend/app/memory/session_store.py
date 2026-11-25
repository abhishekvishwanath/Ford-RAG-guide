"""Session repository interface + SQLite implementation.

The ``SessionRepository`` protocol is what the rest of the app depends on, so a
PostgreSQL or Redis implementation can be swapped in without touching callers.
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from sqlalchemy import create_engine, delete, event, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.conversation.state import SessionState
from app.memory.memory_models import Base, MessageRow, SessionRow


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class SessionRepository(Protocol):
    def create(self) -> SessionState: ...
    def get(self, session_id: str) -> SessionState | None: ...
    def save(self, state: SessionState) -> None: ...
    def delete(self, session_id: str) -> bool: ...
    def append_message(self, session_id: str, role: str, content: str, meta: dict | None = None) -> None: ...
    def recent_messages(self, session_id: str, limit: int | None = None) -> list[dict]: ...
    def purge_expired(self) -> int: ...


class SqliteSessionRepository:
    def __init__(self, database_url: str, message_limit: int = 20, expiry_days: int = 7) -> None:
        if database_url.startswith("sqlite:///"):
            Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False}, future=True)

        @event.listens_for(self.engine, "connect")
        def _fk_on(dbapi_conn, _):  # type: ignore[no-untyped-def]
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        Base.metadata.create_all(self.engine)
        self._sm = sessionmaker(self.engine, expire_on_commit=False)
        self.message_limit = message_limit
        self.expiry = timedelta(days=expiry_days)

    def _db(self) -> Session:
        return self._sm()

    def create(self) -> SessionState:
        now = datetime.now(UTC)
        state = SessionState(session_id=secrets.token_urlsafe(18), created_at=now, updated_at=now,
                             expires_at=now + self.expiry)  # fmt: skip
        with self._db() as db, db.begin():
            db.add(SessionRow(id=state.session_id, created_at=now, updated_at=now, expires_at=state.expires_at,
                              state_json=state.model_dump_json()))  # fmt: skip
        return state

    def get(self, session_id: str) -> SessionState | None:
        with self._db() as db:
            row = db.get(SessionRow, session_id)
            if row is None:
                return None
            if _utc(row.expires_at) < datetime.now(UTC):
                return None
            return SessionState.model_validate_json(row.state_json)

    def save(self, state: SessionState) -> None:
        now = datetime.now(UTC)
        state.updated_at = now
        state.expires_at = now + self.expiry  # sliding expiry
        with self._db() as db, db.begin():
            row = db.get(SessionRow, state.session_id)
            if row is None:
                db.add(SessionRow(id=state.session_id, created_at=state.created_at, updated_at=now,
                                  expires_at=state.expires_at, state_json=state.model_dump_json()))  # fmt: skip
            else:
                row.updated_at, row.expires_at, row.state_json = now, state.expires_at, state.model_dump_json()

    def delete(self, session_id: str) -> bool:
        with self._db() as db, db.begin():
            db.execute(delete(MessageRow).where(MessageRow.session_id == session_id))
            res = db.execute(delete(SessionRow).where(SessionRow.id == session_id))
            return bool(res.rowcount)

    def append_message(self, session_id: str, role: str, content: str, meta: dict | None = None) -> None:
        with self._db() as db, db.begin():
            db.add(MessageRow(session_id=session_id, role=role, content=content, created_at=datetime.now(UTC),
                              meta_json=json.dumps(meta or {}, default=str)))  # fmt: skip
            db.flush()
            count = db.scalar(select(func.count()).select_from(MessageRow).where(MessageRow.session_id == session_id))
            if count and count > self.message_limit:
                keep = select(MessageRow.id).where(MessageRow.session_id == session_id).order_by(
                    MessageRow.id.desc()).limit(self.message_limit)  # fmt: skip
                db.execute(delete(MessageRow).where(MessageRow.session_id == session_id, MessageRow.id.not_in(keep)))

    def recent_messages(self, session_id: str, limit: int | None = None) -> list[dict]:
        with self._db() as db:
            rows = db.scalars(
                select(MessageRow).where(MessageRow.session_id == session_id).order_by(MessageRow.id.desc())
                .limit(limit or self.message_limit)
            ).all()  # fmt: skip
        return [
            {"role": r.role, "content": r.content, "created_at": _utc(r.created_at).isoformat(),
             "meta": json.loads(r.meta_json or "{}")}
            for r in reversed(rows)
        ]  # fmt: skip

    def purge_expired(self) -> int:
        now = datetime.now(UTC)
        with self._db() as db, db.begin():
            ids = db.scalars(select(SessionRow.id).where(SessionRow.expires_at < now)).all()
            if ids:
                db.execute(delete(MessageRow).where(MessageRow.session_id.in_(ids)))
                db.execute(delete(SessionRow).where(SessionRow.id.in_(ids)))
            return len(ids)
