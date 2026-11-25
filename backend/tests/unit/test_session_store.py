from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.conversation.state import Slot, SlotStatus
from app.memory.session_store import SqliteSessionRepository


def repo(tmp_path, **kw):
    return SqliteSessionRepository(f"sqlite:///{tmp_path / 'sessions.db'}", **kw)


def test_persistence_across_restart(tmp_path):
    r1 = repo(tmp_path)
    s = r1.create()
    s.set_slot(Slot(name="model_year", value=2014, status=SlotStatus.CONFIRMED, confidence=1.0))
    r1.save(s)
    r1.append_message(s.session_id, "user", "hello")
    r1.engine.dispose()
    r2 = repo(tmp_path)  # "backend restart"
    s2 = r2.get(s.session_id)
    assert s2 is not None and s2.value("model_year") == 2014
    assert [m["content"] for m in r2.recent_messages(s.session_id)] == ["hello"]


def test_session_isolation(tmp_path):
    r = repo(tmp_path)
    a, b = r.create(), r.create()
    assert a.session_id != b.session_id
    a.set_slot(Slot(name="engine", value="5.0L", status=SlotStatus.USER_PROVIDED, confidence=0.95))
    r.save(a)
    r.append_message(a.session_id, "user", "only in A")
    assert r.get(b.session_id).value("engine") is None
    assert r.recent_messages(b.session_id) == []


def test_message_history_limit(tmp_path):
    r = repo(tmp_path, message_limit=5)
    s = r.create()
    for i in range(12):
        r.append_message(s.session_id, "user", f"m{i}")
    msgs = r.recent_messages(s.session_id)
    assert [m["content"] for m in msgs] == [f"m{i}" for i in range(7, 12)]


def test_expiry_and_purge(tmp_path):
    r = repo(tmp_path, expiry_days=7)
    s = r.create()
    with r._db() as db, db.begin():
        from app.memory.memory_models import SessionRow

        db.get(SessionRow, s.session_id).expires_at = datetime.now(UTC) - timedelta(days=1)
    assert r.get(s.session_id) is None
    assert r.purge_expired() == 1


def test_delete(tmp_path):
    r = repo(tmp_path)
    s = r.create()
    assert r.delete(s.session_id) is True
    assert r.get(s.session_id) is None and r.delete(s.session_id) is False
