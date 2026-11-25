from __future__ import annotations

from fastapi import APIRouter, Depends, Path

from app.api.dependencies import get_container
from app.api.routes.chat import apply_vehicle_selection
from app.conversation.readiness import GateSettings, evaluate
from app.conversation.state import REPAIR_SLOTS, SessionState
from app.core.exceptions import NotFoundError
from app.graph.nodes import _safe_slots
from app.schemas.api import ContextUpdate, SessionInfo
from app.services.container import Container

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
SID = Path(max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


def _info(c: Container, s: SessionState) -> dict:
    return {
        "session_id": s.session_id,
        "created_at": s.created_at.isoformat(),
        "expires_at": s.expires_at.isoformat() if s.expires_at else None,
        "vehicle_context": _safe_slots(s.vehicle),
        "repair_context": _safe_slots(s.repair),
        "readiness": s.readiness.model_dump() if s.readiness else None,
        "workflow_state": s.workflow_state,
        "messages": [
            {"role": m["role"], "content": m["content"], "created_at": m["created_at"]}
            for m in c.sessions.recent_messages(s.session_id)
        ],  # fmt: skip
    }


def _get(c: Container, sid: str) -> SessionState:
    s = c.sessions.get(sid)
    if s is None:
        raise NotFoundError("Session not found or expired.")
    return s


@router.post("", response_model=SessionInfo, status_code=201)
def create_session(c: Container = Depends(get_container)) -> dict:
    return _info(c, c.sessions.create())


@router.get("/{session_id}", response_model=SessionInfo)
def get_session(session_id: str = SID, c: Container = Depends(get_container)) -> dict:
    return _info(c, _get(c, session_id))


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str = SID, c: Container = Depends(get_container)) -> None:
    if not c.sessions.delete(session_id):
        raise NotFoundError("Session not found.")


@router.post("/{session_id}/context", response_model=SessionInfo)
def update_context(body: ContextUpdate, session_id: str = SID, c: Container = Depends(get_container)) -> dict:
    """Vehicle panel edits. Re-evaluates readiness (no retrieval happens here)."""
    with c.session_locks[session_id]:
        s = _get(c, session_id)
        apply_vehicle_selection(s, body.vehicle)
        if body.reset_task:
            for name in REPAIR_SLOTS:
                s.clear_slot(name)
            s.active_procedure = None
        s.readiness = evaluate(s, c.profile, GateSettings.from_settings(c.settings))
        c.sessions.save(s)
        return _info(c, s)
