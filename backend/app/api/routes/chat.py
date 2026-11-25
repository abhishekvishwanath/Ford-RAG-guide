from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_container, rate_limit
from app.conversation.state import SessionState, Slot, SlotStatus
from app.graph.workflow import run_turn
from app.schemas.api import ChatRequest, ChatResponse, VehicleSelection
from app.services.container import Container

router = APIRouter(prefix="/api", tags=["chat"])


def apply_vehicle_selection(state: SessionState, sel: VehicleSelection) -> list[str]:
    """UI selections are explicit user choices -> confirmed. A change invalidates the active procedure."""
    changed = []
    for name, value in sel.model_dump(exclude_none=True).items():
        cur = state.slot(name)
        if cur.value != value or cur.status != SlotStatus.CONFIRMED:
            state.set_slot(Slot(name=name, value=value, status=SlotStatus.CONFIRMED, confidence=1.0, turn=state.turn,
                                source_text="vehicle panel"))  # fmt: skip
            changed.append(name)
    if changed:
        state.active_procedure = None
    return changed


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
def chat(req: ChatRequest, c: Container = Depends(get_container)) -> dict:
    state = c.sessions.get(req.session_id) if req.session_id else None
    if state is None:
        state = c.sessions.create()  # unknown/expired ids start a fresh, isolated session
    with c.session_locks[state.session_id]:
        if req.vehicle:
            apply_vehicle_selection(state, req.vehicle)
            c.sessions.save(state)
        return run_turn(c.workflow, state.session_id, req.message)
