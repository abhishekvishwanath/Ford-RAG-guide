"""Conversation state: vehicle/repair slots with provenance, readiness decision, session state."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

VEHICLE_SLOTS = ("model", "model_year", "engine", "transmission", "body_style", "variant", "market", "vin")
REPAIR_SLOTS = ("task_type", "component", "system", "symptoms", "dtcs", "conditions", "previous_checks", "measurements")
VARIANT_SLOTS = ("engine", "transmission", "body_style")


def now() -> datetime:
    return datetime.now(UTC)


class SlotStatus(StrEnum):
    USER_PROVIDED = "user_provided"
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    USER_UNSURE = "user_unsure"  # user said they don't know
    UNKNOWN = "unknown"


class Slot(BaseModel):
    name: str
    value: Any = None
    status: SlotStatus = SlotStatus.UNKNOWN
    confidence: float = 0.0
    turn: int = 0
    updated_at: datetime = Field(default_factory=now)
    source_text: str | None = None
    basis: str | None = None  # why an inference was made
    conflict_with: Any = None  # a competing value awaiting confirmation
    required: bool = False

    @property
    def filled(self) -> bool:
        return self.value not in (None, "", []) and self.status != SlotStatus.UNKNOWN


class ReadinessStatus(StrEnum):
    READY = "READY"
    NEEDS_INFORMATION = "NEEDS_INFORMATION"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    SAFETY_ESCALATION = "SAFETY_ESCALATION"


class ReadinessDecision(BaseModel):
    status: ReadinessStatus
    score: float = 0.0
    required_slots: list[str] = Field(default_factory=list)
    soft_slots: list[str] = Field(default_factory=list)
    missing_hard: list[str] = Field(default_factory=list)
    missing_soft: list[str] = Field(default_factory=list)
    needs_confirmation: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    low_confidence: list[str] = Field(default_factory=list)
    candidate_procedures: list[str] = Field(default_factory=list)
    variant_reasons: dict[str, str] = Field(default_factory=dict)
    reason: str = ""
    next_slot: str | None = None


class PendingQuestion(BaseModel):
    slots: list[str]
    kind: str  # information | confirmation | safety_ack
    text: str
    options: list[str] = Field(default_factory=list)
    option_ids: list[str] = Field(default_factory=list)  # procedure parent ids for a variant choice
    asked_turn: int = 0


class ActiveProcedure(BaseModel):
    """What the last grounded answer was about — used to resolve 'that bolt', 'step 4'."""

    parent_id: str
    procedure_id: str
    title: str | None = None
    subsection: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)
    vocab: list[str] = Field(default_factory=list)  # words in the procedure text, to recognise follow-ups
    turn: int = 0


class SessionState(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    expires_at: datetime | None = None
    turn: int = 0
    intake_turns: int = 0
    vehicle: dict[str, Slot] = Field(default_factory=dict)
    repair: dict[str, Slot] = Field(default_factory=dict)
    readiness: ReadinessDecision | None = None
    pending_question: PendingQuestion | None = None
    soft_asked: list[str] = Field(default_factory=list)
    safety_acknowledged: bool = False
    active_hazard: str | None = None
    active_procedure: ActiveProcedure | None = None
    chosen_variant: ActiveProcedure | None = None  # procedure the user picked when the manual had variants
    recent_evidence: list[dict] = Field(default_factory=list)  # [{turn, chunk_ids, figure_ids}]
    workflow_state: str = "intake"

    # ------------------------------------------------------------- helpers
    def slot(self, name: str) -> Slot:
        group = self.vehicle if name in VEHICLE_SLOTS else self.repair
        return group.get(name) or Slot(name=name)

    def value(self, name: str) -> Any:
        s = self.slot(name)
        return s.value if s.filled else None

    def set_slot(self, slot: Slot) -> None:
        group = self.vehicle if slot.name in VEHICLE_SLOTS else self.repair
        group[slot.name] = slot

    def clear_slot(self, name: str) -> None:
        (self.vehicle if name in VEHICLE_SLOTS else self.repair).pop(name, None)

    def vehicle_summary(self) -> dict[str, Any]:
        return {k: v.value for k, v in self.vehicle.items() if v.filled and k != "vin"}

    def repair_summary(self) -> dict[str, Any]:
        return {k: v.value for k, v in self.repair.items() if v.filled}
