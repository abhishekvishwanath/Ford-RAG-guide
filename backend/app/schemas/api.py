"""Public API request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

InteractionMode = Literal["asking_context", "awaiting_confirmation", "answer", "safety_escalation", "out_of_scope"]


class VehicleSelection(BaseModel):
    """Explicit selections from the UI vehicle panel (treated as confirmed)."""

    model_year: int | None = Field(default=None, ge=1964, le=2100)
    engine: str | None = Field(default=None, pattern=r"^\d\.\dL$")
    transmission: Literal["automatic", "manual"] | None = None
    body_style: Literal["coupe", "convertible"] | None = None


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=2000)
    vehicle: VehicleSelection | None = None

    @field_validator("message")
    @classmethod
    def strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be blank")
        return v


class Readiness(BaseModel):
    status: str | None
    score: float
    missing_fields: list[str]
    needs_confirmation: list[Any]
    required_fields: list[str]
    reason: str


class ChatResponse(BaseModel):
    session_id: str
    assistant_message: str
    interaction_mode: InteractionMode
    readiness: Readiness
    pending_question: dict | None = None
    vehicle_context: dict
    repair_context: dict
    citations: list[dict]
    retrieved_sources: list[dict]
    images: list[dict]
    tables: list[dict]
    validation: dict | None = None
    conversation_state: dict


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    expires_at: str | None
    vehicle_context: dict
    repair_context: dict
    readiness: dict | None
    workflow_state: str
    messages: list[dict]


class ContextUpdate(BaseModel):
    vehicle: VehicleSelection
    reset_task: bool = False


class IngestRequest(BaseModel):
    stages: list[Literal["parse", "structure", "figures", "chunk", "index"]] | None = None
    page_range: tuple[int, int] | None = None
