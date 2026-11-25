"""Conditional edges. The readiness edge is the only way into the retrieval sub-graph."""

from __future__ import annotations

from app.conversation.state import ReadinessStatus
from app.graph.state import GraphState

READINESS_ROUTES = {
    ReadinessStatus.READY: "build_retrieval_query",
    ReadinessStatus.NEEDS_INFORMATION: "ask_followup",
    ReadinessStatus.NEEDS_CONFIRMATION: "request_confirmation",
    ReadinessStatus.SAFETY_ESCALATION: "handle_safety_escalation",
    ReadinessStatus.OUT_OF_SCOPE: "handle_out_of_scope",
}


def route_after_readiness(state: GraphState) -> str:
    decision = state.get("decision")
    if decision is None:
        return "ask_followup"
    return READINESS_ROUTES.get(decision.status, "ask_followup")


def route_after_assessment(state: GraphState) -> str:
    pkg = state.get("package")
    if pkg is None or not pkg.sufficient:
        return "handle_insufficient_evidence"
    return "build_grounded_prompt"
