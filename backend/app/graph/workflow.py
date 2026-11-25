"""Assemble the LangGraph conversational workflow.

intake:     load_session -> append_user_message -> extract_vehicle_context -> extract_repair_context
            -> update_context_slots -> detect_conflicts -> determine_required_context -> evaluate_readiness
gate:       evaluate_readiness --(READY only)--> build_retrieval_query
            otherwise -> ask_followup | request_confirmation | handle_safety_escalation | handle_out_of_scope
retrieval:  build_retrieval_query -> apply_applicability_filters -> dense_retrieve -> bm25_retrieve
            -> rrf_fusion -> rerank -> expand_procedure_context -> retrieve_visuals_and_tables -> assess_evidence
answer:     assess_evidence --(sufficient)--> build_grounded_prompt -> generate_answer -> validate_answer
            -> attach_citations;  insufficient -> handle_insufficient_evidence
exit:       update_session_state -> return_response
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from app.graph import nodes as n
from app.graph.edges import READINESS_ROUTES, route_after_assessment, route_after_readiness
from app.graph.state import GraphDeps, GraphState

INTAKE = [
    "load_session",
    "append_user_message",
    "extract_vehicle_context",
    "extract_repair_context",
    "update_context_slots",
    "detect_conflicts",
    "determine_required_context",
    "evaluate_readiness",
]
RETRIEVAL = [
    "build_retrieval_query",
    "apply_applicability_filters",
    "dense_retrieve",
    "bm25_retrieve",
    "rrf_fusion",
    "rerank",
    "expand_procedure_context",
    "retrieve_visuals_and_tables",
    "assess_evidence",
]
ANSWER = ["build_grounded_prompt", "generate_answer", "validate_answer", "attach_citations"]
EXITS = ["ask_followup", "request_confirmation", "handle_safety_escalation", "handle_out_of_scope",
         "handle_insufficient_evidence"]  # fmt: skip


def build_workflow(deps: GraphDeps):  # type: ignore[no-untyped-def]
    g = StateGraph(GraphState)
    for name in [*INTAKE, *RETRIEVAL, *ANSWER, *EXITS, "update_session_state", "return_response"]:
        g.add_node(name, partial(getattr(n, name), deps=deps))

    g.add_edge(START, INTAKE[0])
    for a, b in zip(INTAKE, INTAKE[1:], strict=False):
        g.add_edge(a, b)
    g.add_conditional_edges("evaluate_readiness", route_after_readiness, list(set(READINESS_ROUTES.values())))
    for a, b in zip(RETRIEVAL, RETRIEVAL[1:], strict=False):
        g.add_edge(a, b)
    g.add_conditional_edges(
        "assess_evidence", route_after_assessment, ["build_grounded_prompt", "handle_insufficient_evidence"]
    )
    for a, b in zip(ANSWER, ANSWER[1:], strict=False):
        g.add_edge(a, b)
    g.add_edge("attach_citations", "update_session_state")
    for e in EXITS:
        g.add_edge(e, "update_session_state")
    g.add_edge("update_session_state", "return_response")
    g.add_edge("return_response", END)
    return g.compile()


def run_turn(workflow, session_id: str | None, message: str) -> dict:  # type: ignore[no-untyped-def]
    final = workflow.invoke({"session_id": session_id, "message": message, "trace": []})
    return final["response"]
