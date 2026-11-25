"""LangGraph nodes. Each node takes (state, deps) and returns a partial state update."""

from __future__ import annotations

import logging

from app.conversation import context_extractor as cx
from app.conversation.question_planner import describe_context, plan_question
from app.conversation.readiness import GateSettings, evaluate, load_policy, variant_requirements
from app.conversation.safety import safety_message
from app.conversation.state import ActiveProcedure, PendingQuestion, ReadinessStatus
from app.core.exceptions import LLMUnavailableError, RetrievalBlockedError
from app.core.logging import log_event
from app.graph.state import GraphDeps, GraphState

logger = logging.getLogger(__name__)


def _trace(state: GraphState, name: str) -> list[str]:
    return [*state.get("trace", []), name]


def _guard_ready(state: GraphState, node: str) -> None:
    """Defense in depth: retrieval nodes refuse to run unless the gate said READY."""
    decision = state.get("decision")
    if decision is None or decision.status != ReadinessStatus.READY:
        raise RetrievalBlockedError(f"{node} reached without READY readiness decision")


# ----------------------------------------------------------------------------- intake
def load_session(state: GraphState, deps: GraphDeps) -> GraphState:
    sid = state.get("session_id")
    session = deps.repo.get(sid) if sid else None
    if session is None:
        session = deps.repo.create()
    history = deps.repo.recent_messages(session.session_id, deps.settings.session_message_limit)
    return {"session": session, "history": history, "trace": _trace(state, "load_session")}


def append_user_message(state: GraphState, deps: GraphDeps) -> GraphState:
    session = state["session"]
    session.turn += 1
    deps.repo.append_message(session.session_id, "user", state["message"])
    return {"session": session, "trace": _trace(state, "append_user_message")}


def extract_vehicle_context(state: GraphState, deps: GraphDeps) -> GraphState:
    use_llm = deps.llm if deps.settings.llm_context_extraction_enabled else None
    history = [{"role": m["role"], "content": m["content"][:500]} for m in state.get("history", [])]
    ex = cx.extract(state["message"], state["session"], deps.profile, use_llm, deps.settings.llm_extraction_model,
                    history)  # fmt: skip
    return {"extraction": ex, "trace": _trace(state, "extract_vehicle_context")}


def extract_repair_context(state: GraphState, deps: GraphDeps) -> GraphState:
    ex = state["extraction"]
    comp = ex.updates.get("component")
    if comp and not ex.manual_terms:
        ex.manual_terms = cx.manual_terms_for(str(comp.value))
    return {"extraction": ex, "trace": _trace(state, "extract_repair_context")}


def update_context_slots(state: GraphState, deps: GraphDeps) -> GraphState:
    session = state["session"]
    events = cx.apply_extraction(session, state["extraction"], session.turn, state["message"])
    log_event(logger, "context updated", session=session.session_id[:6], events=events)
    return {"session": session, "events": events, "trace": _trace(state, "update_context_slots")}


def detect_conflicts(state: GraphState, deps: GraphDeps) -> GraphState:
    conflicts = [e.split(":", 1)[1] for e in state.get("events", []) if e.startswith("conflict:")]
    return {"events": [*state.get("events", []), *[f"open_conflict:{c}" for c in conflicts]],
            "trace": _trace(state, "detect_conflicts")}  # fmt: skip


def determine_required_context(state: GraphState, deps: GraphDeps) -> GraphState:
    # computed inside evaluate(); this node records the reasons for observability
    reasons, cands = variant_requirements(state["session"], deps.profile, load_policy())
    log_event(logger, "required context", variant_slots=list(reasons), candidates=len(cands))
    return {"trace": _trace(state, "determine_required_context")}


def evaluate_readiness(state: GraphState, deps: GraphDeps) -> GraphState:
    session = state["session"]
    decision = evaluate(
        session,
        deps.profile,
        GateSettings.from_settings(deps.settings),
        other_vehicle=state["extraction"].other_vehicle,
    )
    session.readiness = decision
    log_event(logger, "readiness", status=decision.status.value, score=decision.score, reason=decision.reason)
    return {"session": session, "decision": decision, "trace": _trace(state, "evaluate_readiness")}


# ----------------------------------------------------------------------------- non-retrieval exits
def ask_followup(state: GraphState, deps: GraphDeps) -> GraphState:
    session = state["session"]
    decision = state["decision"]
    q = plan_question(decision, session, deps.profile, deps.settings.max_intake_turns)
    if q.slots and q.slots[0] in decision.soft_slots and q.slots[0] not in session.soft_asked:
        session.soft_asked.append(q.slots[0])
    session.pending_question = q
    session.intake_turns += 1
    session.workflow_state = "asking_context"
    return {"session": session, "question": q, "assistant_message": q.text, "interaction_mode": "asking_context",
            "trace": _trace(state, "ask_followup")}  # fmt: skip


def request_confirmation(state: GraphState, deps: GraphDeps) -> GraphState:
    session = state["session"]
    q = plan_question(state["decision"], session, deps.profile, deps.settings.max_intake_turns)
    session.pending_question = q
    session.intake_turns += 1
    session.workflow_state = "awaiting_confirmation"
    return {"session": session, "question": q, "assistant_message": q.text,
            "interaction_mode": "awaiting_confirmation", "trace": _trace(state, "request_confirmation")}  # fmt: skip


def handle_safety_escalation(state: GraphState, deps: GraphDeps) -> GraphState:
    session = state["session"]
    session.pending_question = PendingQuestion(slots=[], kind="safety_ack", text="Confirm it is safe to continue.")
    session.workflow_state = "safety_escalation"
    return {"session": session, "assistant_message": safety_message(session.active_hazard or ""),
            "interaction_mode": "safety_escalation", "trace": _trace(state, "handle_safety_escalation")}  # fmt: skip


def handle_out_of_scope(state: GraphState, deps: GraphDeps) -> GraphState:
    session = state["session"]
    years = ", ".join(map(str, deps.profile.model_years)) or "the loaded"
    msg = (
        f"{state['decision'].reason} I can only answer from the supplied {years} "
        f"{deps.profile.model_name or ''} workshop manual, so I won't guess procedures or specifications for a "
        "different vehicle. If you are working on a vehicle the manual covers, tell me its details."
    ).replace("  ", " ")
    session.pending_question = None
    for slot in ("model_year", "engine", "transmission"):
        s = session.slot(slot)
        if s.filled and state["decision"].reason and str(s.value) in state["decision"].reason:
            session.clear_slot(slot)  # allow the user to restate the vehicle
    session.workflow_state = "out_of_scope"
    return {"session": session, "assistant_message": msg, "interaction_mode": "out_of_scope",
            "trace": _trace(state, "handle_out_of_scope")}  # fmt: skip


# ----------------------------------------------------------------------------- retrieval (READY only)
def build_retrieval_query(state: GraphState, deps: GraphDeps) -> GraphState:
    _guard_ready(state, "build_retrieval_query")
    assert deps.retrieval is not None
    session = state["session"]
    session.pending_question = None
    session.intake_turns = 0
    session.workflow_state = "searching_manual"
    req = deps.retrieval.build_request(session, state["message"], state["extraction"])
    return {"session": session, "request": req, "trace": _trace(state, "build_retrieval_query")}


def apply_applicability_filters(state: GraphState, deps: GraphDeps) -> GraphState:
    _guard_ready(state, "apply_applicability_filters")
    return {"filter": deps.retrieval.applicability_filter(state["request"]),  # type: ignore[union-attr]
            "trace": _trace(state, "apply_applicability_filters")}  # fmt: skip


def dense_retrieve(state: GraphState, deps: GraphDeps) -> GraphState:
    _guard_ready(state, "dense_retrieve")
    return {"dense": deps.retrieval.dense(state["request"], state["filter"]),  # type: ignore[union-attr]
            "trace": _trace(state, "dense_retrieve")}  # fmt: skip


def bm25_retrieve(state: GraphState, deps: GraphDeps) -> GraphState:
    _guard_ready(state, "bm25_retrieve")
    return {"bm25": deps.retrieval.bm25(state["request"], state["filter"]),  # type: ignore[union-attr]
            "trace": _trace(state, "bm25_retrieve")}  # fmt: skip


def rrf_fusion(state: GraphState, deps: GraphDeps) -> GraphState:
    _guard_ready(state, "rrf_fusion")
    return {"candidates": deps.retrieval.fuse(state["dense"], state["bm25"]),  # type: ignore[union-attr]
            "trace": _trace(state, "rrf_fusion")}  # fmt: skip


def rerank(state: GraphState, deps: GraphDeps) -> GraphState:
    _guard_ready(state, "rerank")
    return {"candidates": deps.retrieval.rerank(state["request"], state["candidates"]),  # type: ignore[union-attr]
            "trace": _trace(state, "rerank")}  # fmt: skip


def expand_procedure_context(state: GraphState, deps: GraphDeps) -> GraphState:
    _guard_ready(state, "expand_procedure_context")
    return {"package": deps.retrieval.expand(state["request"], state["candidates"]),  # type: ignore[union-attr]
            "trace": _trace(state, "expand_procedure_context")}  # fmt: skip


def retrieve_visuals_and_tables(state: GraphState, deps: GraphDeps) -> GraphState:
    _guard_ready(state, "retrieve_visuals_and_tables")
    return {"package": deps.retrieval.visuals_and_tables(state["package"]),  # type: ignore[union-attr]
            "trace": _trace(state, "retrieve_visuals_and_tables")}  # fmt: skip


def assess_evidence(state: GraphState, deps: GraphDeps) -> GraphState:
    _guard_ready(state, "assess_evidence")
    return {"package": deps.retrieval.assess(state["package"]),  # type: ignore[union-attr]
            "trace": _trace(state, "assess_evidence")}  # fmt: skip


def handle_insufficient_evidence(state: GraphState, deps: GraphDeps) -> GraphState:
    session = state["session"]
    pkg = state["package"]
    if pkg.variant_ambiguity:
        msg = (
            "The manual has more than one variant of this procedure and I can't tell which applies to your vehicle "
            "from the details so far:\n"
            + "\n".join(f"- {t}" for t in pkg.variant_ambiguity[:5])
            + "\nWhich one matches your vehicle?"
        )
        session.pending_question = PendingQuestion(slots=["component"], kind="information", text=msg,
                                                   options=pkg.variant_ambiguity[:5], asked_turn=session.turn)  # fmt: skip
        mode = "asking_context"
    else:
        msg = (
            f"I searched the workshop manual for {describe_context(session)}, but I could not find evidence that "
            f"answers this reliably ({pkg.insufficiency_reason or 'no matching procedure'}). I won't fill the gap with "
            "general knowledge. You could rephrase using the manual's part name, give a DTC, or name the section."
        )
        mode = "answer"
    session.workflow_state = "insufficient_evidence"
    return {"session": session, "assistant_message": msg, "interaction_mode": mode, "citations": [],
            "trace": _trace(state, "handle_insufficient_evidence")}  # fmt: skip


# ----------------------------------------------------------------------------- generation
def build_grounded_prompt(state: GraphState, deps: GraphDeps) -> GraphState:
    _guard_ready(state, "build_grounded_prompt")
    prompt = deps.generator.build_prompt(state["session"], state["message"], state["package"])  # type: ignore[union-attr]
    return {"prompt": prompt, "trace": _trace(state, "build_grounded_prompt")}


def generate_answer(state: GraphState, deps: GraphDeps) -> GraphState:
    _guard_ready(state, "generate_answer")
    try:
        raw = deps.generator.generate(state["prompt"])  # type: ignore[union-attr]
        return {"raw_answer": raw, "error": None, "trace": _trace(state, "generate_answer")}
    except LLMUnavailableError as exc:
        return {"raw_answer": "", "error": str(exc), "trace": _trace(state, "generate_answer:llm_unavailable")}


def validate_answer(state: GraphState, deps: GraphDeps) -> GraphState:
    gen = deps.generator
    assert gen is not None
    if state.get("error") or not state.get("raw_answer"):
        answer = gen.fallback_answer(state["package"])
        answer.validation_warnings.append(f"model unavailable: {state.get('error') or 'empty answer'}")
    else:
        answer = gen.validate(state["raw_answer"], state["package"])
        if not answer.grounded:
            rejected = answer
            answer = gen.fallback_answer(state["package"])
            answer.validation_warnings += [f"model answer rejected: {w}" for w in rejected.validation_warnings]
            answer.unsupported_values = rejected.unsupported_values
            log_event(logger, "answer rejected by validator", logging.WARNING, warnings=rejected.validation_warnings)
    return {"answer": answer, "trace": _trace(state, "validate_answer")}


def attach_citations(state: GraphState, deps: GraphDeps) -> GraphState:
    session = state["session"]
    answer = state["answer"]
    pkg = state["package"]
    citations = deps.generator.citations(answer, pkg)  # type: ignore[union-attr]
    cited = {c.chunk_id for c in citations}
    cited_sources = {c.source_id for c in citations}
    main_parent = pkg.items[0].parent_id if pkg.items else None
    by_source = {i.source_id: i for i in pkg.items}
    # show only diagrams tied to evidence the answer actually cites (fallback: the main procedure's)
    pkg.images = [im for im in pkg.images if im.source_id in cited_sources] or [
        im for im in pkg.images if im.source_id in by_source and by_source[im.source_id].parent_id == main_parent
    ]
    main = next((i for i in pkg.items if i.chunk_id in cited), pkg.items[0] if pkg.items else None)
    if main is not None:
        session.active_procedure = ActiveProcedure(
            parent_id=main.parent_id,
            procedure_id=main.parent_id.removeprefix("proc_"),
            title=main.procedure_title,
            subsection=main.subsection,
            chunk_ids=[i.chunk_id for i in pkg.items],
            figure_ids=[im.figure_id for im in pkg.images],
            turn=session.turn,
        )
    session.recent_evidence = [
        *session.recent_evidence[-4:],
        {"turn": session.turn, "chunk_ids": [i.chunk_id for i in pkg.items], "figure_ids": [im.figure_id for im in pkg.images]},
    ]  # fmt: skip
    session.workflow_state = "answered"
    preface = f"**Vehicle and task:** {describe_context(session)}\n\n"
    return {"session": session, "citations": citations, "assistant_message": preface + answer.text,
            "interaction_mode": "answer", "trace": _trace(state, "attach_citations")}  # fmt: skip


# ----------------------------------------------------------------------------- persistence + response
def update_session_state(state: GraphState, deps: GraphDeps) -> GraphState:
    session = state["session"]
    meta = {"interaction_mode": state.get("interaction_mode"),
            "citations": [c.source_id for c in state.get("citations", [])]}  # fmt: skip
    deps.repo.append_message(session.session_id, "assistant", state.get("assistant_message", ""), meta)
    deps.repo.save(session)
    return {"trace": _trace(state, "update_session_state")}


def _safe_slots(slots: dict) -> dict:
    return {
        k: {"value": v.value, "status": v.status.value, "confidence": v.confidence,
            **({"needs_confirmation_of": v.conflict_with} if v.conflict_with not in (None, []) else {}),
            **({"basis": v.basis} if v.basis else {})}
        for k, v in slots.items()
        if v.filled or v.status.value == "user_unsure"
        if k != "vin"
    }  # fmt: skip


def return_response(state: GraphState, deps: GraphDeps) -> GraphState:
    session = state["session"]
    decision = state.get("decision")
    pkg = state.get("package")
    q = (
        session.pending_question
        if state.get("interaction_mode") in ("asking_context", "awaiting_confirmation")
        else None
    )
    response = {
        "session_id": session.session_id,
        "assistant_message": state.get("assistant_message", ""),
        "interaction_mode": state.get("interaction_mode", "asking_context"),
        "readiness": {
            "status": decision.status.value if decision else None,
            "score": decision.score if decision else 0.0,
            "missing_fields": (decision.missing_hard + decision.missing_soft) if decision else [],
            "needs_confirmation": (decision.needs_confirmation + decision.conflicts) if decision else [],
            "required_fields": decision.required_slots if decision else [],
            "reason": decision.reason if decision else "",
        },
        "pending_question": {"slots": q.slots, "kind": q.kind, "options": q.options} if q else None,
        "vehicle_context": _safe_slots(session.vehicle),
        "repair_context": _safe_slots(session.repair),
        "citations": [c.model_dump() for c in state.get("citations", [])],
        "retrieved_sources": [
            {
                "source_id": i.source_id,
                "chunk_id": i.chunk_id,
                "procedure_title": i.procedure_title,
                "section": i.section,
                "subsection": i.subsection,
                "pages": f"{i.page_start}-{i.page_end}",
                "applicability": i.applicability_note,
                "expanded": i.expanded,
            }
            for i in (pkg.items if pkg and state.get("interaction_mode") == "answer" else [])
        ],  # fmt: skip
        "images": [im.model_dump() for im in (pkg.images if pkg and state.get("interaction_mode") == "answer" else [])],
        "tables": [t.model_dump() for t in (pkg.tables if pkg and state.get("interaction_mode") == "answer" else [])],
        "validation": (
            {"warnings": state["answer"].validation_warnings, "unsupported_values": state["answer"].unsupported_values}
            if state.get("answer")
            else None
        ),  # fmt: skip
        "conversation_state": {
            "workflow_state": session.workflow_state,
            "turn": session.turn,
            "intake_turns": session.intake_turns,
            "active_procedure": session.active_procedure.title if session.active_procedure else None,
            "trace": state.get("trace", []),
        },
    }
    return {"response": response, "trace": _trace(state, "return_response")}
