"""Choose and phrase the single most useful next question."""

from __future__ import annotations

from app.conversation.readiness import load_policy
from app.conversation.state import PendingQuestion, ReadinessDecision, ReadinessStatus, SessionState, SlotStatus
from app.services.manual_profile import ManualProfile

TASK_WORDS = {
    "replace": "removal and installation",
    "removal": "removal",
    "installation": "installation",
    "diagnose": "diagnosis",
    "specification": "specification lookup",
    "inspect": "inspection",
    "adjust": "adjustment",
    "service": "service procedure",
    "overview": "description and operation",
    "repair": "repair",
    "disassembly": "disassembly and assembly",
}


def _engine_options(profile: ManualProfile) -> list[str]:
    labels = {"5.8L": "5.8L (GT500)"}
    return [labels.get(e, e) for e in profile.engines]


def _engine_choice(profile: ManualProfile) -> str:
    opts = _engine_options(profile)
    return ", ".join(opts) if opts else "engine displacement, e.g. 5.0L"


def _why(decision: ReadinessDecision, slot: str) -> str:
    reason = decision.variant_reasons.get(slot)
    return f" The manual has separate {slot.replace('_', ' ')}-specific content for this job." if reason else ""


def describe_context(state: SessionState) -> str:
    v = state.vehicle_summary()
    parts = [str(v["model_year"]) if v.get("model_year") else None, "Mustang"]
    if v.get("variant"):
        parts.append(v["variant"])
    if v.get("engine"):
        parts.append(v["engine"])
    if v.get("transmission"):
        parts.append(f"{v['transmission']} transmission")
    if v.get("body_style"):
        parts.append(v["body_style"])
    vehicle = ", ".join(p for p in parts if p)
    task = TASK_WORDS.get(state.value("task_type") or "", state.value("task_type") or "")
    comp = state.value("component") or (state.active_procedure.title if state.active_procedure else None)
    what = " ".join(p for p in [comp, task] if p) or "your request"
    dtcs = state.value("dtcs")
    if dtcs:
        what += f" (DTC {', '.join(dtcs)})"
    return f"{vehicle}; task: {what}"


def plan_question(
    decision: ReadinessDecision, state: SessionState, profile: ManualProfile, max_intake_turns: int = 8
) -> PendingQuestion:
    policy = load_policy()
    hints = policy.get("unsure_hints", {})
    slot = decision.next_slot or "component"

    if decision.status == ReadinessStatus.NEEDS_CONFIRMATION:
        s = state.slot(slot)
        if s.conflict_with not in (None, [], ""):
            others = s.conflict_with if isinstance(s.conflict_with, list) else [s.conflict_with]
            vals = [s.value, *others]
            text = (
                f"I have two different values for the {slot.replace('_', ' ')}: "
                + " or ".join(str(v) for v in vals)
                + ". Which one is correct?"
            )
            return PendingQuestion(slots=[slot], kind="information", text=text, options=[str(v) for v in vals],
                                   asked_turn=state.turn)  # fmt: skip
        basis = f" ({s.basis})" if s.basis else ""
        text = (
            f"Just to confirm before I search the manual: is the {slot.replace('_', ' ')} {s.value}?{basis} "
            "Please confirm, or tell me the correct value."
        )
        return PendingQuestion(
            slots=[slot], kind="confirmation", text=text, options=["Yes", "No"], asked_turn=state.turn
        )

    unsure = state.slot(slot).status == SlotStatus.USER_UNSURE
    hint = f" {hints[slot]}" if unsure and slot in hints else ""
    missing = set(decision.missing_hard)
    if slot == "model_year":
        q = PendingQuestion(slots=["model_year"], kind="information", text="Which model year is the Mustang?" + hint)
    elif slot == "component":
        q = PendingQuestion(
            slots=["component"],
            kind="information",
            text="Which component or system are you working on, or what symptom are you seeing?",
        )
    elif slot == "task_type":
        comp = state.value("component") or "it"
        q = PendingQuestion(
            slots=["task_type"],
            kind="information",
            text=f"What do you need for the {comp}: removal and installation, diagnosis, or a specification?",
            options=["Removal and installation", "Diagnosis", "Specification"],
        )
    elif slot in ("engine", "transmission") and {"engine", "transmission"} <= missing and not unsure:
        q = PendingQuestion(
            slots=["engine", "transmission"],
            kind="information",
            text=(
                f"Which engine ({_engine_choice(profile)}) and which transmission "
                f"({' or '.join(profile.transmissions or ['automatic', 'manual'])}) does it have?"
                + _why(decision, "engine")
            ),
            options=_engine_options(profile),
        )
    elif slot == "engine":
        q = PendingQuestion(
            slots=["engine"],
            kind="information",
            text=f"Which engine is fitted ({_engine_choice(profile)})?" + _why(decision, "engine") + hint,
            options=_engine_options(profile),
        )
    elif slot == "transmission":
        opts = [t.capitalize() for t in (profile.transmissions or ["automatic", "manual"])]
        q = PendingQuestion(
            slots=["transmission"],
            kind="information",
            text="Is it the manual or the automatic transmission?" + _why(decision, "transmission") + hint,
            options=opts,
        )
    elif slot == "body_style":
        q = PendingQuestion(
            slots=["body_style"],
            kind="information",
            text="Is it a coupe or a convertible?" + _why(decision, "body_style"),
            options=["Coupe", "Convertible"],
        )
    elif slot == "symptom_or_dtc":
        q = PendingQuestion(
            slots=["symptom_or_dtc"],
            kind="information",
            text="What symptom are you seeing, and are any diagnostic trouble codes (DTCs) stored?",
        )
    elif slot == "conditions":
        q = PendingQuestion(
            slots=["conditions"],
            kind="information",
            text="When does it happen (cold start, at speed, under load, intermittently)? Say 'not sure' to skip.",
        )
    else:
        q = PendingQuestion(slots=[slot], kind="information", text=f"Could you tell me the {slot.replace('_', ' ')}?")
    q.asked_turn = state.turn
    if state.intake_turns >= max_intake_turns:
        q.text += (
            " I still need this before I can search the manual safely; without it I could return the wrong "
            "procedure or specification."
        )
    return q
