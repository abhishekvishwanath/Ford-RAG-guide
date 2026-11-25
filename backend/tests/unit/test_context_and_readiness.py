from __future__ import annotations

import pytest

from app.conversation import context_extractor as cx
from app.conversation.readiness import GateSettings, evaluate
from app.conversation.state import PendingQuestion, ReadinessStatus, SessionState, SlotStatus
from tests.fakes import make_profile

PROFILE = make_profile()
GATE = GateSettings()


def turn(state: SessionState, text: str):
    state.turn += 1
    ex = cx.extract(text, state, PROFILE)
    events = cx.apply_extraction(state, ex, state.turn, text)
    decision = evaluate(state, PROFILE, GATE, other_vehicle=ex.other_vehicle)
    state.readiness = decision
    return ex, events, decision


def new() -> SessionState:
    return SessionState(session_id="s1")


# ------------------------------------------------------------------ extraction
@pytest.mark.parametrize(
    ("text", "slot", "value"),
    [
        ("2014 Mustang", "model_year", 2014),
        ("it's a 5.0L", "engine", "5.0L"),
        ("3.7 liter v6", "engine", "3.7L"),
        ("6R80 automatic", "transmission", "automatic"),
        ("six speed manual, MT82", "transmission", "manual"),
        ("convertible", "body_style", "convertible"),
        ("I need to replace the water pump", "task_type", "replace"),
        ("I need to replace the water pump", "component", "water pump"),
        ("what's the torque for the caliper bolts", "task_type", "specification"),
        ("getting DTC P0128 and b11e8:16", "dtcs", ["P0128", "B11E8:16"]),
    ],
)
def test_rule_extraction(text, slot, value):
    ex = cx.rule_extract(text, new(), PROFILE)
    assert ex.updates[slot].value == value


def test_torque_number_is_not_an_engine():
    assert "engine" not in cx.rule_extract("tighten to 5.0 Nm", new(), PROFILE).updates


def test_synonym_maps_to_manual_term():
    assert cx.manual_terms_for("water pump") == ["coolant pump"]


def test_trim_name_is_only_an_inference():
    ex = cx.rule_extract("2014 Mustang GT", new(), PROFILE)
    up = ex.updates["engine"]
    assert up.value == "5.0L" and up.status == SlotStatus.INFERRED and up.basis


def test_hazard_and_other_vehicle_detection():
    assert cx.rule_extract("there's fuel leaking under the car", new(), PROFILE).hazard == "fuel_leak"
    assert cx.rule_extract("car slipped off the jack stand", new(), PROFILE).hazard == "unstable_vehicle_support"
    assert cx.rule_extract("blue smoke from exhaust on startup", new(), PROFILE).hazard is None
    assert cx.rule_extract("my F-150 needs pads", new(), PROFILE).other_vehicle.lower() == "f-150"


# ------------------------------------------------------------------ readiness
def test_spec_example_conversation_reaches_ready_only_at_the_end():
    s = new()
    _, _, d = turn(s, "I need to replace the water pump.")
    assert d.status == ReadinessStatus.NEEDS_INFORMATION and d.next_slot == "model_year"
    _, _, d = turn(s, "2014")
    assert d.status == ReadinessStatus.NEEDS_INFORMATION and d.next_slot == "engine"
    assert "engine" in d.variant_reasons  # Coolant Pump has 3.7L and 5.0L variants
    _, _, d = turn(s, "5.0L V8")
    assert d.status == ReadinessStatus.READY, d.reason
    assert "transmission" not in d.required_slots  # not discriminating for this component


def test_missing_hard_slot_blocks_even_with_high_score():
    s = new()
    turn(s, "replace the water pump on my 5.0L automatic coupe")  # everything but the year
    d = s.readiness
    assert d.status == ReadinessStatus.NEEDS_INFORMATION and "model_year" in d.missing_hard


def test_generic_component_needs_no_engine():
    s = new()
    _, _, d = turn(s, "2014 mustang, remove the door trim panel")
    assert d.status == ReadinessStatus.READY
    assert "engine" not in d.required_slots


def test_transmission_and_body_requirements_are_dynamic():
    s = new()
    _, _, d = turn(s, "2014 mustang replace the clutch disc and pressure plate")
    assert "transmission" in d.missing_hard
    s = new()
    _, _, d = turn(s, "2014 mustang remove the quarter trim panel")
    assert "body_style" in d.missing_hard


def test_inferred_engine_requires_confirmation_then_ready():
    s = new()
    _, _, d = turn(s, "2014 Mustang GT, replace the coolant pump")
    assert d.status == ReadinessStatus.NEEDS_CONFIRMATION and "engine" in d.needs_confirmation
    s.pending_question = PendingQuestion(slots=["engine"], kind="confirmation", text="?")
    _, events, d = turn(s, "yes")
    assert "confirmed:engine" in events and d.status == ReadinessStatus.READY
    assert s.slot("engine").status == SlotStatus.CONFIRMED


def test_out_of_scope_year_and_vehicle():
    _, _, d = turn(new(), "2018 mustang water pump replacement")
    assert d.status == ReadinessStatus.OUT_OF_SCOPE and "2018" in d.reason
    _, _, d = turn(new(), "2014 F-150 water pump")
    assert d.status == ReadinessStatus.OUT_OF_SCOPE


def test_conflicting_values_need_confirmation():
    _, _, d = turn(new(), "2013 or 2014 mustang door trim panel removal")
    assert d.status == ReadinessStatus.NEEDS_CONFIRMATION and "model_year" in d.conflicts


def test_unconfirmed_change_creates_conflict_but_correction_replaces():
    s = new()
    turn(s, "2014 mustang 5.0L replace coolant pump")
    _, events, d = turn(s, "it has the 3.7L")
    assert "conflict:engine" in events and d.status == ReadinessStatus.NEEDS_CONFIRMATION
    s2 = new()
    turn(s2, "2014 mustang 5.0L replace coolant pump")
    _, events, d = turn(s2, "sorry, actually it's the 3.7L")
    assert "corrected:engine" in events and s2.value("engine") == "3.7L" and d.status == ReadinessStatus.READY


def test_task_change_resets_repair_but_keeps_vehicle():
    s = new()
    turn(s, "2014 mustang 5.0L replace coolant pump")
    _, events, d = turn(s, "now I need to remove the door trim panel")
    assert "task_changed" in events
    assert s.value("model_year") == 2014 and s.value("engine") == "5.0L"
    assert s.value("component") == "door trim panel" and d.status == ReadinessStatus.READY


def test_user_unsure_keeps_slot_missing_and_offers_hint():
    from app.conversation.question_planner import plan_question

    s = new()
    turn(s, "2014 mustang replace the coolant pump")
    s.pending_question = PendingQuestion(slots=["engine"], kind="information", text="?")
    _, events, d = turn(s, "I don't know")
    assert "unsure:engine" in events and d.status == ReadinessStatus.NEEDS_INFORMATION
    q = plan_question(d, s, PROFILE)
    assert "VIN" in q.text and "Section 100-01" in q.text


def test_safety_escalation_precedes_everything_and_ack_resumes():
    s = new()
    _, _, d = turn(s, "2014 mustang 5.0L, fuel is leaking everywhere while replacing the fuel pump")
    assert d.status == ReadinessStatus.SAFETY_ESCALATION
    _, events, d = turn(s, "ok it's safe now, leak contained")
    assert "hazard_acknowledged" in events and d.status != ReadinessStatus.SAFETY_ESCALATION


def test_followup_uses_active_procedure():
    from app.conversation.state import ActiveProcedure

    s = new()
    turn(s, "2014 mustang 5.0L replace coolant pump")
    s.active_procedure = ActiveProcedure(parent_id="proc_CP50", procedure_id="CP50", title="Coolant Pump — 5.0L")
    ex, _, d = turn(s, "what torque for those bolts?")
    assert ex.is_followup and d.status == ReadinessStatus.READY


def test_variant_only_answer_is_not_a_component():
    s = new()
    turn(s, "I need to replace the water pump.")
    turn(s, "2014")
    s.pending_question = PendingQuestion(slots=["engine", "transmission"], kind="information", text="?")
    _, events, d = turn(s, "5.0L")
    assert "task_changed" not in events and s.value("component") == "water pump"
    assert d.status == ReadinessStatus.READY


def test_llm_echo_of_known_fact_does_not_downgrade_provenance():
    s = new()
    turn(s, "2014 mustang replace the coolant pump")
    ex = cx.rule_extract("5.0L V8", s, PROFILE)
    cx.merge_llm(ex, {"model_year": 2014, "engine_displacement": "5.0L"}, "5.0L V8", s.vehicle_summary())
    cx.apply_extraction(s, ex, 2, "5.0L V8")
    assert s.slot("model_year").status == SlotStatus.USER_PROVIDED
    assert evaluate(s, PROFILE, GATE).status == ReadinessStatus.READY
