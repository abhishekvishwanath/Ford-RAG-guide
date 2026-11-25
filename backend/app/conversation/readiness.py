"""Explainable context-readiness gate.

Order of evaluation:
1. Safety escalation (unacknowledged immediate hazard).
2. Out of scope (another vehicle, or a model year the manual does not cover).
3. Required slots: always-hard + task-policy + *dynamic variant slots* (engine /
   transmission / body) that become hard when the manual's candidate procedures for the
   component differ by that variant.
4. Missing hard slots -> NEEDS_INFORMATION (a high score never overrides this).
5. Conflicts, inferred or low-confidence required slots -> NEEDS_CONFIRMATION.
6. Weighted score over hard+soft slots must reach the threshold -> READY.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.conversation.context_extractor import manual_terms_for
from app.conversation.state import ReadinessDecision, ReadinessStatus, SessionState, SlotStatus
from app.core.config import PROJECT_ROOT, Settings
from app.services.manual_profile import CatalogueMatch, ManualProfile

POLICY_PATH = PROJECT_ROOT / "config/readiness_policy.json"
VARIANT_DIMS = {"engine": "engines", "transmission": "transmissions", "body_style": "body_styles"}


@lru_cache
def load_policy(path: Path = POLICY_PATH) -> dict:
    return json.loads(path.read_text())


@dataclass
class GateSettings:
    threshold: float = 0.85
    min_confidence: float = 0.80
    max_intake_turns: int = 8
    confirm_inferred: bool = True

    @classmethod
    def from_settings(cls, s: Settings) -> GateSettings:
        return cls(
            s.readiness_threshold,
            s.min_required_slot_confidence,
            s.max_intake_turns,
            s.require_confirmation_for_inferred_vehicle_fields,
        )


def candidate_procedures(state: SessionState, profile: ManualProfile) -> list[CatalogueMatch]:
    comp = state.value("component")
    if state.active_procedure and not comp:
        comp = state.active_procedure.title
    if not comp:
        return []
    task = state.value("task_type")
    queries = [comp, *manual_terms_for(comp)]
    seen: dict[str, CatalogueMatch] = {}
    for q in queries:
        for m in profile.match_component(q, task):
            if m.parent_id not in seen or m.score > seen[m.parent_id].score:
                seen[m.parent_id] = m
    matches = sorted(seen.values(), key=lambda m: -m.score)
    if matches:
        top = matches[0].score
        matches = [m for m in matches if m.score >= top - 0.25]
    return matches[:12]


def variant_requirements(
    state: SessionState, profile: ManualProfile, policy: dict
) -> tuple[dict[str, str], list[CatalogueMatch]]:
    """Which variant fields discriminate between the manual's procedures for this component."""
    reasons: dict[str, str] = {}
    cands = candidate_procedures(state, profile)
    if cands:
        for slot, attr in VARIANT_DIMS.items():
            specific = [c for c in cands if getattr(c, attr)]
            if specific:
                values = sorted({v for c in specific for v in getattr(c, attr)})
                titles = "; ".join(dict.fromkeys(c.title for c in specific[:3]))
                reasons[slot] = (
                    f"the manual has {slot.replace('_', ' ')}-specific content ({', '.join(values)}): {titles}"
                )
        return reasons, cands
    text = " ".join(filter(None, [state.value("component"), state.value("system"), state.value("symptoms")])).lower()
    for slot, keywords in policy.get("fallback_variant_requirements", {}).items():
        if slot.startswith("_"):
            continue
        if any(k in text for k in keywords):
            reasons[slot] = f"{slot} usually changes the procedure for this system in this manual"
    return reasons, cands


def _satisfied(state: SessionState, slot: str) -> bool:
    if slot == "component":
        if state.value("component") or state.active_procedure is not None:
            return True
        return state.value("task_type") == "diagnose" and bool(state.value("symptoms") or state.value("dtcs"))
    if slot == "symptom_or_dtc":
        return bool(state.value("symptoms") or state.value("dtcs") or state.value("component"))
    s = state.slot(slot)
    return s.filled and s.status != SlotStatus.USER_UNSURE


def evaluate(
    state: SessionState,
    profile: ManualProfile,
    gate: GateSettings,
    *,
    other_vehicle: str | None = None,
    policy: dict | None = None,
) -> ReadinessDecision:
    policy = policy or load_policy()
    weights: dict[str, float] = policy["weights"]

    # 1. safety
    if state.active_hazard and not state.safety_acknowledged:
        return ReadinessDecision(
            status=ReadinessStatus.SAFETY_ESCALATION,
            reason=f"Immediate hazard reported ({state.active_hazard.replace('_', ' ')}).",
        )

    # 2. scope
    if other_vehicle:
        return ReadinessDecision(
            status=ReadinessStatus.OUT_OF_SCOPE,
            reason=f"The loaded manual covers the {profile.model_name or 'supplied'} only, not '{other_vehicle}'.",
        )
    year = state.value("model_year")
    year_conflict = state.slot("model_year").conflict_with not in (None, [], "")
    if year is not None and not year_conflict and profile.available and not profile.covers_year(int(year)):
        years = ", ".join(map(str, profile.model_years))
        return ReadinessDecision(
            status=ReadinessStatus.OUT_OF_SCOPE,
            reason=f"The loaded workshop manual covers model year {years} only; {year} is not covered.",
        )
    for slot, attr in (("engine", profile.engines), ("transmission", profile.transmissions)):
        val = state.value(slot)
        slot_state = state.slot(slot)
        if (
            val
            and attr
            and val not in attr
            and slot_state.status != SlotStatus.INFERRED
            and slot_state.conflict_with in (None, [])
        ):
            return ReadinessDecision(
                status=ReadinessStatus.OUT_OF_SCOPE,
                reason=f"The manual covers {slot}s {', '.join(attr)}; '{val}' is not covered.",
            )

    # 3. required slots
    task = state.value("task_type")
    task_policy = policy["task_policies"].get(task or "", {"hard": ["task_type"], "soft": []})
    hard = list(dict.fromkeys([*policy["always_hard"], *task_policy["hard"]]))
    soft = list(task_policy.get("soft", []))
    variant_reasons, cands = variant_requirements(state, profile, policy)
    for slot in variant_reasons:
        if slot not in hard:
            hard.append(slot)
    if state.active_procedure is not None and not state.value("component"):
        hard = [h for h in hard if h != "task_type"] or hard  # follow-up within an answered procedure

    missing_hard = [s for s in hard if not _satisfied(state, s)]
    missing_soft = [s for s in soft if not _satisfied(state, s) and s not in state.soft_asked]
    conflicts = [s for s in hard + soft if state.slot(s).conflict_with not in (None, [], "")]
    needs_conf, low_conf = [], []
    for s in hard:
        slot = state.slot(s)
        if not slot.filled or s in ("component", "symptom_or_dtc"):
            continue
        if slot.status == SlotStatus.INFERRED and gate.confirm_inferred:
            needs_conf.append(s)
        elif slot.status != SlotStatus.CONFIRMED and slot.confidence < gate.min_confidence and s not in conflicts:
            low_conf.append(s)

    total = sum(weights.get(s, 0.05) for s in hard + soft) or 1.0
    got = sum(
        weights.get(s, 0.05)
        for s in hard + soft
        if (_satisfied(state, s) or s in state.soft_asked) and s not in conflicts and s not in needs_conf
    )
    score = round(got / total, 3)
    decision = ReadinessDecision(
        status=ReadinessStatus.READY,
        score=score,
        required_slots=hard,
        soft_slots=soft,
        missing_hard=missing_hard,
        missing_soft=missing_soft,
        needs_confirmation=needs_conf + low_conf,
        conflicts=conflicts,
        low_confidence=low_conf,
        candidate_procedures=[f"{c.title} ({c.section})" for c in cands[:6]],
        variant_reasons=variant_reasons,
    )
    order = policy["slot_priority"]

    def first(slots: list[str]) -> str | None:
        return sorted(slots, key=lambda s: order.index(s) if s in order else 99)[0] if slots else None

    if conflicts:  # contradictions in what the user already said are resolved first
        decision.status = ReadinessStatus.NEEDS_CONFIRMATION
        decision.next_slot = first(conflicts)
        decision.reason = "Conflicting values: " + ", ".join(conflicts)
    elif missing_hard:
        decision.status = ReadinessStatus.NEEDS_INFORMATION
        decision.next_slot = first(missing_hard)
        decision.reason = "Missing required: " + ", ".join(missing_hard)
    elif needs_conf or low_conf:
        decision.status = ReadinessStatus.NEEDS_CONFIRMATION
        decision.next_slot = first(conflicts or needs_conf or low_conf)
        decision.reason = "Needs confirmation: " + ", ".join(conflicts + needs_conf + low_conf)
    elif missing_soft and len(state.soft_asked) < policy.get("max_soft_questions", 1) and score < gate.threshold:
        decision.status = ReadinessStatus.NEEDS_INFORMATION
        decision.next_slot = first(missing_soft)
        decision.reason = "Helpful detail missing: " + ", ".join(missing_soft)
    elif score < gate.threshold and missing_soft:
        decision.reason = "Optional details not provided; proceeding with required context complete."
    else:
        decision.reason = "All required context present and confirmed."
    if decision.status == ReadinessStatus.READY and score < gate.threshold and not missing_soft:
        decision.status = ReadinessStatus.NEEDS_INFORMATION
        decision.reason = f"Readiness score {score} below threshold {gate.threshold}."
    return decision
