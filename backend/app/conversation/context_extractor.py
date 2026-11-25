"""Extract vehicle/repair context from a user turn and apply it to session state.

Layering:
1. Deterministic rules capture explicit facts (year, displacement, transmission, DTCs,
   task verbs, hazards, yes/no/unsure answers) with high confidence.
2. Component wording is matched against the manual's procedure catalogue + synonyms.
3. An optional LLM pass fills free-text fields (component, symptoms, conditions) and
   interprets short answers. LLM vehicle facts are accepted only if the user's own words
   support them; otherwise they are stored as *inferred* and must be confirmed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.conversation.state import (
    REPAIR_SLOTS,
    ActiveProcedure,
    SessionState,
    Slot,
    SlotStatus,
)
from app.core.config import PROJECT_ROOT
from app.generation.llm_client import LLMClient, parse_json
from app.ingestion.metadata import load_vocabulary
from app.services.manual_profile import ManualProfile

logger = logging.getLogger(__name__)

SYNONYMS_PATH = PROJECT_ROOT / "config/synonyms.json"

YEAR_RE = re.compile(r"(?<![\w.-])(19[6-9]\d|20[0-4]\d)(?![\w.-])")
SHORT_YEAR_RE = re.compile(r"(?:'|’)(\d{2})\b")
DISPLACEMENT_RE = re.compile(r"(?<![\w.])([1-9])\.(\d)\s*(?:l\b|liter|litre)?", re.I)
DTC_RE = re.compile(r"\b([BCPU][0-9A-F]{4})(?::([0-9A-F]{2}))?\b", re.I)
AUTO_RE = re.compile(r"\b(automatic|auto|6r80|slushbox|paddle shift\w*)\b", re.I)
MANUAL_RE = re.compile(
    r"\b(manual(?! transmission fluid)|stick(?:\s?shift)?|mt-?82|tr-?6060|6[- ]?speed manual|clutch pedal)\b", re.I
)
BODY_RE = re.compile(r"\b(coupe|convertible)\b", re.I)
YES_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|correct|right|that'?s right|confirm(ed)?|affirmative|sure|ok(ay)?|exactly)\b", re.I
)
NO_RE = re.compile(r"^\s*(no|nope|nah|incorrect|wrong|not quite)\b", re.I)
UNSURE_RE = re.compile(r"\b(don'?t know|do not know|not sure|no idea|unsure|dunno|can'?t tell|idk)\b", re.I)
CORRECTION_RE = re.compile(r"\b(actually|sorry|i meant|correction|my mistake|wait|it'?s really)\b", re.I)
FOLLOWUP_RE = re.compile(
    r"\b(that|those|these|this|it|its|them|step\s*\d+|the (first|second|third|fourth|last|next|previous) "
    r"(one|step|bolt|bolts|nut|nuts|connector|part)|same)\b",
    re.I,
)
OTHER_VEHICLE_RE = re.compile(
    r"\b(f-?150|f-?250|focus|fiesta|explorer|escape|ranger|bronco|edge|fusion|taurus|expedition|transit|"
    r"mach-?e|maverick|camaro|corvette|challenger|charger|civic|accord|corolla|camry|silverado|"
    r"chevy|chevrolet|dodge|toyota|honda|nissan|bmw|audi|mercedes|subaru|hyundai|kia|jeep)\b",
    re.I,
)

TASK_RULES: list[tuple[str, re.Pattern]] = [
    ("replace", re.compile(r"\b(replac\w*|swap\w*|chang(e|ing)|r\s?&\s?r|r and r|renew\w*)\b", re.I)),
    ("removal", re.compile(r"\b(remov\w*|take (it )?off|pull(ing)? (out|off))\b", re.I)),
    ("installation", re.compile(r"\b(install\w*|refit\w*|put (it )?back)\b", re.I)),
    ("adjust", re.compile(r"\b(adjust\w*|align\w*|aim\w*|calibrat\w*)\b", re.I)),
    (
        "diagnose",
        re.compile(
            r"\b(diagnos\w*|troubleshoot\w*|won'?t (start|crank)|no start|nois\w*|leak\w*|misfir\w*|check engine|"
            r"warning light|fault\w*|problem|issue|dtc|codes?|vibrat\w*|overheat\w*|stall\w*|grind\w*|squeal\w*|"
            r"clunk\w*|rattl\w*|rough idle|not working|doesn'?t work|inoperative|intermittent)\b",
            re.I,
        ),
    ),
    (
        "specification",
        re.compile(
            r"\b(torque|specs?|specification\w*|capacit\w*|how much (oil|coolant|fluid)|clearance|tighten\w* to|"
            r"what size|fluid type|which (oil|fluid))\b",
            re.I,
        ),
    ),
    ("inspect", re.compile(r"\b(inspect\w*|check(ing)?|measur\w*|test(ing)?)\b", re.I)),
    ("service", re.compile(r"\b(flush\w*|drain\w*|refill\w*|fill(ing)?|bleed\w*|service)\b", re.I)),
    (
        "overview",
        re.compile(r"\b(how does .{1,40} work|what does .{1,40} do|explain|description|how .{1,30} works)\b", re.I),
    ),
]

HAZARD_RULES: list[tuple[str, re.Pattern]] = [
    (
        "fire_or_smoke",
        re.compile(
            r"\b(on fire|fire|flames?|burning (smell|wires?|insulation)|smok\w* (from|out of|coming)\s+(the )?"
            r"(engine|hood|dash|wiring|battery|under))\b",
            re.I,
        ),
    ),
    (
        "fuel_leak",
        re.compile(
            r"\b((fuel|gas|gasoline|petrol) (is )?(leak\w*|dripping|pouring|spill\w*)|smell\w* (of )?(fuel|gas(oline)?)|"
            r"leaking (fuel|gas))\b",
            re.I,
        ),
    ),
    (
        "brake_failure",
        re.compile(
            r"\b(no brakes?|brakes? (failed|failure|gone|not working|went out)|pedal (goes|went|sinks) to the floor|"
            r"can'?t stop)\b",
            re.I,
        ),
    ),
    (
        "unstable_vehicle_support",
        re.compile(
            r"\b((fell|falling|slipp\w*|unstable|wobbl\w*|tipp\w*|shifting) .{0,25}\b(jack|lift|hoist|stands?)|"
            r"(jack|lift|hoist|stands?)\b.{0,25}\b(fell|falling|slipp\w*|unstable|wobbl\w*|tipp\w*))\b",
            re.I,
        ),
    ),
    (
        "electrical_hazard",
        re.compile(
            r"\b(electric(al)? shock|got shocked|sparking|arcing|battery (is )?(swollen|bulging|hissing|exploded|smoking))\b",
            re.I,
        ),
    ),
    ("injury", re.compile(r"\b(injur\w*|bleeding|burn(ed|t) (my|his|her)|someone (is )?hurt)\b", re.I)),
]
SAFE_ACK_RE = re.compile(
    r"\b(it'?s safe|safe now|is (now )?(safe|fixed|contained|stopped|out|off)|been (fixed|contained|stopped|cleaned)|"
    r"no (longer|more) (leaking|smoking|burning)|understood|acknowledg\w*|proceed|continue|all clear|cleaned (it )?up)\b",
    re.I,
)


STATUS_STRENGTH = {
    SlotStatus.CONFIRMED: 4,
    SlotStatus.USER_PROVIDED: 3,
    SlotStatus.INFERRED: 2,
    SlotStatus.USER_UNSURE: 1,
    SlotStatus.UNKNOWN: 0,
}


@dataclass
class SlotUpdate:
    value: Any
    status: SlotStatus = SlotStatus.USER_PROVIDED
    confidence: float = 0.95
    basis: str | None = None
    source: str = "rule"


@dataclass
class Extraction:
    updates: dict[str, SlotUpdate] = field(default_factory=dict)
    conflicts: dict[str, list[Any]] = field(default_factory=dict)  # same-message contradictions
    confirmation: bool | None = None
    unsure: bool = False
    hazard: str | None = None
    safety_ack: bool = False
    other_vehicle: str | None = None
    is_followup: bool = False
    correction: bool = False
    manual_terms: list[str] = field(default_factory=list)
    llm_used: bool = False


@lru_cache
def load_synonyms() -> dict[str, list[str]]:
    if not SYNONYMS_PATH.exists():
        return {}
    return json.loads(SYNONYMS_PATH.read_text()).get("components", {})


def manual_terms_for(component: str) -> list[str]:
    """Manual terminology for a user component phrase (synonym expansion)."""
    comp = component.lower()
    out: list[str] = []
    for key, vals in sorted(load_synonyms().items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", comp):
            out.extend(v for v in vals if v not in out)
    return out


# --------------------------------------------------------------------------- rules
def _is_followup(text: str, state: SessionState, ex: Extraction) -> bool:
    """A question about the procedure just answered (pronoun, step reference, or a spec/step question
    whose wording overlaps the active procedure title rather than naming a new catalogue component)."""
    ap = state.active_procedure
    if ap is None:
        return False
    if FOLLOWUP_RE.search(text) and "component" not in ex.updates:
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    title_words = (
        set(re.findall(r"[a-z]+", (ap.title or "").lower()))
        | set(re.findall(r"[a-z]+", " ".join(manual_terms_for(str(state.value("component") or "")))))
        | set(re.findall(r"[a-z]+", str(state.value("component") or "").lower()))
    )
    title_words -= {"and", "the", "of", "with", "l"}
    asks_detail = re.search(
        r"\b(torque|tighten|spec|step|bolt|nut|connector|which|what|where|how many|clearance)\b", text, re.I
    )
    new_comp = ex.updates.get("component")
    same_topic = bool(words & title_words) and (
        new_comp is None or bool(set(str(new_comp.value).split()) & title_words)
    )
    return bool(asks_detail and same_topic)


def _years(text: str, expecting_year: bool) -> list[int]:
    years = [int(y) for y in YEAR_RE.findall(text)]
    if expecting_year:
        years += [2000 + int(y) for y in SHORT_YEAR_RE.findall(text) if int(y) < 50]
        m = re.fullmatch(r"\s*(\d{2})\s*", text)
        if m:
            years.append(2000 + int(m.group(1)) if int(m.group(1)) < 50 else 1900 + int(m.group(1)))
    return list(dict.fromkeys(years))


def _engines(text: str, covered: list[str]) -> list[str]:
    """Explicit displacements ("5.0L", "5.0 liter"), or a bare covered value ("the 5.0") not followed by a unit."""
    found = []
    for m in DISPLACEMENT_RE.finditer(text):
        disp = f"{m.group(1)}.{m.group(2)}L"
        explicit_l = m.group(0).lower().rstrip().endswith(("l", "liter", "litre"))
        unit_follows = re.match(
            r"\s*(nm|mm|in\b|v\b|volt|psi|kpa|bar|lb|ohm|qt|quart|%)", text[m.end() : m.end() + 8], re.I
        )
        if explicit_l or (disp in covered and not unit_follows):
            found.append(disp)
    return list(dict.fromkeys(found))


def _model_inferences(text: str) -> list[tuple[str, str, str]]:
    """(engine, variant/term, basis) inferred from trim names. Always need confirmation."""
    raw = load_vocabulary().raw.get("user_aliases", {}).get("model_inferences", {})
    out = []
    low = text.lower()
    for term in sorted(raw, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low):
            out.append((raw[term]["engine"], term, raw[term]["basis"]))
            low = re.sub(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", " ", low)
    return out


def detect_task(text: str) -> str | None:
    hits = [name for name, rx in TASK_RULES if rx.search(text)]
    if "removal" in hits and "installation" in hits:
        return "replace"
    return hits[0] if hits else None


def detect_hazard(text: str) -> str | None:
    for name, rx in HAZARD_RULES:
        if rx.search(text):
            return name
    return None


def detect_component(text: str, profile: ManualProfile, task: str | None) -> tuple[str | None, list[str]]:
    """Find the component phrase: synonym keys first, then catalogue titles."""
    low = text.lower()
    for key in sorted(load_synonyms(), key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", low):
            return key, manual_terms_for(key)
    matches = profile.match_component(text, task)
    if matches and matches[0].score >= 0.9:
        title = re.sub(r"\s*[—–-]\s*(\d\.\dL.*|GT500|Coupe|Convertible).*$", "", matches[0].title).strip()
        return title.lower(), []
    return None, []


def rule_extract(text: str, state: SessionState, profile: ManualProfile) -> Extraction:
    ex = Extraction()
    pending = state.pending_question
    pending_slots = pending.slots if pending else []

    if (h := detect_hazard(text)) is not None:
        ex.hazard = h
    if state.active_hazard and SAFE_ACK_RE.search(text):
        ex.safety_ack = True
    if m := OTHER_VEHICLE_RE.search(text):
        ex.other_vehicle = m.group(1)
    ex.correction = bool(CORRECTION_RE.search(text))
    if pending and pending.kind == "confirmation":
        if YES_RE.search(text):
            ex.confirmation = True
        elif NO_RE.search(text):
            ex.confirmation = False
    if UNSURE_RE.search(text):
        ex.unsure = True

    years = _years(text, "model_year" in pending_slots or bool(re.search(r"mustang|model year|\byear\b", text, re.I)))
    if len(years) == 1:
        ex.updates["model_year"] = SlotUpdate(years[0])
    elif len(years) > 1:
        ex.conflicts["model_year"] = years

    engines = _engines(text, profile.engines)
    if len(engines) == 1:
        ex.updates["engine"] = SlotUpdate(engines[0])
    elif len(engines) > 1:
        ex.conflicts["engine"] = engines
    inferred = _model_inferences(text)
    for eng, term, basis in inferred:
        if term in ("gt500", "shelby"):
            ex.updates["variant"] = SlotUpdate("GT500")
        if "engine" not in ex.updates and "engine" not in ex.conflicts:
            ex.updates["engine"] = SlotUpdate(eng, SlotStatus.INFERRED, 0.7, basis=f"'{term}': {basis}")

    auto, manual = bool(AUTO_RE.search(text)), bool(MANUAL_RE.search(text))
    if auto and not manual:
        ex.updates["transmission"] = SlotUpdate("automatic")
    elif manual and not auto:
        ex.updates["transmission"] = SlotUpdate("manual")
    elif auto and manual:
        ex.conflicts["transmission"] = ["automatic", "manual"]

    bodies = {b.lower() for b in BODY_RE.findall(text)}
    if len(bodies) == 1:
        ex.updates["body_style"] = SlotUpdate(bodies.pop())

    dtcs = [f"{a.upper()}{':' + b.upper() if b else ''}" for a, b in DTC_RE.findall(text)]
    if dtcs:
        ex.updates["dtcs"] = SlotUpdate(dtcs)

    task = detect_task(text)
    if task:
        ex.updates["task_type"] = SlotUpdate(task, confidence=0.85)
    component, terms = detect_component(text, profile, task or state.value("task_type"))
    if component:
        ex.updates["component"] = SlotUpdate(component, confidence=0.9)
        ex.manual_terms = terms
    if (task or state.value("task_type")) == "diagnose" and len(text.split()) >= 3 and not ex.hazard:
        ex.updates.setdefault("symptoms", SlotUpdate(text.strip()[:300], confidence=0.8))

    ex.is_followup = _is_followup(text, state, ex)
    # short answers to a pending single-slot question ("auto", "the 5.0")
    if pending and pending.kind == "information" and len(pending_slots) == 1 and pending_slots[0] not in ex.updates:
        slot = pending_slots[0]
        if slot == "component" and not ex.unsure and 0 < len(text.split()) <= 8:
            ex.updates["component"] = SlotUpdate(text.strip().strip(".").lower(), confidence=0.85)
            ex.manual_terms = manual_terms_for(text)
        if slot in ("symptom_or_dtc", "conditions", "symptoms") and not ex.unsure and text.strip():
            ex.updates["symptoms" if slot != "conditions" else "conditions"] = SlotUpdate(
                text.strip()[:300], confidence=0.85
            )
        if slot == "task_type" and not ex.unsure:
            low = text.lower()
            if re.search(r"\bboth\b|remove.*install|full", low):
                ex.updates["task_type"] = SlotUpdate("replace", confidence=0.85)
            elif re.search(r"diagnos", low):
                ex.updates["task_type"] = SlotUpdate("diagnose", confidence=0.85)
    return ex


# --------------------------------------------------------------------------- LLM
LLM_SYSTEM = """You extract structured context from a workshop technician's message for a Ford Mustang
service-manual assistant. Return ONLY a JSON object with these keys (null when not stated):
{"model_year": int|null, "engine_displacement": "e.g. 5.0L"|null, "transmission": "automatic"|"manual"|null,
 "body_style": "coupe"|"convertible"|null, "component": "part or system named by the user, in their words"|null,
 "task_type": "replace"|"removal"|"installation"|"repair"|"inspect"|"adjust"|"service"|"diagnose"|"specification"|"overview"|null,
 "symptoms": string|null, "conditions": "when the problem occurs"|null, "previous_checks": string|null,
 "measurements": string|null, "dtcs": [string], "refers_to_previous_answer": bool, "user_does_not_know": bool,
 "confirmation": "yes"|"no"|null, "immediate_hazard": string|null, "other_vehicle": string|null}
Never guess values the user did not state. Do not answer the question."""


def llm_extract(llm: LLMClient, model: str, text: str, state: SessionState, history: list[dict]) -> dict:
    context = {
        "known_vehicle": state.vehicle_summary(),
        "known_repair": state.repair_summary(),
        "pending_question": state.pending_question.text if state.pending_question else None,
        "active_procedure": state.active_procedure.title if state.active_procedure else None,
    }
    msgs = [
        {"role": "system", "content": LLM_SYSTEM},
        {"role": "user", "content": f"Context: {json.dumps(context, default=str)}\nRecent: "
         f"{json.dumps(history[-4:])}\nMessage: {text}"},
    ]  # fmt: skip
    return parse_json(llm.complete(msgs, model=model, json_mode=True, max_tokens=700, temperature=0))


def _supported(value: Any, text: str) -> bool:
    """A vehicle value from the LLM is 'user provided' only if the user's words contain it."""
    if value is None:
        return False
    v = str(value).lower().removesuffix("l")
    return v in text.lower()


def merge_llm(ex: Extraction, data: dict, text: str, known: dict | None = None) -> None:
    known = known or {}
    ex.llm_used = True
    for slot, key in (("model_year", "model_year"), ("engine", "engine_displacement")):
        val = data.get(key)
        if val and not _supported(val, text) and known.get(slot) is not None:
            continue  # the model is echoing context, not reporting something new
        if val and slot not in ex.updates and slot not in ex.conflicts:
            if slot == "engine":
                m = re.search(r"(\d\.\d)", str(val))
                val = f"{m.group(1)}L" if m else None
            if val:
                status = SlotStatus.USER_PROVIDED if _supported(val, text) else SlotStatus.INFERRED
                ex.updates[slot] = SlotUpdate(val, status, 0.85 if status == SlotStatus.USER_PROVIDED else 0.6,
                                              basis=None if status == SlotStatus.USER_PROVIDED else "language-model interpretation",
                                              source="llm")  # fmt: skip
    trans = data.get("transmission")
    if (
        trans in ("automatic", "manual")
        and known.get("transmission")
        and not (AUTO_RE if trans == "automatic" else MANUAL_RE).search(text)
    ):
        trans = None  # echo of known context
    if trans in ("automatic", "manual") and "transmission" not in ex.updates and "transmission" not in ex.conflicts:
        supported = bool((AUTO_RE if trans == "automatic" else MANUAL_RE).search(text))
        ex.updates["transmission"] = SlotUpdate(
            trans, SlotStatus.USER_PROVIDED if supported else SlotStatus.INFERRED, 0.85 if supported else 0.6,
            basis=None if supported else "language-model interpretation", source="llm",
        )  # fmt: skip
    body = data.get("body_style")
    if body in ("coupe", "convertible") and "body_style" not in ex.updates and body in text.lower():
        ex.updates["body_style"] = SlotUpdate(body, source="llm")
    for slot in ("component", "task_type", "symptoms", "conditions", "previous_checks", "measurements"):
        val = data.get(slot)
        if val and slot not in ex.updates:
            ex.updates[slot] = SlotUpdate(str(val)[:300] if slot != "component" else str(val).lower()[:80],
                                          confidence=0.8, source="llm")  # fmt: skip
            if slot == "component":
                ex.manual_terms = manual_terms_for(str(val))
    dtcs = [d.upper() for d in data.get("dtcs") or [] if DTC_RE.fullmatch(str(d).split(":")[0])]
    if dtcs and "dtcs" not in ex.updates:
        ex.updates["dtcs"] = SlotUpdate(dtcs, source="llm")
    if data.get("refers_to_previous_answer"):
        ex.is_followup = ex.is_followup or "component" not in ex.updates
    if data.get("user_does_not_know"):
        ex.unsure = True
    if ex.confirmation is None and data.get("confirmation") in ("yes", "no"):
        ex.confirmation = data["confirmation"] == "yes"
    if not ex.other_vehicle and data.get("other_vehicle") and "mustang" not in str(data["other_vehicle"]).lower():
        ex.other_vehicle = str(data["other_vehicle"])
    # LLM hazard flags are advisory only when rules agree on hazard vocabulary
    if not ex.hazard and data.get("immediate_hazard") and detect_hazard(str(data["immediate_hazard"])):
        ex.hazard = detect_hazard(str(data["immediate_hazard"]))


def extract(
    text: str,
    state: SessionState,
    profile: ManualProfile,
    llm: LLMClient | None = None,
    llm_model: str | None = None,
    history: list[dict] | None = None,
) -> Extraction:
    ex = rule_extract(text, state, profile)
    if llm is not None:
        try:
            merge_llm(ex, llm_extract(llm, llm_model or "", text, state, history or []), text, state.vehicle_summary())
        except Exception as exc:  # LLM is an enhancement; rules still work
            logger.warning("llm extraction skipped: %s", type(exc).__name__)
    return ex


# --------------------------------------------------------------------------- apply
def apply_extraction(state: SessionState, ex: Extraction, turn: int, text: str) -> list[str]:
    """Update slots without losing confirmed values. Returns change events."""
    events: list[str] = []
    pending = state.pending_question

    if ex.hazard:
        state.active_hazard, state.safety_acknowledged = ex.hazard, False
        events.append(f"hazard:{ex.hazard}")
    elif ex.safety_ack and state.active_hazard:
        state.safety_acknowledged = True
        events.append("hazard_acknowledged")

    # confirmation answers apply to the slots in the pending confirmation
    if pending and pending.kind == "confirmation" and ex.confirmation is not None:
        for name in pending.slots:
            s = state.slot(name)
            if ex.confirmation:
                if s.conflict_with is not None and name not in ex.updates:
                    events.append(f"conflict_unresolved:{name}")  # 'yes' cannot pick between two values
                    continue
                s.status, s.confidence, s.basis = SlotStatus.CONFIRMED, 1.0, s.basis
                state.set_slot(s)
                events.append(f"confirmed:{name}")
            elif name not in ex.updates:
                state.clear_slot(name)
                events.append(f"rejected:{name}")

    if ex.unsure and pending and pending.kind == "information":
        for name in pending.slots:
            if name not in ex.updates:
                state.set_slot(Slot(name=name, status=SlotStatus.USER_UNSURE, turn=turn, source_text=text[:200]))
                events.append(f"unsure:{name}")

    # task change: a new component (not a follow-up) resets repair context but keeps the vehicle
    if ex.is_followup and "component" in ex.updates and state.value("component"):
        ex.updates.pop("component")  # "the pump bolts" refines the active procedure; not a new task
    new_comp = ex.updates.get("component")
    old_comp = state.value("component")
    if new_comp and old_comp and new_comp.value != old_comp and not ex.is_followup:
        for name in REPAIR_SLOTS:
            if name != "component" and name not in ex.updates:
                state.clear_slot(name)
        state.active_procedure = None
        state.soft_asked = []
        events.append("task_changed")

    for name, values in ex.conflicts.items():
        state.set_slot(Slot(name=name, value=values[0], status=SlotStatus.USER_PROVIDED, confidence=0.5,
                            turn=turn, conflict_with=values[1:], source_text=text[:200]))  # fmt: skip
        events.append(f"conflict:{name}")

    for name, up in ex.updates.items():
        cur = state.slot(name)
        answering = pending is not None and name in pending.slots
        if cur.filled and cur.value != up.value and cur.status in (SlotStatus.CONFIRMED, SlotStatus.USER_PROVIDED):
            if name in ("task_type", "symptoms", "conditions", "previous_checks", "measurements", "component", "dtcs"):
                pass  # repair details naturally evolve
            elif not (ex.correction or answering):
                cur.conflict_with = up.value
                cur.confidence = min(cur.confidence, 0.5)
                state.set_slot(cur)
                events.append(f"conflict:{name}")
                continue
            else:
                events.append(f"corrected:{name}")
                if name in ("model_year", "engine", "transmission", "body_style"):
                    state.active_procedure = None  # vehicle changed: old evidence no longer applies
        if cur.filled and cur.value == up.value and STATUS_STRENGTH[cur.status] >= STATUS_STRENGTH[up.status]:
            continue  # a restatement never downgrades provenance (e.g. an LLM echo of a known fact)
        state.set_slot(Slot(name=name, value=up.value, status=up.status, confidence=up.confidence, turn=turn,
                            basis=up.basis, source_text=text[:200]))  # fmt: skip
        events.append(f"set:{name}")
    return events


def resolve_followup(state: SessionState) -> ActiveProcedure | None:
    return state.active_procedure
