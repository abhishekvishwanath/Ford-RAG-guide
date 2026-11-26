"""Parent/procedure context expansion under a token budget.

A retrieved child chunk may lack earlier steps, the special-tools list or a section
warning. For the best-matching procedure we add, in order of priority:
  1. the whole procedure when the task is a procedure and it fits the budget, else
  2. the procedure's first chunk (tools/warnings preamble), the hit chunks, their neighbours,
     and chunks covering steps the user referenced.
Other relevant procedures are included unexpanded. Evidence keeps manual sequence order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.applicability_filter import VehicleFilter
from app.schemas.chunk import Chunk
from app.schemas.retrieval import Candidate, EvidenceItem, RetrievalRequest
from app.services.manual_repository import ManualRepository

PROCEDURAL_TASKS = {"replace", "removal", "installation", "repair", "service", "adjust", "inspect", "disassembly"}
# title suffixes that distinguish otherwise-identical procedures (engine, body, caliper type, side…)
VARIANT_SUFFIX_RE = re.compile(
    r"\s*[—–-]\s*(\d\.\dL.*|GT500.*|Coupe|Convertible|Shelby.*|\d\s*Piston.*|LH|RH|Front|Rear|Center"
    r"|Automatic Transmission|Manual Transmission)$",
    re.I,
)


@dataclass
class ExpansionConfig:
    top_k: int = 5
    max_chunks: int = 8
    token_budget: int = 9000
    max_other_procedures: int = 3


R_AND_I = {"REMOVAL AND INSTALLATION", "IN-VEHICLE REPAIR", "REMOVAL", "INSTALLATION"}
SHOP = {"GENERAL PROCEDURES", "DISASSEMBLY AND ASSEMBLY", "DISASSEMBLY AND ASSEMBLY OF SUBASSEMBLIES", "DISASSEMBLY",
        "ASSEMBLY"}  # fmt: skip
DOC_TYPE_FIT: dict[str, dict[str, float]] = {
    # task -> {doc type: bonus}; a procedure of a fitting type is preferred as the main one
    "replace": {**dict.fromkeys(R_AND_I, 0.35), **dict.fromkeys(SHOP, 0.15)},
    "removal": {**dict.fromkeys(R_AND_I, 0.35), **dict.fromkeys(SHOP, 0.15)},
    "installation": {**dict.fromkeys(R_AND_I, 0.35), **dict.fromkeys(SHOP, 0.15)},
    "repair": {**dict.fromkeys(R_AND_I, 0.35), **dict.fromkeys(SHOP, 0.25)},
    "disassembly": {**dict.fromkeys(SHOP, 0.35), **dict.fromkeys(R_AND_I, 0.15)},
    "inspect": {"GENERAL PROCEDURES": 0.35, "DIAGNOSIS AND TESTING": 0.2},
    "adjust": {"GENERAL PROCEDURES": 0.35},
    "service": {"GENERAL PROCEDURES": 0.35, **dict.fromkeys(R_AND_I, 0.15)},
    "diagnose": {"DIAGNOSIS AND TESTING": 0.35},
    "overview": {"DESCRIPTION AND OPERATION": 0.35},
    "specification": {"SPECIFICATIONS": 0.35},
}
STOP = {"and", "the", "a", "of", "to", "on", "in", "for", "with", "my", "is", "it", "how", "do", "i", "what"}


def _title_words(s: str | None) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in STOP and len(w) > 1}


def choose_main_parent(req: RetrievalRequest, hits: list[Chunk]) -> str:
    """Pick the procedure the answer is about: rank weight + task/doc-type fit + title match.

    The top-ranked chunk alone is not enough: a 'Clutch' diagnosis page can outrank the 'Clutch Disc and
    Pressure Plate' removal procedure for a replacement job, and the front hub can outrank the rear one."""
    fit = DOC_TYPE_FIT.get(req.task_type or "", {})
    asked = _title_words(" ".join(filter(None, [req.original_message, req.task_text, req.component])))
    scores: dict[str, float] = {}
    for rank, h in enumerate(hits):
        scores[h.parent_id] = scores.get(h.parent_id, 0.0) + 1.0 / (rank + 2)
    seen: set[str] = set()
    for h in hits:
        if h.parent_id in seen:
            continue
        seen.add(h.parent_id)
        scores[h.parent_id] += fit.get(h.doc_type or "", 0.0)
        # each title word the technician used counts: "Clutch Disc and Pressure Plate" beats "Clutch"
        tw = _title_words(base_title(h.procedure_title) if h.procedure_title else None)
        scores[h.parent_id] += 0.15 * len(tw & asked)
    return max(scores, key=lambda p: (scores[p], -next(i for i, h in enumerate(hits) if h.parent_id == p)))


SPEC_Q_RE = re.compile(r"\b(spec|specs|specification|torque|tighten|gap|clearance|capacity|limits?|angle|tolerance)\b", re.I)


def section_spec_chunk(req: RetrievalRequest, main: Chunk, repo: ManualRepository, flt: VehicleFilter) -> Chunk | None:
    """For a spec question about a procedure, the section's SPECIFICATIONS table chunk that best matches it."""
    if not SPEC_Q_RE.search(req.original_message) or (main.doc_type or "") == "SPECIFICATIONS":
        return None
    asked = _title_words(" ".join(filter(None, [req.original_message, req.component, main.procedure_title])))
    asked -= {"spec", "specification", "torque", "what", "mustang", "doing", "adjusted", "adjust", "adjustment"}
    best, best_score = None, 0
    for p in repo.procedures.values():
        if p.section_code != main.section_code or (p.doc_type or "") != "SPECIFICATIONS":
            continue
        for cid in p.chunk_ids:
            ch = repo.chunk(cid)
            if ch is None or not flt.allows(ch):
                continue
            score = len(asked & _title_words(ch.content))
            if score > best_score:
                best, best_score = ch, score
    return best if best_score >= 1 else None


def base_title(title: str | None) -> str:
    return VARIANT_SUFFIX_RE.sub("", title or "").strip().lower()


def detect_variant_ambiguity(cands: list[Candidate], repo: ManualRepository, flt: VehicleFilter) -> list[str]:
    """Two compatible procedures that differ only by a variant suffix the user hasn't specified."""
    return list(variant_options(cands, repo))


AXLE_RE = re.compile(r"\b(front|rear)\b", re.I)


def variant_options(
    cands: list[Candidate], repo: ManualRepository, flt: VehicleFilter | None = None, message: str = ""
) -> dict[str, str]:
    """title -> parent_id for procedures that differ only by a variant suffix.

    Ambiguity is detected among the top candidates; the options offered are then completed from the whole
    catalogue (same base title and doc type, compatible with the vehicle), so a '4 Piston' variant that did
    not happen to rank is still offered. 'front'/'rear' in the question removes the other axle's sections."""
    groups: dict[str, dict[str, str]] = {}
    doc_types: dict[str, set[str]] = {}
    for c in cands:
        ch = repo.chunk(c.chunk_id)
        if ch is None or not ch.procedure_title:
            continue
        key = base_title(ch.procedure_title)
        groups.setdefault(key, {}).setdefault(ch.procedure_title, ch.parent_id)
        doc_types.setdefault(key, set()).add(ch.doc_type or "")
    axle = AXLE_RE.search(message)
    other_axle = {"front": "rear", "rear": "front"}[axle.group(1).lower()] if axle else None
    out: dict[str, str] = {}
    for key, variants in groups.items():
        if len(variants) < 2:
            continue
        for p in repo.procedures.values():
            if not p.title or base_title(p.title) != key or (p.doc_type or "") not in doc_types[key]:
                continue
            first = repo.chunk(p.chunk_ids[0]) if p.chunk_ids else None
            if flt is not None and first is not None and not flt.allows(first):
                continue
            variants.setdefault(p.title, p.parent_id)
        if other_axle:
            for title, pid in list(variants.items()):
                proc = repo.procedures.get(pid)
                where = f"{proc.section_name if proc else ''} {title}".lower()
                if other_axle in where and len(variants) > 1:
                    del variants[title]
        if len(variants) > 1:
            out.update(dict(sorted(variants.items())))
    return out


def expand(
    req: RetrievalRequest,
    ranked: list[Candidate],
    repo: ManualRepository,
    flt: VehicleFilter,
    cfg: ExpansionConfig,
) -> list[EvidenceItem]:
    primary = ranked[: cfg.top_k]
    if not primary:
        return []
    score_of = {c.chunk_id: (c.rerank_score if c.rerank_score is not None else c.rrf_score) for c in ranked}
    hits: list[Chunk] = [h for h in (repo.chunk(c.chunk_id) for c in primary) if h is not None]
    main_parent = req.focus_parent_id if req.focus_parent_id in {h.parent_id for h in hits} else choose_main_parent(req, hits)
    if req.focus_parent_id and req.focus_parent_id in repo.procedures and req.focus_steps:
        main_parent = req.focus_parent_id

    selected: dict[str, bool] = {}  # chunk_id -> expanded?
    budget = cfg.token_budget

    def add(ch: Chunk, expanded: bool) -> bool:
        nonlocal budget
        if ch.chunk_id in selected or not flt.allows(ch):
            return False
        if ch.token_count > budget or len(selected) >= cfg.max_chunks:
            return False
        selected[ch.chunk_id] = expanded
        budget -= ch.token_count
        return True

    family = repo.parent_chunks(main_parent)
    family_ok = [c for c in family if flt.allows(c)]
    hit_ids = {h.chunk_id for h in hits}
    total = sum(c.token_count for c in family_ok)
    if req.task_type in PROCEDURAL_TASKS and total <= cfg.token_budget * 0.75 and len(family_ok) <= cfg.max_chunks:
        for c in family_ok:
            add(c, c.chunk_id not in hit_ids)
    else:
        if req.focus_steps:
            for c in family_ok:
                if c.step_range and any(c.step_range[0] <= s <= c.step_range[1] for s in req.focus_steps):
                    add(c, c.chunk_id not in hit_ids)
        for h in hits:
            if h.parent_id == main_parent:
                add(h, False)
        if (
            family_ok
            and family_ok[0].chunk_id not in selected
            and ("has_special_tools" in family_ok[0].flags or "has_warning" in family_ok[0].flags)
        ):
            add(family_ok[0], True)
        for h in [h for h in hits if h.parent_id == main_parent]:
            for nb in family_ok:
                if abs(nb.sequence_index - h.sequence_index) == 1:
                    add(nb, True)

    main_chunk = next((c for c in family_ok), None)
    spec = section_spec_chunk(req, main_chunk, repo, flt) if main_chunk else None
    if spec is not None and spec.chunk_id not in selected:
        if len(selected) >= cfg.max_chunks:  # make room: drop the last expanded neighbour
            drop = next((cid for cid in reversed(list(selected)) if selected[cid]), None)
            if drop:
                budget += repo.chunk(drop).token_count  # type: ignore[union-attr]
                del selected[drop]
        add(spec, False)

    others = 0
    seen_parents = {main_parent}
    for h in hits:
        if h.parent_id in seen_parents:
            if h.parent_id != main_parent:
                add(h, False)
            continue
        if others >= cfg.max_other_procedures:
            break
        if add(h, False):
            others += 1
            seen_parents.add(h.parent_id)

    chunks = [repo.chunk(cid) for cid in selected]
    order = {main_parent: 0}  # main procedure first, others in rank order
    for i, h in enumerate(hits):
        order.setdefault(h.parent_id, i + 1)
    chunks.sort(key=lambda c: (order.get(c.parent_id, 99), c.sequence_index))  # type: ignore[union-attr]

    items: list[EvidenceItem] = []
    for i, c in enumerate(chunks, start=1):
        assert c is not None
        items.append(
            EvidenceItem(
                source_id=f"S{i}",
                chunk_id=c.chunk_id,
                parent_id=c.parent_id,
                procedure_title=c.procedure_title,
                section=" ".join(filter(None, [c.section_code, c.section_name])) or None,
                doc_type=c.doc_type,
                subsection=c.subsection,
                page_start=c.page_start,
                page_end=c.page_end,
                printed_page_start=c.printed_page_start,
                printed_page_end=c.printed_page_end,
                applicability_note=flt.note(c),
                content=c.content,
                figure_ocr=c.figure_ocr,
                figure_ids=c.figure_ids,
                table_ids=c.table_ids,
                expanded=selected[c.chunk_id],
                score=round(score_of.get(c.chunk_id, 0.0), 4),
            )
        )
    return items
