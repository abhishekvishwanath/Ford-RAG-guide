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
VARIANT_SUFFIX_RE = re.compile(r"\s*[—–-]\s*(\d\.\dL.*|GT500.*|Coupe|Convertible|Shelby.*)$", re.I)


@dataclass
class ExpansionConfig:
    top_k: int = 5
    max_chunks: int = 8
    token_budget: int = 9000
    max_other_procedures: int = 3


def base_title(title: str | None) -> str:
    return VARIANT_SUFFIX_RE.sub("", title or "").strip().lower()


def detect_variant_ambiguity(cands: list[Candidate], repo: ManualRepository, flt: VehicleFilter) -> list[str]:
    """Two compatible procedures that differ only by a variant suffix the user hasn't specified."""
    titles: dict[str, set[str]] = {}
    for c in cands:
        ch = repo.chunk(c.chunk_id)
        if ch is None or not ch.procedure_title:
            continue
        titles.setdefault(base_title(ch.procedure_title), set()).add(ch.procedure_title)
    out: list[str] = []
    for variants in titles.values():
        if len(variants) > 1:
            out.extend(sorted(variants))
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
    main_parent = req.focus_parent_id if req.focus_parent_id in {h.parent_id for h in hits} else hits[0].parent_id
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
    order = {main_parent: 0}
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
