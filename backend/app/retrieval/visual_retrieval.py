"""Original-diagram and table retrieval for selected evidence.

Visuals come only from figure records associated with the evidence chunks during
ingestion (never generated). Ranking: association strength, then manual order
(evidence order, page, position), so multi-image sequences keep their order.
"""

from __future__ import annotations

from app.schemas.retrieval import EvidenceItem, ImageRef, TableRef
from app.services.manual_repository import ManualRepository

METHOD_RANK = {"in_step": 0, "legend_table": 1, "section_preamble": 2, "page_proximity": 3, "special_tools": 4}
MIN_ASSOCIATION_CONFIDENCE = 0.6


def retrieve_images(items: list[EvidenceItem], repo: ManualRepository, limit: int) -> list[ImageRef]:
    candidates = []
    for order, item in enumerate(items):
        for fid in item.figure_ids:
            f = repo.figures.get(fid)
            if f is None or f.association_confidence < MIN_ASSOCIATION_CONFIDENCE:
                continue
            candidates.append((order, f, item))
    # illustrations first; tool icons only when room is left
    main_parent = items[0].parent_id if items else None
    candidates.sort(key=lambda t: (t[2].parent_id != main_parent, t[1].image_type != "illustration",
                                   METHOD_RANK.get(t[1].association_method, 9), t[0], t[1].pdf_page_index,
                                   t[1].bbox[1]))  # fmt: skip
    picked = candidates[:limit]
    picked.sort(key=lambda t: (t[0], t[1].pdf_page_index, t[1].bbox[1]))  # present in manual order
    out: list[ImageRef] = []
    seen_assets: set[str] = set()
    for _, f, item in picked:
        if f.asset_id in seen_assets:
            continue
        seen_assets.add(f.asset_id)
        where = " — ".join(filter(None, [f.procedure_title, f.subsection]))
        step = f", step {f.nearest_step}" if f.nearest_step else ""
        page = f"PDF page {f.page_number}" + (f" ({f.printed_page})" if f.printed_page else "")
        kind = "Special tool" if f.image_type == "tool_icon" else "Manual illustration"
        out.append(
            ImageRef(
                asset_id=f.asset_id,
                figure_id=f.figure_id,
                url=f"/api/manual-assets/{f.asset_id}",
                caption=f"{kind}: {where}{step}; {page}",
                figure_numbers=f.figure_numbers,
                page_number=f.page_number,
                printed_page=f.printed_page,
                procedure_title=f.procedure_title,
                section=f.section_code,
                nearest_step=f.nearest_step,
                association=f.association_method,
                confidence=f.association_confidence,
                source_id=item.source_id,
            )
        )
    return out


def retrieve_tables(items: list[EvidenceItem], repo: ManualRepository, limit: int = 6) -> list[TableRef]:
    out: list[TableRef] = []
    for item in items:
        for tid in item.table_ids:
            t = repo.tables.get(tid)
            if t is None or not t.rows:
                continue
            out.append(TableRef(table_id=tid, source_id=item.source_id, header=t.header, rows=t.rows,
                                page_number=t.pdf_page_index + 1))  # fmt: skip
            if len(out) >= limit:
                return out
    return out
