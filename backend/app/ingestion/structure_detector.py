"""Document-level structure: pages -> procedure documents with hierarchy.

* Procedure boundaries come from the source procedure id printed in each page
  footer (the strongest signal in this print format).
* Hierarchy labels come from the PDF bookmark tree, tolerating wrong targets.
* Element post-processing fixes sequence semantics that span paragraphs/pages:
  - admonitions that open a step are moved *before* the step they govern,
    and the step absorbs its action text;
  - indented re-numbered items inside a step become sub-steps;
  - header-less tables at the top of a page continue the previous page's table.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from difflib import SequenceMatcher

from app.ingestion.boilerplate import MANUAL_LABEL_RE, REVISION_RE
from app.schemas.document import Element, ElementKind, PageRecord, ProcedureDocument

SUBSTEP_INDENT = 8.0
_BREAKS = {ElementKind.STEP, ElementKind.HEADING, ElementKind.QUALIFIER, ElementKind.TITLE, ElementKind.TABLE}


class TocIndex:
    """Resolve bookmark paths for a start page, tolerant to misplaced bookmarks."""

    def __init__(self, toc: list[list]) -> None:
        self.entries: list[tuple[int, list[str]]] = []  # (1-based page, path)
        stack: list[str] = []
        for level, title, page in toc:
            stack = stack[: level - 1] + [" ".join(str(title).split())]
            self.entries.append((int(page), list(stack)))

    def path_for(self, pdf_page_index: int, title: str | None) -> list[str]:
        page = pdf_page_index + 1
        same = [p for pg, p in self.entries if pg == page]
        if same:
            if title:
                best = max(same, key=lambda p: (SequenceMatcher(None, p[-1].lower(), title.lower()).ratio(), len(p)))
                return best
            return max(same, key=len)
        before = [(pg, p) for pg, p in self.entries if pg < page]
        if not before:
            return []
        max_pg = max(pg for pg, _ in before)
        return max((p for pg, p in before if pg == max_pg), key=len)


def iter_procedure_groups(pages: Iterable[PageRecord]) -> Iterator[list[PageRecord]]:
    """Stream consecutive pages sharing a procedure id (pages must be in page order)."""
    group: list[PageRecord] = []
    current_id: str | None = None
    for rec in pages:
        pid = rec.seb_id or current_id
        if rec.seb_id is None and current_id is not None:
            rec.diagnostics.warnings.append("procedure_id_inherited_from_previous_page")
            rec.seb_id = current_id
        # the same id printed again from "Page 1" is a repeated copy -> separate document
        restarted = bool(group) and rec.printed_page == 1 and group[-1].printed_page not in (None, 0)
        if group and (pid != current_id or restarted):
            yield group
            group = []
        group.append(rec)
        current_id = pid
    if group:
        yield group


def _fix_substeps(elements: list[Element]) -> list[Element]:
    main_x: float | None = None
    last_main = 0
    out: list[Element] = []
    for el in elements:
        if el.kind == ElementKind.STEP and el.bbox is not None:
            x = el.bbox[0]
            if main_x is None or x <= main_x + SUBSTEP_INDENT:
                if main_x is None or x < main_x:
                    main_x = x
                last_main = el.step_number or last_main
            elif (el.step_number or 0) <= last_main:
                el = el.model_copy(
                    update={"kind": ElementKind.LIST_ITEM, "marker": f"{el.step_number}.", "step_number": None}
                )
        elif el.kind in (ElementKind.HEADING, ElementKind.TITLE):
            last_main = 0
        out.append(el)
    return out


_JOINABLE = {ElementKind.STEP, ElementKind.ADMONITION, ElementKind.LIST_ITEM, ElementKind.PARAGRAPH}


def _join_continuations(elements: list[Element]) -> list[Element]:
    """A paragraph starting in lower case continues the previous text element (wrap/page break)."""
    out: list[Element] = []
    last_text_idx: int | None = None
    for el in elements:
        if (
            el.kind == ElementKind.PARAGRAPH
            and el.text[:1].islower()
            and last_text_idx is not None
            and out[last_text_idx].kind in _JOINABLE
        ):
            prev = out[last_text_idx]
            out[last_text_idx] = prev.model_copy(update={"text": f"{prev.text} {el.text}".strip()})
            continue
        out.append(el)
        if el.kind != ElementKind.FIGURE:
            last_text_idx = len(out) - 1 if el.text.strip() or el.kind == ElementKind.STEP else last_text_idx
    return out


def _attach_step_admonitions(elements: list[Element]) -> list[Element]:
    """[STEP n (empty), ADM, ADM, PARAGRAPH] -> [ADM(step n), ADM(step n), STEP n(text=PARAGRAPH)]."""
    out: list[Element] = []
    i = 0
    while i < len(elements):
        el = elements[i]
        if el.kind == ElementKind.STEP and not el.text.strip():
            j = i + 1
            adms: list[Element] = []
            between: list[Element] = []
            action: Element | None = None
            while j < len(elements):
                nxt = elements[j]
                if nxt.kind == ElementKind.ADMONITION:
                    adms.append(nxt.model_copy(update={"step_number": el.step_number}))
                elif nxt.kind == ElementKind.PARAGRAPH and action is None:
                    action = nxt
                    j += 1
                    break
                elif nxt.kind == ElementKind.FIGURE:
                    between.append(nxt)
                else:
                    break
                j += 1
            out.extend(adms)
            out.append(el.model_copy(update={"text": action.text if action else ""}))
            out.extend(between)
            i = j
            continue
        out.append(el)
        i += 1
    return out


def _link_tables(elements: list[Element]) -> list[Element]:
    prev_table: Element | None = None
    out: list[Element] = []
    for el in elements:
        if el.kind == ElementKind.TABLE and el.table is not None:
            t = el.table
            if prev_table is not None and prev_table.table is not None and not t.header:
                pt = prev_table.table
                width = len(pt.header or (pt.rows[0] if pt.rows else []))
                cur_width = len(t.rows[0]) if t.rows else len(t.leading_fragment or [])
                if width == cur_width and el.pdf_page_index == prev_table.pdf_page_index + 1:
                    t = t.model_copy(update={"continued_from": pt.table_id, "header": pt.header})
                    if t.leading_fragment and pt.rows and len(pt.rows[-1]) == len(t.leading_fragment):
                        pt.rows[-1] = [
                            (a + " " + b).strip() for a, b in zip(pt.rows[-1], t.leading_fragment, strict=True)
                        ]
                        prev_table.text = pt.to_text()
                    el = el.model_copy(update={"table": t, "text": t.to_text()})
            prev_table = el
        elif el.kind != ElementKind.FIGURE:
            prev_table = None if el.kind in _BREAKS else prev_table
        out.append(el)
    return out


def build_procedure(pages: list[PageRecord], toc: TocIndex) -> ProcedureDocument:
    elements = [e for p in pages for e in p.elements]
    doc = ProcedureDocument(
        procedure_id=pages[0].seb_id or f"UNKNOWN_{pages[0].pdf_page_index}",
        pdf_page_start=pages[0].pdf_page_index,
        pdf_page_end=pages[-1].pdf_page_index,
        printed_total=pages[0].printed_total,
        manual_label=next((p.manual_label for p in pages if p.manual_label), None),
        printed_pages={p.pdf_page_index: p.printed_page for p in pages if p.printed_page is not None},
    )
    if pages[0].printed_page not in (None, 1):
        doc.warnings.append(f"starts_mid_document_at_printed_page_{pages[0].printed_page}")
    for el in elements:
        if el.kind == ElementKind.TITLE and doc.title is None:
            doc.title = el.text
        elif el.kind == ElementKind.SECTION_HEADER and doc.section_code is None:
            code, _, name = el.text.partition(":")
            doc.section_code, doc.section_name = code.strip(), name.strip()
        elif el.kind == ElementKind.DOC_TYPE and doc.doc_type is None and not MANUAL_LABEL_RE.match(el.text):
            doc.doc_type = el.text
        elif el.kind == ElementKind.REVISION and doc.revision_date is None and (m := REVISION_RE.match(el.text)):
            doc.revision_date = m.group(1)
    body = [
        e for e in elements if e.kind not in (ElementKind.SECTION_HEADER, ElementKind.DOC_TYPE, ElementKind.REVISION)
    ]
    body = _fix_substeps(body)
    body = _join_continuations(body)
    body = _attach_step_admonitions(body)
    body = _link_tables(body)
    doc.elements = body
    doc.toc_path = toc.path_for(doc.pdf_page_start, doc.title)
    if doc.title is None:
        doc.title = doc.toc_path[-1] if doc.toc_path else None
        doc.warnings.append("title_from_bookmarks" if doc.title else "no_title")
    return doc


def iter_procedures(pages: Iterable[PageRecord], toc: list[list]) -> Iterator[ProcedureDocument]:
    index = TocIndex(toc)
    for group in iter_procedure_groups(pages):
        yield build_procedure(group, index)
