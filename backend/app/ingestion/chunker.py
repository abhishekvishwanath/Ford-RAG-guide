"""Structure-based hierarchical chunking.

Hierarchy: manual group (L1) -> section (L2) -> procedure document (L3) -> sub-section /
step-group child chunks (L4).

Rules implemented here (see docs/architecture.md):
* A procedure no longer than ``heading_split_word_threshold`` words stays one chunk.
* Otherwise split at sub-section headings (Removal / Installation / Pinpoint test ...);
  tiny sections merge forward (e.g. Special Tools + legend join Removal).
* Sections above the max token size are packed from *atomic units*: a step together with
  its warnings, bullets and figures; a table; a qualifier-scoped block. Units are never
  split, except tables (by rows, header repeated) and, as a last resort, an oversize single
  unit (token fallback with overlap).
* Continuation chunks repeat the procedure context, the active applicability qualifier and
  the section's WARNING/CAUTION preamble.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.embeddings.tokenizer import TokenCounter
from app.ingestion import metadata as md
from app.schemas.chunk import Applicability, Chunk, FigureRecord, ProcedureRecord
from app.schemas.document import Element, ElementKind, ProcedureDocument

TORQUE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:Nm|N·m|lb-ft|lb-in)\b")
DTC_RE = re.compile(r"\b[BCPU][0-9A-F]{4}(?::[0-9A-F]{2})?\b")
SMALL_SECTION_TOKENS = 150


@dataclass
class ChunkingConfig:
    target_tokens: int = 550
    max_tokens: int = 900
    overlap_tokens: int = 60
    heading_split_word_threshold: int = 400


@dataclass
class Unit:
    kind: str  # step | table | text | qualifier | admonition
    elements: list[Element]
    qualifier: str | None = None
    text: str = ""
    tokens: int = 0

    @property
    def step_numbers(self) -> list[int]:
        return [e.step_number for e in self.elements if e.kind == ElementKind.STEP and e.step_number is not None]


@dataclass
class Section:
    heading: str | None
    units: list[Unit] = field(default_factory=list)

    def tokens(self) -> int:
        return sum(u.tokens for u in self.units)


# ------------------------------------------------------------------ rendering
def render_element(el: Element) -> str:
    k = el.kind
    if k == ElementKind.STEP:
        return f"{el.step_number}. {el.text}".rstrip()
    if k == ElementKind.LIST_ITEM:
        return f"   {el.marker or '-'} {el.text}"
    if k == ElementKind.QUALIFIER:
        return f"[{el.text}]"
    if k == ElementKind.HEADING:
        return f"### {el.text}"
    if k == ElementKind.TABLE and el.table is not None:
        return el.table.to_text()
    if k == ElementKind.FIGURE and el.figure is not None:
        return f"[Figure {el.figure.figure_id}]" if el.figure.image_type == "illustration" else ""
    return el.text


def render(elements: list[Element]) -> str:
    return "\n".join(t for t in (render_element(e) for e in elements) if t)


# ------------------------------------------------------------------ units/sections
def build_sections(doc: ProcedureDocument) -> list[Section]:
    sections: list[Section] = [Section(heading=None)]
    qualifier: str | None = None
    pending_adm: list[Element] = []
    pending_figs: list[Element] = []
    current_step: Unit | None = None

    def add(unit: Unit) -> None:
        nonlocal current_step
        if pending_figs:
            unit.elements[:0] = pending_figs.copy()
            pending_figs.clear()
        sections[-1].units.append(unit)
        current_step = unit if unit.kind == "step" else None

    for el in doc.elements:
        k = el.kind
        if k == ElementKind.TITLE:
            continue
        if k == ElementKind.HEADING:
            sections.append(Section(heading=el.text))
            qualifier, current_step = None, None
            continue
        if k == ElementKind.QUALIFIER:
            qualifier = el.text
            add(Unit("qualifier", [el], qualifier=qualifier))
            continue
        if k == ElementKind.ADMONITION and el.step_number is not None:
            pending_adm.append(el)
            continue
        if k == ElementKind.STEP:
            add(Unit("step", [*pending_adm, el], qualifier=qualifier))
            pending_adm.clear()
            continue
        if pending_adm:  # admonitions attached to a step that never appeared
            add(Unit("admonition", pending_adm.copy(), qualifier=qualifier))
            pending_adm.clear()
        if k in (ElementKind.LIST_ITEM, ElementKind.PARAGRAPH) and current_step is not None:
            current_step.elements.append(el)
            continue
        if k == ElementKind.FIGURE:
            if current_step is not None:
                current_step.elements.append(el)
            else:
                pending_figs.append(el)  # e.g. exploded view preceding its legend table
            continue
        if k == ElementKind.TABLE:
            add(Unit("table", [el], qualifier=qualifier))
            continue
        if k == ElementKind.ADMONITION:
            add(Unit("admonition", [el], qualifier=qualifier))
            continue
        add(Unit("text", [el], qualifier=qualifier))
    if pending_adm:
        add(Unit("admonition", pending_adm.copy(), qualifier=qualifier))
    if pending_figs:
        target = sections[-1].units[-1] if sections[-1].units else None
        if target is None:
            sections[-1].units.append(Unit("text", pending_figs.copy(), qualifier=qualifier))
        else:
            target.elements.extend(pending_figs)
    return [s for s in sections if s.units]


# ------------------------------------------------------------------ classification
def classify_chunk(doc: ProcedureDocument, heading: str | None, text: str, units: list[Unit]) -> str:
    dt = (doc.doc_type or "").upper()
    h = (heading or "").lower()
    title = (doc.title or "").lower()
    table_tokens = sum(u.tokens for u in units if u.kind == "table")
    total = sum(u.tokens for u in units) or 1
    mostly_table = table_tokens / total > 0.6
    header_text = " ".join(" ".join(e.table.header) for u in units for e in u.elements if e.table is not None).lower()
    if "pin" in header_text and ("circuit" in header_text or "connector" in header_text):
        return "connector_reference"
    if "wiring" in h or "wiring" in title:
        return "wiring"
    if "SPECIFICATION" in dt or "specification" in h:
        return "torque_table" if TORQUE_RE.search(text) or "nm" in header_text else "specification"
    if "symptom" in h or "symptom" in title:
        return "troubleshooting"
    if "DIAGNOSIS" in dt or "pinpoint" in h or "dtc" in h:
        return "diagnostic"
    if "DESCRIPTION" in dt:
        return "overview"
    if h.startswith("removal") or dt == "REMOVAL":
        return "removal"
    if h.startswith("installation") or dt == "INSTALLATION":
        return "installation"
    if re.search(r"adjust|alignment|aim", title + " " + h):
        return "adjustment"
    if re.search(r"inspect|check|measure|test", title + " " + h):
        return "inspection"
    if mostly_table:
        return "table"
    if units and all(u.kind == "admonition" for u in units):
        return "warning"
    return "procedure"


def content_flags(text: str, units: list[Unit]) -> list[str]:
    flags = []
    adm_types = {e.admonition_type for u in units for e in u.elements if e.admonition_type}
    if adm_types & {"WARNING", "CAUTION", "DANGER"}:
        flags.append("has_warning")
    if "NOTICE" in adm_types:
        flags.append("has_notice")
    if TORQUE_RE.search(text):
        flags.append("has_torque")
    if any(u.kind == "table" for u in units):
        flags.append("has_table")
    if DTC_RE.search(text):
        flags.append("has_dtc")
    if any(e.kind == ElementKind.FIGURE for u in units for e in u.elements):
        flags.append("has_figure")
    if re.search(r"special tool", text, re.I):
        flags.append("has_special_tools")
    return flags


# ------------------------------------------------------------------ chunker
class HierarchicalChunker:
    def __init__(self, cfg: ChunkingConfig, counter: TokenCounter, document_id: str, document_title: str,
                 source_hash: str, model_years: list[int]) -> None:  # fmt: skip
        self.cfg = cfg
        self.counter = counter
        self.document_id = document_id
        self.document_title = document_title
        self.source_hash = source_hash
        self.model_years = model_years
        self._seen_ids: dict[str, int] = {}

    # ---------------------------------------------------------- helpers
    def _procedure_key(self, doc: ProcedureDocument) -> str:
        n = self._seen_ids.get(doc.procedure_id, 0)
        self._seen_ids[doc.procedure_id] = n + 1
        return doc.procedure_id if n == 0 else f"{doc.procedure_id}_{n + 1}"

    def _context_header(self, doc: ProcedureDocument, subsection: str | None, continued: bool) -> str:
        parts = [
            self.document_title,
            " ".join(filter(None, [doc.section_code, doc.section_name])) or None,
            doc.doc_type,
            doc.title,
            (subsection + (" (continued)" if continued else ""))
            if subsection
            else ("(continued)" if continued else None),
        ]
        return " > ".join(p for p in parts if p)

    def _measure(self, units: list[Unit]) -> None:
        for u in units:
            u.text = render(u.elements)
            u.tokens = self.counter.count(u.text)

    @staticmethod
    def _preamble_warnings(section: Section) -> list[Element]:
        out = []
        for u in section.units:
            if u.kind == "step":
                break
            out.extend(e for e in u.elements if e.admonition_type in ("WARNING", "CAUTION", "DANGER"))
        return out

    def _split_table_unit(self, unit: Unit, target: int) -> list[Unit]:
        el = unit.elements[-1]
        t = el.table
        assert t is not None
        figs = [e for e in unit.elements if e.kind == ElementKind.FIGURE]
        header = " | ".join(t.header)
        budget = target - self.counter.count(header) - 10
        parts: list[Unit] = []
        rows: list[list[str]] = []
        used = 0
        for r in t.rows:
            rt = self.counter.count(" | ".join(r))
            if rows and used + rt > budget:
                parts.append(self._table_part(unit, el, rows, figs if not parts else []))
                rows, used = [], 0
            rows.append(r)
            used += rt
        if rows:
            parts.append(self._table_part(unit, el, rows, figs if not parts else []))
        return parts

    def _table_part(self, unit: Unit, el: Element, rows: list[list[str]], figs: list[Element]) -> Unit:
        assert el.table is not None
        t = el.table.model_copy(update={"rows": rows})
        part = Unit("table", [*figs, el.model_copy(update={"table": t, "text": t.to_text()})], qualifier=unit.qualifier)
        part.text = render(part.elements)
        part.tokens = self.counter.count(part.text)
        return part

    def _split_oversize_unit(self, unit: Unit, target: int) -> list[Unit]:
        windows = self.counter.split_by_tokens(unit.text, target, self.cfg.overlap_tokens)
        parts = []
        for i, w in enumerate(windows):
            figs = [e for e in unit.elements if e.kind == ElementKind.FIGURE] if i == 0 else []
            u = Unit(unit.kind, [*figs, unit.elements[-1].model_copy(update={"text": w})], qualifier=unit.qualifier)
            u.text, u.tokens = w, self.counter.count(w)
            parts.append(u)
        return parts

    def _pack(self, units: list[Unit], overhead: int = 0) -> list[tuple[list[Unit], str]]:
        """Greedy packing at unit boundaries; returns (units, split_reason) groups.

        ``overhead`` = tokens of the context header + repeated warnings added to every chunk."""
        cfg = ChunkingConfig(
            target_tokens=max(64, self.cfg.target_tokens - overhead),
            max_tokens=max(96, self.cfg.max_tokens - overhead),
            overlap_tokens=self.cfg.overlap_tokens,
            heading_split_word_threshold=self.cfg.heading_split_word_threshold,
        )
        expanded: list[tuple[Unit, str]] = []
        for u in units:
            if u.tokens > cfg.max_tokens and u.kind == "table":
                expanded.extend((p, "table_rows") for p in self._split_table_unit(u, cfg.target_tokens))
            elif u.tokens > cfg.max_tokens:
                expanded.extend((p, "token_fallback") for p in self._split_oversize_unit(u, cfg.target_tokens))
            else:
                expanded.append((u, "step_group"))
        groups: list[tuple[list[Unit], str]] = []
        cur: list[Unit] = []
        cur_tokens = 0
        cur_reason = "step_group"
        for u, reason in expanded:
            fits_target = cur_tokens + u.tokens <= cfg.target_tokens
            fits_max_small = cur_tokens < cfg.target_tokens * 0.5 and cur_tokens + u.tokens <= cfg.max_tokens
            if cur and not (fits_target or fits_max_small):
                # never end a chunk on a dangling qualifier line
                carry = [cur.pop()] if cur and cur[-1].kind == "qualifier" else []
                if cur:
                    groups.append((cur, cur_reason))
                cur, cur_tokens = carry, sum(c.tokens for c in carry)
                cur_reason = "step_group"
            cur.append(u)
            cur_tokens += u.tokens
            if reason != "step_group":
                cur_reason = reason
        if cur:
            groups.append((cur, cur_reason))
        return groups

    # ---------------------------------------------------------- main
    def chunk_document(
        self, doc: ProcedureDocument, figure_ocr: dict[str, dict]
    ) -> tuple[list[Chunk], ProcedureRecord, list[FigureRecord]]:
        key = self._procedure_key(doc)
        parent_id = f"proc_{key}"
        proc_app = md.procedure_applicability(self.model_years, doc.toc_path, doc.section_name, doc.title, doc.doc_type)
        sections = build_sections(doc)
        for s in sections:
            self._measure(s.units)

        all_units = [u for s in sections for u in s.units]
        total_words = sum(len(u.text.split()) for u in all_units)
        total_tokens = sum(u.tokens for u in all_units)

        plans: list[tuple[str | None, list[Unit], str, bool, list[Element]]] = []
        if total_words <= self.cfg.heading_split_word_threshold and total_tokens <= self.cfg.max_tokens:
            plans.append((sections[0].heading if len(sections) == 1 else None, all_units, "whole_procedure", False, []))
        else:
            merged: list[Section] = []
            for s in sections:
                if (
                    merged
                    and merged[-1].tokens() < SMALL_SECTION_TOKENS
                    and merged[-1].tokens() + s.tokens() <= self.cfg.max_tokens
                ):
                    prev = merged[-1]
                    head_unit = (
                        Unit(
                            "text",
                            [
                                Element(
                                    kind=ElementKind.HEADING,
                                    text=s.heading,
                                    pdf_page_index=s.units[0].elements[0].pdf_page_index,
                                )
                            ],
                        )
                        if s.heading
                        else None
                    )
                    if head_unit:
                        self._measure([head_unit])
                    prev.units.extend(([head_unit] if head_unit else []) + s.units)
                    prev.heading = prev.heading or s.heading
                else:
                    merged.append(Section(heading=s.heading, units=list(s.units)))
            header_tokens = self.counter.count(self._context_header(doc, "x" * 12, True)) + 8
            for s in merged:
                warnings = self._preamble_warnings(s)
                overhead = header_tokens + self.counter.count(" ".join(w.text for w in warnings))
                if s.tokens() + header_tokens <= self.cfg.max_tokens:
                    plans.append((s.heading, s.units, "heading", False, warnings))
                    continue
                for i, (units, reason) in enumerate(self._pack(s.units, overhead)):
                    plans.append((s.heading, units, reason, i > 0, warnings))

        chunks: list[Chunk] = []
        figure_records: list[FigureRecord] = []
        printed_total = doc.printed_total
        for seq, (heading, units, reason, continued, warnings) in enumerate(plans):
            body_parts = []
            qual = units[0].qualifier if units and units[0].kind != "qualifier" else None
            if continued and warnings:
                body_parts.append("(Repeated from the start of this section)\n" + "\n".join(w.text for w in warnings))
            if continued and qual:
                body_parts.append(f"[{qual}]")
            if heading and reason != "whole_procedure" and not continued:
                body_parts.append(f"### {heading}")
            body_parts.extend(u.text for u in units if u.text)
            content = "\n".join(body_parts).strip()
            header = self._context_header(doc, heading if reason != "whole_procedure" else None, continued)
            elements = [e for u in units for e in u.elements]
            pages = sorted({e.pdf_page_index for e in elements}) or [doc.pdf_page_start]
            quals = list(dict.fromkeys([u.qualifier for u in units if u.qualifier] + ([qual] if qual else [])))
            app = md.merge(proc_app, *(md.from_qualifier(q) for q in quals))
            steps = [n for u in units for n in u.step_numbers]
            chunk_id = f"chk_{key}_{seq:03d}"
            fig_ids, ocr_snips = [], []
            for u in units:
                for e in u.elements:
                    if e.kind != ElementKind.FIGURE or e.figure is None:
                        continue
                    f = e.figure
                    fig_ids.append(f.figure_id)
                    ocr = figure_ocr.get(f.asset_id, {})
                    if ocr.get("text") and (ocr.get("confidence") or 0) >= 50:
                        ocr_snips.append(ocr["text"][:400])
                    method, conf = self._association(u, f.image_type, heading)
                    figure_records.append(
                        FigureRecord(
                            figure_id=f.figure_id,
                            asset_id=f.asset_id,
                            document_id=self.document_id,
                            source_hash=self.source_hash,
                            pdf_page_index=f.pdf_page_index,
                            page_number=f.pdf_page_index + 1,
                            printed_page=self._printed(doc, f.pdf_page_index),
                            bbox=f.bbox,
                            image_type=f.image_type,
                            extraction_method=f.extraction_method,
                            procedure_id=key,
                            source_procedure_id=doc.procedure_id,
                            procedure_title=doc.title,
                            section_code=doc.section_code,
                            related_chunk_ids=[chunk_id],
                            nearest_step=(u.step_numbers or [None])[0],
                            subsection=heading,
                            association_method=method,
                            association_confidence=conf,
                            figure_numbers=ocr.get("figure_numbers", []),
                            ocr_text=ocr.get("text", ""),
                            ocr_confidence=ocr.get("confidence"),
                            width_px=f.width_px,
                            height_px=f.height_px,
                            asset_relpath=f.asset_relpath,
                        )
                    )
            full = f"{header}\n{content}"
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=self.document_id,
                document_title=self.document_title,
                source_hash=self.source_hash,
                toc_path=doc.toc_path,
                system_group=doc.toc_path[0] if doc.toc_path else None,
                section_code=doc.section_code,
                section_name=doc.section_name,
                parent_id=parent_id,
                procedure_id=key,
                source_procedure_id=doc.procedure_id,
                procedure_title=doc.title,
                doc_type=doc.doc_type,
                subsection=heading,
                chunk_type=classify_chunk(doc, heading, content, units),
                sequence_index=seq,
                is_continuation=continued,
                split_reason=reason,
                step_range=(min(steps), max(steps)) if steps else None,
                page_start=pages[0] + 1,
                page_end=pages[-1] + 1,
                pdf_page_start=pages[0],
                pdf_page_end=pages[-1],
                printed_page_start=self._printed(doc, pages[0], printed_total),
                printed_page_end=self._printed(doc, pages[-1], printed_total),
                figure_ids=fig_ids,
                table_ids=[e.table.table_id for u in units for e in u.elements if e.table is not None],
                applicability=app,
                flags=content_flags(content, units),
                context_header=header,
                content=content,
                figure_ocr=" | ".join(ocr_snips)[:1500],
                content_hash=hashlib.sha256(full.encode()).hexdigest()[:16],
                token_count=self.counter.count(full),
            )
            chunks.append(chunk)

        record = ProcedureRecord(
            parent_id=parent_id,
            procedure_id=key,
            source_procedure_id=doc.procedure_id,
            title=doc.title,
            section_code=doc.section_code,
            section_name=doc.section_name,
            doc_type=doc.doc_type,
            toc_path=doc.toc_path,
            page_start=doc.pdf_page_start + 1,
            page_end=doc.pdf_page_end + 1,
            printed_total=doc.printed_total,
            revision_date=doc.revision_date,
            chunk_ids=[c.chunk_id for c in chunks],
            applicability=md.merge(proc_app, *(c.applicability for c in chunks)),
            subsections=[s.heading for s in sections if s.heading],
        )
        return chunks, record, figure_records

    @staticmethod
    def _printed(doc: ProcedureDocument, pdf_index: int, total: int | None = None) -> str | None:
        total = total or doc.printed_total
        if total is None:
            return None
        n = doc.printed_pages.get(pdf_index) or doc.printed_pages.get(str(pdf_index))  # type: ignore[call-overload]
        if n is None:
            n = pdf_index - doc.pdf_page_start + 1
        return f"Page {n} of {total}" if 1 <= n <= total else None

    @staticmethod
    def _association(unit: Unit, image_type: str, heading: str | None) -> tuple[str, float]:
        if image_type == "tool_icon":
            return "special_tools", 0.6
        if unit.kind == "step":
            return "in_step", 0.9
        if unit.kind == "table":
            return "legend_table", 0.85
        return "section_preamble", 0.7


def iter_chunks(chunker: HierarchicalChunker, docs: Iterator[ProcedureDocument], figure_ocr: dict[str, dict]):
    for doc in docs:
        yield chunker.chunk_document(doc, figure_ocr)


__all__ = ["Applicability", "ChunkingConfig", "HierarchicalChunker", "build_sections", "classify_chunk", "iter_chunks"]
