"""Single-page layout parsing: PDF page -> ``PageRecord`` with classified elements.

The classifier relies on typography (size, bold, italic), position and generic
service-manual conventions (numbered steps, WARNING/CAUTION/NOTICE/NOTE labels,
vehicle-applicability phrasing). No procedure-specific headings or page numbers
are hardcoded.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app.ingestion import boilerplate as bp
from app.ingestion.ocr_service import needs_page_ocr, ocr_image, tesseract_available
from app.ingestion.table_extractor import _Line, extract_tables
from app.ingestion.visual_extractor import extract_figures
from app.schemas.document import Element, ElementKind, PageDiagnostics, PageRecord

STEP_RE = re.compile(r"^(\d{1,3})\.(?=\s|$)\s*(.*)$", re.S)  # "5.8L" is not a step
SUBSTEP_RE = re.compile(r"^([a-z])\.\s+(.*)$", re.S)
ADMONITION_RE = re.compile(r"^(WARNING|CAUTION|NOTICE|NOTE|IMPORTANT|DANGER)\s*:\s*(.*)$", re.S | re.I)
QUALIFIER_RE = re.compile(
    r"^(?:all vehicles\b.*|vehicles?\b.*|.*\bvehicles?$|.*\b(?:engines?|transmissions?)$"
    r"|(?:\d\.\dL|gt500|shelby|coupe|convertible)\b.*)$",
    re.I,
)
ANSWER_LABELS = frozenset({"yes", "no"})
BULLET_FONTS = ("wingdings", "symbol", "zapfdingbats")
PARA_GAP = 4.0  # vertical gap (pt) that starts a new paragraph
SAME_LINE_TOL = 2.5
MAX_INLINE_GAP = 30.0  # wider gaps on one baseline are separate columns
CELL_GAP = 10.0  # span gap that separates table cells inside one PDF line
WATERMARK_MAX_SIZE = 5.0


@dataclass
class PLine:
    y0: float
    y1: float
    x0: float
    x1: float
    text: str
    size: float
    bold: bool
    italic: bool
    bullet: bool = False
    fonts: set[str] = field(default_factory=set)


@dataclass
class Paragraph:
    lines: list[PLine]

    @property
    def text(self) -> str:
        return " ".join(ln.text for ln in self.lines).strip()

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            min(ln.x0 for ln in self.lines),
            min(ln.y0 for ln in self.lines),
            max(ln.x1 for ln in self.lines),
            max(ln.y1 for ln in self.lines),
        )

    @property
    def all_bold(self) -> bool:
        return all(ln.bold for ln in self.lines)


def _span_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _span_italic(span: dict) -> bool:
    font = span["font"].lower()
    return bool(span["flags"] & 2) or "italic" in font or "oblique" in font


def _raw_lines(page: pymupdf.Page) -> list[PLine]:
    out: list[PLine] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for spans in _split_by_baseline([s for s in line["spans"] if s["text"].strip()]):
                pl = _make_line(spans)
                if pl:
                    out.append(pl)
    return out


def _split_by_baseline(spans: list[dict]) -> list[list[dict]]:
    """PDF 'lines' can hold spans from different visual lines or table cells; split them.

    Wide horizontal gaps separate table cells; ordinary text is re-joined later by
    ``_merge_same_baseline`` (gap < MAX_INLINE_GAP)."""
    groups: list[list[dict]] = []
    for s in spans:
        prev = groups[-1][-1] if groups else None
        same_baseline = prev is not None and abs(prev["origin"][1] - s["origin"][1]) < SAME_LINE_TOL
        cell_gap = prev is not None and s["bbox"][0] - prev["bbox"][2] > CELL_GAP
        if same_baseline and not cell_gap:
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


def _make_line(spans: list[dict]) -> PLine | None:
    if not spans or max(s["size"] for s in spans) <= WATERMARK_MAX_SIZE:
        return None  # empty, or tiny watermark text
    first = spans[0]["text"].strip()
    bullet = any(f in spans[0]["font"].lower() for f in BULLET_FONTS) or (
        len(first) == 1 and "\uf000" <= first <= "\uf0ff"
    )
    text_spans = spans[1:] if bullet and len(first) <= 2 and len(spans) > 1 else spans
    text = "".join(s["text"] for s in text_spans)
    if bullet and text_spans is spans:  # glyph fused into the text span
        text = text.lstrip()[1:]
    text = " ".join(bp.strip_emails(text).split())
    if not text:
        return None
    return PLine(
        y0=min(s["bbox"][1] for s in spans),
        y1=max(s["bbox"][3] for s in spans),
        x0=min(s["bbox"][0] for s in spans),
        x1=max(s["bbox"][2] for s in spans),
        text=text,
        size=max(s["size"] for s in text_spans),
        bold=all(_span_bold(s) for s in text_spans),
        italic=all(_span_italic(s) for s in text_spans),
        bullet=bullet,
        fonts={s["font"] for s in text_spans},
    )


def _merge_same_baseline(lines: list[PLine]) -> list[PLine]:
    lines = sorted(lines, key=lambda ln: (round(ln.y0), ln.x0))
    merged: list[PLine] = []
    for ln in lines:
        prev = merged[-1] if merged else None
        number_only = prev is not None and re.fullmatch(r"\d{1,3}\.|[a-z]\.", prev.text) is not None
        close = prev is not None and (ln.x0 - prev.x1 < MAX_INLINE_GAP or number_only)
        if prev and close and abs(prev.y0 - ln.y0) < SAME_LINE_TOL and ln.x0 >= prev.x1 - 1:
            prev.text = f"{prev.text} {ln.text}"
            prev.x1 = max(prev.x1, ln.x1)
            prev.y1 = max(prev.y1, ln.y1)
            # a step number followed by text takes the text's style
            if number_only:
                prev.bold, prev.italic = ln.bold, ln.italic
            else:
                prev.bold = prev.bold and ln.bold
                prev.italic = prev.italic and ln.italic
            prev.size = max(prev.size, ln.size)
            prev.bullet = prev.bullet or ln.bullet
            prev.fonts |= ln.fonts
        else:
            merged.append(PLine(**{**ln.__dict__, "fonts": set(ln.fonts)}))
    return merged


def _is_header_line(ln: PLine) -> bool:
    """Small italic lines form the per-procedure header block (section / manual / doc type)."""
    return ln.italic and not ln.bold and ln.size <= 9.5


def _starts_block(ln: PLine) -> bool:
    return bool(ln.bullet or STEP_RE.match(ln.text) or ADMONITION_RE.match(ln.text) or SUBSTEP_RE.match(ln.text))


def _group_paragraphs(lines: list[PLine]) -> list[Paragraph]:
    paras: list[Paragraph] = []
    for ln in lines:
        cur = paras[-1] if paras else None
        if cur is None:
            paras.append(Paragraph([ln]))
            continue
        last = cur.lines[-1]
        first = cur.lines[0]
        gap = ln.y0 - last.y1
        cur_adm = ADMONITION_RE.match(first.text)
        style_break = False
        if cur_adm and cur_adm.group(1).upper() != "NOTE":
            style_break = last.bold != ln.bold  # bold admonition body ended
        elif not cur_adm and not STEP_RE.match(first.text) and not first.bullet:
            style_break = last.bold != ln.bold or last.italic != ln.italic
        header_like = _is_header_line(ln) or _is_header_line(last)
        if gap > PARA_GAP or _starts_block(ln) or abs(ln.size - last.size) > 1.5 or style_break or header_like:
            paras.append(Paragraph([ln]))
        else:
            cur.lines.append(ln)
    return paras


def _classify(para: Paragraph, pdf_page_index: int, is_first_page: bool, seen_title: bool) -> list[Element]:
    """Classify a paragraph. A step that opens with an admonition becomes [step marker, admonition]."""
    el = _classify_one(para, pdf_page_index, is_first_page, seen_title)
    if el.kind == ElementKind.STEP and (m := ADMONITION_RE.match(el.text)):
        adm = el.model_copy(
            update={"kind": ElementKind.ADMONITION, "admonition_type": m.group(1).upper(), "step_number": None}
        )
        return [el.model_copy(update={"text": ""}), adm]
    return [el]


def _classify_one(para: Paragraph, pdf_page_index: int, is_first_page: bool, seen_title: bool) -> Element:
    text = para.text
    first = para.lines[0]
    bbox = para.bbox
    base = {"pdf_page_index": pdf_page_index, "bbox": bbox, "bold": para.all_bold}

    if first.italic and first.size <= 9.5 and not first.bold:
        if m := bp.SECTION_HEADER_RE.match(text):
            return Element(kind=ElementKind.SECTION_HEADER, text=f"{m.group(1)}: {m.group(2).strip()}", **base)
        if text.isupper() or text.replace(" - ", " ").replace("-", " ").isupper():
            return Element(kind=ElementKind.DOC_TYPE, text=text, **base)
    if bp.REVISION_RE.match(text):
        return Element(kind=ElementKind.REVISION, text=text, **base)
    if is_first_page and not seen_title and para.all_bold and first.size >= 11.5:
        return Element(kind=ElementKind.TITLE, text=text, heading_level=1, **base)
    if m := ADMONITION_RE.match(text):
        return Element(kind=ElementKind.ADMONITION, text=text, admonition_type=m.group(1).upper(), **base)
    if m := STEP_RE.match(text):
        return Element(kind=ElementKind.STEP, text=m.group(2).strip(), step_number=int(m.group(1)), **base)
    if first.bullet:
        return Element(kind=ElementKind.LIST_ITEM, text=text, marker="•", **base)
    if m := SUBSTEP_RE.match(text):
        return Element(kind=ElementKind.LIST_ITEM, text=m.group(2).strip(), marker=m.group(1), **base)
    words = len(text.split())
    if text.rstrip(".:").lower() in ANSWER_LABELS:
        return Element(kind=ElementKind.PARAGRAPH, text=text, **base)  # diagnostic result label, not a section
    if para.all_bold and words <= 15 and not text.endswith("."):
        if QUALIFIER_RE.match(text) and words <= 12:
            return Element(kind=ElementKind.QUALIFIER, text=text, **base)
        return Element(kind=ElementKind.HEADING, text=text, heading_level=2 if first.size >= 11.5 else 3, **base)
    return Element(kind=ElementKind.PARAGRAPH, text=text, **base)


def _footer_info(lines: list[PLine], page_h: float, record: PageRecord) -> list[PLine]:
    """Extract footer facts into the record and return non-boilerplate lines."""
    keep: list[PLine] = []
    footer_text: list[str] = []
    for ln in lines:
        t = ln.text.strip()
        if m := bp.PAGE_OF_RE.match(t):
            record.printed_page, record.printed_total = int(m.group(1)), int(m.group(2))
            continue
        if m := bp.MANUAL_LABEL_RE.match(t):
            if not (ln.italic and ln.size <= 9.5):  # italic copy belongs to the header block
                record.manual_label = m.group(1)
                continue
            record.manual_label = record.manual_label or m.group(1)
            continue
        if bp.PRINT_DATE_RE.match(t):
            record.print_date = t
            continue
        if bp.SOURCE_URL_RE.match(t) or (ln.y0 > 0.9 * page_h and "HTM" in t.upper()):
            footer_text.append(t)
            continue
        keep.append(ln)
    if m := bp.SEB_ID_RE.search("".join(footer_text)):
        record.seb_id = m.group(1).upper()
    return keep


def parse_page(
    doc: pymupdf.Document,
    pdf_page_index: int,
    images_dir: Path,
    xref_cache: dict,
    *,
    ocr_enabled: bool = True,
    ocr_min_chars: int = 40,
) -> PageRecord:
    t0 = time.perf_counter()
    page = doc[pdf_page_index]
    record = PageRecord(pdf_page_index=pdf_page_index)
    diag = PageDiagnostics()
    raw = _raw_lines(page)
    diag.char_count = sum(len(ln.text) for ln in raw)
    lines = _footer_info(raw, page.rect.height, record)
    if record.seb_id is None:
        diag.warnings.append("no_procedure_id_in_footer")

    elements: list[Element] = []
    if ocr_enabled and needs_page_ocr(diag.char_count, ocr_min_chars) and tesseract_available():
        pix = page.get_pixmap(dpi=300)
        from PIL import Image

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        res = ocr_image(img, psm=3)
        if res.text.strip():
            record.text_source = "ocr"
            diag.ocr_used, diag.ocr_confidence = True, res.confidence
            elements.append(Element(kind=ElementKind.PARAGRAPH, text=res.text, pdf_page_index=pdf_page_index))
        else:
            record.text_source = "none"

    tlines = [_Line(ln.y0, ln.y1, ln.x0, ln.x1, ln.text, ln.bold) for ln in lines]
    tables = extract_tables(page, pdf_page_index, tlines)
    table_rects = [pymupdf.Rect(t.bbox) for t in tables]

    def in_table(ln: PLine) -> bool:
        c = pymupdf.Point((ln.x0 + ln.x1) / 2, (ln.y0 + ln.y1) / 2)
        return any(r.contains(c) for r in table_rects)

    body = _merge_same_baseline([ln for ln in lines if not in_table(ln)])
    paragraphs = _group_paragraphs(body)
    diag.n_paragraphs = len(paragraphs)
    is_first = record.printed_page == 1
    seen_title = False
    for para in paragraphs:
        for el in _classify(para, pdf_page_index, is_first, seen_title):
            seen_title = seen_title or el.kind == ElementKind.TITLE
            elements.append(el)

    for t in tables:
        elements.append(
            Element(kind=ElementKind.TABLE, text=t.to_text(), pdf_page_index=pdf_page_index, bbox=t.bbox, table=t)
        )
    figures, banners = extract_figures(doc, page, pdf_page_index, images_dir, xref_cache)
    for f in figures:
        elements.append(Element(kind=ElementKind.FIGURE, pdf_page_index=pdf_page_index, bbox=f.bbox, figure=f))

    elements.sort(key=lambda e: (e.bbox[1] if e.bbox else 0.0, e.bbox[0] if e.bbox else 0.0))
    record.elements = elements
    diag.n_tables, diag.n_figures, diag.n_banners_removed = len(tables), len(figures), banners
    if record.text_source == "text_layer" and not any(e.kind != ElementKind.FIGURE for e in elements):
        diag.warnings.append("no_body_text")
    diag.parse_ms = round((time.perf_counter() - t0) * 1000, 1)
    record.diagnostics = diag
    return record
