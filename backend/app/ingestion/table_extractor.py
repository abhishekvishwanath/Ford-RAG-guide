"""Table reconstruction from PDF page geometry.

Strategy:
1. Ruled tables: ``page.find_tables()`` (vector line strategy) gives cell geometry.
2. Un-ruled tables: only when a bold multi-cell header row is present, run the text
   strategy clipped below that header (prevents prose being mistaken for tables).
3. Cell text is re-extracted with ``get_text(clip=cell)`` because the built-in
   extractor drops inter-word spaces on this kind of print.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pymupdf

from app.ingestion.boilerplate import strip_emails
from app.schemas.document import BBox, TableData

logger = logging.getLogger(__name__)


@dataclass
class _Line:
    y0: float
    y1: float
    x0: float
    x1: float
    text: str
    bold: bool


HEADER_CENTER_TOL = 16.0  # header labels may be centred over left-aligned data
PRIVATE_USE_GLYPH = re.compile(r"[\uf000-\uf0ff]")


def _clean(text: str) -> str:
    """Normalise whitespace; bullet glyphs from symbol fonts become a plain bullet."""
    return " ".join(PRIVATE_USE_GLYPH.sub(" • ", strip_emails(text)).split())


def _cell_text(page: pymupdf.Page, cell: tuple[float, float, float, float] | None) -> str:
    if cell is None:
        return ""
    return _clean(page.get_text("text", clip=pymupdf.Rect(cell)))


def _rows_from_table(page: pymupdf.Page, table) -> list[list[str]]:  # type: ignore[no-untyped-def]
    rows: list[list[str]] = []
    for row in table.rows:
        rows.append([_cell_text(page, c) for c in row.cells])
    return rows


def _merge_continuation_rows(rows: list[list[str]]) -> list[list[str]]:
    """In text-strategy tables, wrapped cell text shows up as rows with an empty first cell."""
    merged: list[list[str]] = []
    for r in rows:
        if not any(r):
            continue
        if merged and not r[0] and len(r) == len(merged[-1]):
            merged[-1] = [(a + " " + b).strip() if b else a for a, b in zip(merged[-1], r, strict=True)]
        else:
            merged.append(r)
    return merged


def _has_internal_gap(page: pymupdf.Page, ln: _Line, gap: float = 15.0) -> bool:
    """A single bold span like 'Item     Specification' is a header row printed as one span."""
    words = sorted(page.get_text("words", clip=pymupdf.Rect(ln.x0 - 1, ln.y0 - 1, ln.x1 + 1, ln.y1 + 1)))
    return any(b[0] - a[2] > gap for a, b in zip(words, words[1:], strict=False))


def _bold_header_rows(lines: list[_Line], page: pymupdf.Page | None = None) -> list[list[_Line]]:
    """Rows where >=2 short bold cells share a baseline with a clear horizontal gap."""
    by_y: dict[int, list[_Line]] = {}
    for ln in lines:
        by_y.setdefault(round(ln.y0), []).append(ln)
    rows = []
    for _, group in sorted(by_y.items()):
        bold = sorted((g for g in group if g.bold and len(g.text.split()) <= 5), key=lambda g: g.x0)
        if (
            len(bold) >= 2
            and all(b.x0 - a.x1 > 8 for a, b in zip(bold, bold[1:], strict=False))
            or len(bold) == 1
            and len(group) == 1
            and page is not None
            and _has_internal_gap(page, bold[0])
        ):
            rows.append(bold)
    return rows


def _column_bounds(table) -> list[tuple[float, float]]:  # type: ignore[no-untyped-def]
    for row in table.rows:
        if all(c is not None for c in row.cells):
            return [(c[0], c[2]) for c in row.cells]
    return []


def _absorb_trailing_rows(
    lines: list[_Line], bbox: BBox, cols: list[tuple[float, float]], page_h: float
) -> tuple[list[str] | None, BBox]:
    """Rows cut off at a page break lose their bottom rule; pick up column-aligned lines below."""
    if not cols:
        return None, bbox
    x_min, x_max = cols[0][0] - 2, cols[-1][1] + 2
    below = sorted((ln for ln in lines if ln.y0 >= bbox[3] - 1 and ln.y1 < page_h * 0.93), key=lambda ln: ln.y0)
    picked: list[_Line] = []
    last_y = bbox[3]
    for ln in below:
        if ln.y0 - last_y > 14 or ln.x0 < x_min or ln.x1 > x_max or ln.bold:
            break
        picked.append(ln)
        last_y = max(last_y, ln.y1)
    if not picked:
        return None, bbox
    cells = [""] * len(cols)
    for ln in picked:
        idx = max((i for i, (cx0, _) in enumerate(cols) if ln.x0 >= cx0 - 3), default=0)
        cells[idx] = (cells[idx] + " " + _clean(ln.text)).strip()
    return cells, (bbox[0], bbox[1], bbox[2], last_y)


def _leading_fragment(
    lines: list[_Line], bbox: BBox, cols: list[tuple[float, float]], page_h: float
) -> tuple[list[str] | None, BBox]:
    """Lines above a header-less table, aligned to its columns, continue a row from the previous page."""
    if not cols:
        return None, bbox
    x_min, x_max = cols[0][0] - 2, cols[-1][1] + 2
    above = sorted((ln for ln in lines if ln.y1 <= bbox[1] + 1 and ln.y0 > page_h * 0.05), key=lambda ln: -ln.y0)
    picked: list[_Line] = []
    top = bbox[1]
    for ln in above:
        if top - ln.y1 > 14 or ln.x0 < x_min or ln.x1 > x_max or ln.bold:
            break
        picked.append(ln)
        top = min(top, ln.y0)
    if not picked or any(p.y0 < page_h * 0.05 for p in picked):
        return None, bbox
    cells = [""] * len(cols)
    for ln in sorted(picked, key=lambda ln: ln.y0):
        idx = max((i for i, (cx0, _) in enumerate(cols) if ln.x0 >= cx0 - 3), default=0)
        cells[idx] = (cells[idx] + " " + _clean(ln.text)).strip()
    return cells, (bbox[0], top, bbox[2], bbox[3])


def _cluster(xs: list[float], tol: float = 6.0) -> list[float]:
    out: list[float] = []
    for x in sorted(xs):
        if not out or x - out[-1] > tol:
            out.append(x)
    return out


def _rows_by_columns(
    page: pymupdf.Page, lines: list[_Line], hdr: list[_Line], page_h: float
) -> tuple[list[str], list[list[str]], float]:
    """Last-resort reconstruction of an un-ruled table below a bold header row.

    Columns come from the x-positions of the data cells (the header may be one merged span);
    header words are then assigned to those columns using word-level boxes."""
    header_ids = {id(h) for h in hdr}
    top = max(h.y1 for h in hdr)
    left = min(h.x0 for h in hdr)
    region: list[_Line] = []
    last_bottom = top
    for ln in sorted(
        (ln for ln in lines if ln.y0 >= top - 1 and id(ln) not in header_ids), key=lambda ln: (round(ln.y0), ln.x0)
    ):
        if ln.y1 >= page_h * 0.93 or ln.bold or ln.x0 < left - HEADER_CENTER_TOL or ln.y0 - last_bottom > 20:
            break
        region.append(ln)
        last_bottom = max(last_bottom, ln.y1)
    if not region:
        return [], [], top
    col_x = _cluster([ln.x0 for ln in region])
    col_x = [c for c in col_x if c >= left - HEADER_CENTER_TOL]

    def col_of(x: float) -> int:
        return max((i for i, cx in enumerate(col_x) if x >= cx - 4), default=0)

    header = [""] * len(col_x)
    for w in page.get_text(
        "words", clip=pymupdf.Rect(left - HEADER_CENTER_TOL, min(h.y0 for h in hdr) - 1, page.rect.width, top + 1)
    ):
        i = col_of(w[0])
        header[i] = (header[i] + " " + w[4]).strip()
    rows: list[list[str]] = []
    row_y: float | None = None
    for ln in region:
        idx = col_of(ln.x0)
        if row_y is None or abs(ln.y0 - row_y) > 2.5:
            row_y = ln.y0
            if idx == 0 or not rows:
                rows.append([""] * len(col_x))
        rows[-1][idx] = (rows[-1][idx] + " " + _clean(ln.text)).strip()
    return header, [r for r in rows if any(r)], last_bottom


def extract_tables(page: pymupdf.Page, pdf_page_index: int, lines: list[_Line]) -> list[TableData]:
    tables: list[TableData] = []
    taken: list[pymupdf.Rect] = []
    page_h = page.rect.height
    try:
        found = page.find_tables()
    except Exception as exc:  # pragma: no cover - defensive, pymupdf internal errors
        logger.warning("find_tables failed on page %s: %s", pdf_page_index, exc)
        found = None
    for i, t in enumerate(found.tables if found else []):
        if t.row_count < 2 or t.col_count < 2:
            continue
        rows = [r for r in _rows_from_table(page, t) if any(r)]
        if len(rows) < 2:
            continue
        bbox: BBox = tuple(t.bbox)  # type: ignore[assignment]
        extra, bbox = _absorb_trailing_rows(lines, bbox, _column_bounds(t), page_h)
        if extra and any(extra):
            rows.append(extra)
        top_bold = [ln for ln in lines if ln.bold and bbox[1] - 1 <= ln.y0 <= bbox[1] + 18]
        header, body = (rows[0], rows[1:]) if top_bold else ([], rows)
        fragment = None
        if not header:
            fragment, bbox = _leading_fragment(lines, bbox, _column_bounds(t), page_h)
        tables.append(
            TableData(
                table_id=f"tbl_p{pdf_page_index}_{i}",
                header=header,
                rows=body,
                bbox=bbox,
                pdf_page_index=pdf_page_index,
                method="lines",
                leading_fragment=fragment if fragment and any(fragment) else None,
            )
        )
        taken.append(pymupdf.Rect(bbox))

    # Un-ruled tables triggered by a bold header row outside existing tables
    for j, hdr in enumerate(_bold_header_rows(lines, page)):
        hy = min(h.y0 for h in hdr)
        if any(r.y0 - 3 <= hy <= r.y1 for r in taken):
            continue
        clip = pymupdf.Rect(0, max(h.y1 for h in hdr) + 0.5, page.rect.width, page_h * 0.93)
        header = [_clean(h.text) for h in hdr]
        try:
            tt = page.find_tables(clip=clip, strategy="text")
        except Exception:  # pragma: no cover
            tt = None
        rows: list[list[str]] = []
        bbox: BBox | None = None
        if tt and tt.tables and tt.tables[0].col_count == len(header):
            t = tt.tables[0]
            rows = [r for r in _merge_continuation_rows(_rows_from_table(page, t)) if r != header]
            bbox = (min(t.bbox[0], hdr[0].x0), hy, max(t.bbox[2], hdr[-1].x1), t.bbox[3])
        method = "text_header"
        if not rows:
            header, rows, bottom = _rows_by_columns(page, lines, hdr, page_h)
            if len(header) < 2:
                continue
            bbox = (hdr[0].x0 - 2, hy, max(ln.x1 for ln in lines if hy <= ln.y0 <= bottom), bottom)
            method = "column_assign"
        if not rows or bbox is None:
            continue
        tables.append(
            TableData(
                table_id=f"tbl_p{pdf_page_index}_t{j}",
                header=header,
                rows=rows,
                bbox=bbox,
                pdf_page_index=pdf_page_index,
                method=method,
            )
        )
        taken.append(pymupdf.Rect(bbox))
    return tables
