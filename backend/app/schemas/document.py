"""Normalized intermediate document representation produced by the parser.

PDF page indices are 0-based (``pdf_page_index``); everything shown to users converts
to 1-based page numbers. ``printed_page`` is the procedure-local "Page X of N" value.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]


class ElementKind(StrEnum):
    TITLE = "title"
    SECTION_HEADER = "section_header"
    DOC_TYPE = "doc_type"
    REVISION = "revision"
    HEADING = "heading"
    QUALIFIER = "qualifier"  # vehicle applicability line, e.g. "Vehicles with 5.8L (4V) engine"
    STEP = "step"
    LIST_ITEM = "list_item"
    ADMONITION = "admonition"  # WARNING / CAUTION / NOTICE / NOTE
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"


class TableData(BaseModel):
    table_id: str
    header: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    bbox: BBox
    pdf_page_index: int
    method: str  # lines | text_header
    continued_from: str | None = None  # table_id of the table this continues (cross-page)
    leading_fragment: list[str] | None = None  # cell text of a row cut by the previous page break

    def to_text(self) -> str:
        lines = []
        if self.header:
            lines.append(" | ".join(self.header))
        lines.extend(" | ".join(r) for r in self.rows)
        return "\n".join(lines)


class FigureRef(BaseModel):
    figure_id: str
    asset_id: str  # content-hash id of the stored image file (deduplicated)
    pdf_page_index: int
    bbox: BBox
    width_px: int
    height_px: int
    image_type: str  # illustration | tool_icon
    extraction_method: str  # embedded_image | page_region_render
    asset_relpath: str  # relative to images_dir; never exposed to clients


class Element(BaseModel):
    kind: ElementKind
    text: str = ""
    pdf_page_index: int
    bbox: BBox | None = None
    step_number: int | None = None
    marker: str | None = None  # bullet / sub-step letter
    admonition_type: str | None = None
    heading_level: int | None = None
    bold: bool = False
    table: TableData | None = None
    figure: FigureRef | None = None


class PageDiagnostics(BaseModel):
    char_count: int = 0
    n_paragraphs: int = 0
    n_tables: int = 0
    n_figures: int = 0
    n_banners_removed: int = 0
    ocr_used: bool = False
    ocr_confidence: float | None = None
    parse_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class PageRecord(BaseModel):
    pdf_page_index: int
    seb_id: str | None = None
    printed_page: int | None = None
    printed_total: int | None = None
    manual_label: str | None = None  # e.g. "2014 Mustang Workshop Manual"
    print_date: str | None = None
    text_source: str = "text_layer"  # text_layer | ocr | none
    elements: list[Element] = Field(default_factory=list)
    diagnostics: PageDiagnostics = Field(default_factory=PageDiagnostics)


class ProcedureDocument(BaseModel):
    """One source procedure document (a Ford SEB file), possibly spanning many pages."""

    procedure_id: str  # SEB id as printed in the source footer
    title: str | None = None
    section_code: str | None = None  # e.g. "204-01"
    section_name: str | None = None  # e.g. "Front Suspension"
    doc_type: str | None = None  # e.g. "REMOVAL AND INSTALLATION"
    revision_date: str | None = None
    manual_label: str | None = None
    pdf_page_start: int
    pdf_page_end: int
    printed_total: int | None = None
    printed_pages: dict[int, int] = Field(default_factory=dict)  # pdf index -> printed "Page X"
    toc_path: list[str] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    document_id: str
    file_name: str
    source_hash: str
    page_count: int
    manual_labels: dict[str, int] = Field(default_factory=dict)
    model_years: list[int] = Field(default_factory=list)
    model_name: str | None = None
    print_dates: list[str] = Field(default_factory=list)
    toc_entries: int = 0
