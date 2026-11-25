"""Chunk, procedure-parent and figure records produced by hierarchical chunking."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Applicability(BaseModel):
    """Vehicle applicability. Empty lists mean "not variant-specific / unknown", never "all confirmed"."""

    model_years: list[int] = Field(default_factory=list)
    engines: list[str] = Field(default_factory=list)  # normalised, e.g. "5.0L"
    transmissions: list[str] = Field(default_factory=list)  # "automatic" | "manual" | codes e.g. "6R80"
    body_styles: list[str] = Field(default_factory=list)  # "coupe" | "convertible"
    variants: list[str] = Field(default_factory=list)  # e.g. "GT500"
    equipment_conditions: list[str] = Field(default_factory=list)  # raw qualifier text not normalised
    exclusions: list[str] = Field(default_factory=list)  # "Vehicles without ..." text
    sources: list[str] = Field(default_factory=list)  # document | section | title | qualifier
    raw_qualifiers: list[str] = Field(default_factory=list)

    def is_engine_specific(self) -> bool:
        return bool(self.engines)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    source_hash: str
    manual_revision: str | None = None
    toc_path: list[str] = Field(default_factory=list)
    system_group: str | None = None  # bookmark level-1/2 label, e.g. "2. Chassis"
    section_code: str | None = None
    section_name: str | None = None
    parent_id: str  # procedure parent record id
    procedure_id: str  # unique procedure key (source id, de-duplicated)
    source_procedure_id: str  # id exactly as printed in the source (e.g. SEB41004)
    procedure_title: str | None = None
    doc_type: str | None = None
    subsection: str | None = None  # e.g. "Removal", "Installation", "PINPOINT TEST A: ..."
    chunk_type: str
    sequence_index: int
    is_continuation: bool = False
    split_reason: str  # whole_procedure | heading | step_group | table_rows | token_fallback
    step_range: tuple[int, int] | None = None
    page_start: int  # 1-based PDF page
    page_end: int
    pdf_page_start: int  # 0-based PDF index
    pdf_page_end: int
    printed_page_start: str | None = None  # "Page 3 of 9" (procedure-local, as printed)
    printed_page_end: str | None = None
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    applicability: Applicability = Field(default_factory=Applicability)
    flags: list[str] = Field(default_factory=list)  # has_warning | has_torque | has_table | has_dtc ...
    context_header: str
    content: str
    figure_ocr: str = ""  # OCR text from associated illustrations (search aid, labelled as OCR)
    content_hash: str
    token_count: int


class ProcedureRecord(BaseModel):
    """Parent record used for context expansion and citation."""

    parent_id: str
    procedure_id: str
    source_procedure_id: str
    title: str | None = None
    section_code: str | None = None
    section_name: str | None = None
    doc_type: str | None = None
    toc_path: list[str]
    page_start: int
    page_end: int
    printed_total: int | None = None
    revision_date: str | None = None
    chunk_ids: list[str]
    applicability: Applicability
    subsections: list[str] = Field(default_factory=list)


class FigureRecord(BaseModel):
    figure_id: str
    asset_id: str
    document_id: str
    source_hash: str
    pdf_page_index: int
    page_number: int  # 1-based
    printed_page: str | None = None
    bbox: tuple[float, float, float, float]
    image_type: str
    extraction_method: str
    procedure_id: str
    source_procedure_id: str
    procedure_title: str | None = None
    section_code: str | None = None
    related_chunk_ids: list[str] = Field(default_factory=list)
    nearest_step: int | None = None
    subsection: str | None = None
    association_method: str  # in_step | section_preamble | legend_table | special_tools | page_proximity
    association_confidence: float
    figure_numbers: list[str] = Field(default_factory=list)  # OCR-derived figure ids printed inside image
    ocr_text: str = ""
    ocr_confidence: float | None = None
    width_px: int
    height_px: int
    asset_relpath: str
