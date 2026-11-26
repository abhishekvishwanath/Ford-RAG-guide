"""Retrieval request/evidence and answer/citation schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query_text: str  # semantic query
    keyword_query: str  # BM25 query (exact identifiers + synonyms)
    original_message: str
    model_year: int | None = None
    engine: str | None = None
    transmission: str | None = None
    body_style: str | None = None
    task_type: str | None = None
    component: str | None = None
    manual_terms: list[str] = Field(default_factory=list)
    dtcs: list[str] = Field(default_factory=list)
    focus_parent_id: str | None = None  # follow-up: prefer the active procedure
    focus_steps: list[int] = Field(default_factory=list)  # follow-up "step 4"
    preferred_chunk_types: list[str] = Field(default_factory=list)
    task_text: str = ""  # the technician's wording of the job, when this turn only answered a question


class Candidate(BaseModel):
    chunk_id: str
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None  # final ordering score (reranker or normalised RRF, plus boosts)
    rerank_raw: float | None = None  # cross-encoder relevance (0..1) before boosts
    applicability: str = "matches"  # matches | general | uncertain


class EvidenceItem(BaseModel):
    source_id: str  # S1, S2 ... used for citation markers
    chunk_id: str
    parent_id: str
    procedure_title: str | None
    section: str | None
    doc_type: str | None
    subsection: str | None
    page_start: int
    page_end: int
    printed_page_start: str | None
    printed_page_end: str | None
    applicability_note: str
    content: str
    figure_ocr: str = ""
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    expanded: bool = False  # added by parent/procedure expansion
    score: float = 0.0


class ImageRef(BaseModel):
    asset_id: str
    figure_id: str
    url: str
    caption: str
    figure_numbers: list[str] = Field(default_factory=list)
    page_number: int
    printed_page: str | None = None
    procedure_title: str | None = None
    section: str | None = None
    nearest_step: int | None = None
    association: str
    confidence: float
    source_id: str | None = None


class TableRef(BaseModel):
    table_id: str
    source_id: str
    header: list[str]
    rows: list[list[str]]
    page_number: int


class EvidencePackage(BaseModel):
    request: RetrievalRequest
    items: list[EvidenceItem] = Field(default_factory=list)
    images: list[ImageRef] = Field(default_factory=list)
    tables: list[TableRef] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    sufficient: bool = False
    insufficiency_reason: str | None = None
    variant_ambiguity: list[str] = Field(default_factory=list)  # procedure titles differing only by variant
    variant_parent_ids: list[str] = Field(default_factory=list)  # parallel to variant_ambiguity
    token_estimate: int = 0


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    document_title: str
    section: str | None
    procedure_title: str | None
    procedure_id: str
    subsection: str | None
    pdf_pages: str  # "77-78" (1-based PDF page numbers)
    printed_pages: str | None  # "Page 2 of 9 - Page 3 of 9" (procedure-local, as printed)
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    text: str
    cited_source_ids: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    unsupported_values: list[str] = Field(default_factory=list)
    grounded: bool = True
    model: str | None = None
