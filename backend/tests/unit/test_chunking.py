from __future__ import annotations

import pymupdf
import pytest

from app.embeddings.tokenizer import TokenCounter
from app.ingestion import metadata as md
from app.ingestion.chunker import ChunkingConfig, HierarchicalChunker, build_sections
from app.ingestion.layout_parser import parse_page
from app.ingestion.structure_detector import iter_procedures
from app.schemas.document import Element, ElementKind, ProcedureDocument, TableData


class WordCounter(TokenCounter):
    """Deterministic, offline token counter for unit tests (1 token per word)."""

    def __init__(self) -> None:
        self._tok = None

    def count(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def docs(synthetic_pdf, tmp_path):
    pdf = pymupdf.open(synthetic_pdf)
    pages = [parse_page(pdf, i, tmp_path, {}, ocr_enabled=False) for i in range(pdf.page_count)]
    return list(iter_procedures(pages, pdf.get_toc()))


def make_chunker(**cfg) -> HierarchicalChunker:
    return HierarchicalChunker(
        ChunkingConfig(**{"target_tokens": 550, "max_tokens": 900, "overlap_tokens": 20,
                          "heading_split_word_threshold": 400, **cfg}),
        WordCounter(), document_id="doc", document_title="Testcar 2099 Workshop Manual",
        source_hash="abc", model_years=[2099],
    )  # fmt: skip


# ---------------------------------------------------------------- applicability
@pytest.mark.parametrize(
    ("phrase", "engines", "trans", "body", "excl"),
    [
        ("Vehicles with 5.8L (4V) engine", ["5.8L"], [], [], []),
        ("3.7L vehicles", ["3.7L"], [], [], []),
        ("Vehicles with automatic transmission", [], ["automatic"], [], []),
        ("307-01 Automatic Transaxle-Transmission - 6R80", [], ["6R80", "automatic"], [], []),
        ("Quarter Trim Panel — Convertible", [], [], ["convertible"], []),
        ("Shock Absorber — GT500", ["5.8L"], [], [], []),
        ("Vehicles without tapped pinion flanges", [], [], [], ["Vehicles without tapped pinion flanges"]),
    ],
)
def test_parse_phrase(phrase, engines, trans, body, excl):
    a = md.parse_phrase(phrase)
    assert sorted(a.engines) == engines
    assert a.transmissions == trans
    assert a.body_styles == body
    assert a.exclusions == excl


def test_qualifier_keeps_unnormalised_equipment_condition():
    a = md.from_qualifier("Vehicles with 6 piston calipers")
    assert a.equipment_conditions == ["Vehicles with 6 piston calipers"]
    assert md.from_qualifier("All vehicles").equipment_conditions == []


def test_flatten_for_vector_store():
    meta = md.flatten_for_vector_store(md.parse_phrase("Vehicles with 5.0L engine"))
    assert meta["engine_any"] is False and meta["eng_5_0l"] is True and meta["trans_any"] is True


# ---------------------------------------------------------------- units & chunks
def test_units_keep_warnings_with_steps(docs):
    sections = build_sections(docs[0])
    step2 = next(u for s in sections for u in s.units if 2 in u.step_numbers)
    kinds = [e.kind for e in step2.elements]
    assert kinds.index(ElementKind.ADMONITION) < kinds.index(ElementKind.STEP)


def test_small_procedure_is_single_whole_chunk(docs):
    chunks, record, figs = make_chunker().chunk_document(docs[0], {})
    assert len(chunks) == 1 and chunks[0].split_reason == "whole_procedure"
    c = chunks[0]
    assert c.parent_id == record.parent_id == "proc_SEB99001"
    assert record.chunk_ids == [c.chunk_id]
    assert c.page_start == 1 and c.page_end == 2
    assert c.printed_page_start == "Page 1 of 2" and c.printed_page_end == "Page 2 of 2"
    assert c.applicability.model_years == [2099]
    assert "9.9L" in c.applicability.engines  # from the qualifier line
    assert {"has_warning", "has_torque", "has_table"} <= set(c.flags)
    # step order preserved
    assert c.content.index("1. Disconnect") < c.content.index("2. Remove") < c.content.index("3. Remove")


def test_heading_split_and_context_on_continuations(docs):
    chunks, _, _ = make_chunker(heading_split_word_threshold=10, max_tokens=40, target_tokens=25).chunk_document(
        docs[0], {}
    )
    assert len(chunks) > 1
    assert all(c.parent_id == "proc_SEB99001" for c in chunks)
    assert [c.sequence_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert "Widget Bracket" in c.context_header
    # the WARNING in the Removal preamble is repeated on continuation chunks of that section
    cont = [c for c in chunks if c.is_continuation and c.subsection == "Removal"]
    for c in cont:
        assert "WARNING: Support the widget" in c.content


def test_figure_association(docs):
    chunks, _, figs = make_chunker().chunk_document(docs[0], {"x": {}})
    assert len(figs) == 1
    f = figs[0]
    assert f.related_chunk_ids == [chunks[0].chunk_id]
    assert f.association_method == "in_step" and f.nearest_step == 3
    assert f.figure_id in chunks[0].figure_ids


def test_table_split_repeats_header():
    rows = [[f"Bolt {i}", str(i), str(i + 1)] for i in range(60)]
    t = TableData(
        table_id="t1",
        header=["Description", "Nm", "lb-ft"],
        rows=rows,
        bbox=(0, 0, 1, 1),
        pdf_page_index=0,
        method="lines",
    )
    doc = ProcedureDocument(
        procedure_id="SEBX", title="Specs", doc_type="SPECIFICATIONS", pdf_page_start=0, pdf_page_end=0,
        elements=[
            Element(kind=ElementKind.HEADING, text="Torque Specifications", pdf_page_index=0),
            Element(kind=ElementKind.TABLE, text=t.to_text(), table=t, pdf_page_index=0),
        ],
    )  # fmt: skip
    chunks, _, _ = make_chunker(target_tokens=60, max_tokens=100, heading_split_word_threshold=10).chunk_document(
        doc, {}
    )
    assert len(chunks) > 1
    assert all(c.split_reason == "table_rows" for c in chunks)
    assert all(c.content.count("Description | Nm | lb-ft") == 1 for c in chunks)
    all_rows = "\n".join(c.content for c in chunks)
    for r in rows:  # no row lost, none broken
        assert " | ".join(r) in all_rows
    assert all(c.chunk_type == "torque_table" for c in chunks)


def test_duplicate_source_ids_get_unique_keys(docs):
    ch = make_chunker()
    a, _, _ = ch.chunk_document(docs[0], {})
    b, _, _ = ch.chunk_document(docs[0], {})
    assert a[0].procedure_id == "SEB99001" and b[0].procedure_id == "SEB99001_2"
    assert a[0].source_procedure_id == b[0].source_procedure_id == "SEB99001"


def test_token_fallback_for_oversize_step():
    long_text = " ".join(f"word{i}" for i in range(300))
    doc = ProcedureDocument(
        procedure_id="SEBY", title="Long", pdf_page_start=0, pdf_page_end=0,
        elements=[Element(kind=ElementKind.HEADING, text="Removal", pdf_page_index=0),
                  Element(kind=ElementKind.STEP, step_number=1, text=long_text, pdf_page_index=0)],
    )  # fmt: skip
    chunks, _, _ = make_chunker(target_tokens=100, max_tokens=150, heading_split_word_threshold=10).chunk_document(
        doc, {}
    )
    assert len(chunks) >= 3
    assert all(c.split_reason == "token_fallback" for c in chunks)
