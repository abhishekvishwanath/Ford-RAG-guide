from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.ingestion import ocr_service
from app.ingestion.layout_parser import STEP_RE, parse_page
from app.ingestion.pdf_parser import PageParseRunner, ParseOptions
from app.ingestion.structure_detector import TocIndex, iter_procedures
from app.schemas.document import ElementKind


@pytest.fixture
def parsed(synthetic_pdf: Path, tmp_path: Path):
    doc = pymupdf.open(synthetic_pdf)
    cache: dict = {}
    return [parse_page(doc, i, tmp_path, cache, ocr_enabled=False) for i in range(doc.page_count)]


def kinds(rec):
    return [e.kind for e in rec.elements]


def test_footer_and_page_numbering(parsed):
    p0 = parsed[0]
    assert p0.seb_id == "SEB99001"
    assert (p0.printed_page, p0.printed_total) == (1, 2)
    assert p0.manual_label == "2099 Testcar Workshop Manual"
    assert parsed[2].seb_id == "SEB99002"


def test_watermark_and_boilerplate_removed(parsed):
    all_text = " ".join(e.text for r in parsed for e in r.elements)
    assert "@example.com" not in all_text
    assert "Page 1 of" not in all_text
    assert "file://" not in all_text


def test_heading_and_header_block_detection(parsed):
    els = parsed[0].elements
    title = [e for e in els if e.kind == ElementKind.TITLE]
    assert title and title[0].text == "Widget Bracket"
    assert any(e.kind == ElementKind.SECTION_HEADER and e.text.startswith("999-01") for e in els)
    assert any(e.kind == ElementKind.DOC_TYPE and e.text == "REMOVAL AND INSTALLATION" for e in els)
    assert any(e.kind == ElementKind.HEADING and e.text == "Removal" for e in els)


def test_qualifiers_steps_and_admonitions(parsed):
    els = parsed[0].elements
    quals = [e.text for e in els if e.kind == ElementKind.QUALIFIER]
    assert quals == ["All vehicles", "Vehicles with 9.9L engine"]
    steps = [e.step_number for e in els if e.kind == ElementKind.STEP]
    assert steps == [1, 2, 3]
    adm = [e for e in els if e.kind == ElementKind.ADMONITION]
    assert {a.admonition_type for a in adm} == {"WARNING", "NOTICE"}


def test_step_regex_does_not_match_displacement():
    assert STEP_RE.match("5.8L (4V) in Section 303-12") is None
    assert STEP_RE.match("12. Remove the bolt").group(1) == "12"


def test_table_row_integrity(parsed):
    tables = [e.table for e in parsed[1].elements if e.kind == ElementKind.TABLE]
    assert len(tables) == 1
    t = tables[0]
    assert t.header == ["Description", "Nm", "lb-ft"]
    assert ["Widget nut", "77", "57"] in t.rows and ["Bracket bolts", "25", "18"] in t.rows


def test_figures_extracted_and_banners_removed(parsed, tmp_path):
    figs = [e.figure for e in parsed[0].elements if e.kind == ElementKind.FIGURE]
    assert len(figs) == 1 and figs[0].image_type == "illustration"
    assert parsed[0].diagnostics.n_banners_removed == 1
    assert (tmp_path / figs[0].asset_relpath).exists()


def test_figure_dedup_by_content(synthetic_pdf, tmp_path):
    doc = pymupdf.open(synthetic_pdf)
    cache: dict = {}
    a = parse_page(doc, 0, tmp_path, cache, ocr_enabled=False)
    b = parse_page(doc, 0, tmp_path, cache, ocr_enabled=False)
    fa = [e.figure.asset_id for e in a.elements if e.figure]
    fb = [e.figure.asset_id for e in b.elements if e.figure]
    assert fa == fb
    assert len(list(tmp_path.rglob("*.png"))) == 1


def test_ocr_routing_rule():
    assert ocr_service.needs_page_ocr(0, 40) is True
    assert ocr_service.needs_page_ocr(500, 40) is False


@pytest.mark.skipif(not ocr_service.tesseract_available(), reason="tesseract not installed")
def test_figure_ocr_reads_callouts(parsed, tmp_path):
    fig = next(e.figure for e in parsed[0].elements if e.figure)
    res = ocr_service.ocr_file(tmp_path / fig.asset_relpath)
    assert "77" in res.text and "Nm" in res.text
    assert ocr_service.find_figure_numbers("see N0999999 and E146523") == ["N0999999", "E146523"]


def test_structure_groups_procedures_and_orders_warnings(parsed, synthetic_pdf):
    toc = pymupdf.open(synthetic_pdf).get_toc()
    docs = list(iter_procedures(parsed, toc))
    assert [d.procedure_id for d in docs] == ["SEB99001", "SEB99002"]
    a = docs[0]
    assert (a.section_code, a.section_name, a.doc_type) == ("999-01", "Widget System", "REMOVAL AND INSTALLATION")
    assert a.revision_date == "01/02/2098"
    assert a.pdf_page_start == 0 and a.pdf_page_end == 1
    assert a.toc_path[-1] == "Widget Bracket"
    # NOTICE inside step 2 is moved before the step and the step keeps its action text
    seq = [(e.kind, e.step_number) for e in a.elements if e.kind in (ElementKind.ADMONITION, ElementKind.STEP)]
    i = seq.index((ElementKind.STEP, 2))
    assert seq[i - 1] == (ElementKind.ADMONITION, 2)
    step2 = next(e for e in a.elements if e.kind == ElementKind.STEP and e.step_number == 2)
    assert step2.text == "Remove the widget nut."


def test_toc_index_tolerates_missing_exact_page():
    idx = TocIndex([[1, "A", 1], [2, "B", 5], [3, "C", 5], [2, "D", 9]])
    assert idx.path_for(6, None) == ["A", "B", "C"]  # pdf index 6 -> page 7 -> last entry before
    assert idx.path_for(4, "c") == ["A", "B", "C"]


def test_runner_checkpoint_resume_and_failure_reporting(synthetic_pdf, tmp_path, monkeypatch):
    opts = ParseOptions(batch_size=2, workers=1, ocr_enabled=False, retries=1)
    runner = PageParseRunner(synthetic_pdf, tmp_path / "proc", tmp_path / "img", opts)
    r1 = runner.run()
    assert r1.pages_parsed == 3 and r1.pages_failed == [] and r1.batches_total == 2
    r2 = runner.run()
    assert r2.batches_skipped_from_checkpoint == 2
    pages = list(runner.iter_pages())
    assert [p.pdf_page_index for p in pages] == [0, 1, 2]


def test_parse_batch_records_failures(synthetic_pdf, tmp_path, monkeypatch):
    from app.ingestion import layout_parser, pdf_parser

    real = layout_parser.parse_page

    def flaky(doc, idx, *a, **k):
        if idx == 1:
            raise RuntimeError("boom")
        return real(doc, idx, *a, **k)

    monkeypatch.setattr(layout_parser, "parse_page", flaky)
    monkeypatch.setattr(pdf_parser, "_WORKER_DOC", None)
    out = tmp_path / "b.jsonl"
    res = pdf_parser.parse_batch(str(synthetic_pdf), 0, 3, str(out), str(tmp_path), ParseOptions(retries=1))
    assert res["failed"] == [1]
    lines = out.read_text().splitlines()
    assert len(lines) == 3 and '"error":"RuntimeError: boom"' in lines[1]


def test_unruled_legend_table_with_merged_header_span(tmp_path):
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((70, 100), "Item Part Number", fontname="hebo", fontsize=10)  # one merged header span
    p.insert_text((254, 100), "Description", fontname="hebo", fontsize=10)
    for i, (num, part, desc) in enumerate([("1", "AB123", "Widget hose"), ("2", "CD456", "Widget clamp (2 required)")]):
        y = 117 + i * 17
        p.insert_text((70, y), num, fontname="helv", fontsize=10)
        p.insert_text((97, y), part, fontname="helv", fontsize=10)
        p.insert_text((166, y), desc, fontname="helv", fontsize=10)
    rec = parse_page(doc, 0, tmp_path, {}, ocr_enabled=False)
    tables = [e.table for e in rec.elements if e.kind == ElementKind.TABLE]
    assert len(tables) == 1
    assert tables[0].header == ["Item", "Part Number", "Description"]
    assert tables[0].rows == [["1", "AB123", "Widget hose"], ["2", "CD456", "Widget clamp (2 required)"]]
    assert not any(e.kind == ElementKind.HEADING for e in rec.elements)
