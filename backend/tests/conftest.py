"""Shared fixtures. The synthetic manual imitates the *layout conventions* of a
browser-printed workshop manual; its content is invented test data, not manual text."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

LABEL = "2041 Testcar Workshop Manual"


def _footer(page: pymupdf.Page, n: int, total: int, seb: str) -> None:
    y = 750
    page.insert_text((50, y), f"Page {n} of {total}", fontname="tiro", fontsize=12)
    page.insert_text((50, y + 12), LABEL, fontname="tiro", fontsize=12)
    page.insert_text((50, y + 24), "2041-01-02", fontname="tiro", fontsize=12)
    page.insert_text(
        (250, y), f"file://C:\\X\\cache\\SEB~us~en~file={seb}.HTM~gen~ref.HTM", fontname="tiro", fontsize=8
    )
    page.insert_text((250, y + 30), "someone@example.com", fontname="helv", fontsize=4)
    # print banners (wide short images at the page edges)
    banner = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 10), 0)
    banner.clear_with(200)
    page.insert_image(pymupdf.Rect(0, 741, 306, 792), pixmap=banner)


def _header(page: pymupdf.Page, title: str, section: str, doc_type: str) -> None:
    page.insert_text((66, 80), title, fontname="hebo", fontsize=12)
    page.insert_text((68, 100), f"SECTION {section}", fontname="heit", fontsize=9)
    page.insert_text((400, 100), LABEL, fontname="heit", fontsize=9)
    page.insert_text((68, 113), doc_type, fontname="heit", fontsize=9)
    page.insert_text((400, 113), "Procedure revision date: 01/02/2098", fontname="helv", fontsize=8)


def _diagram(text: str) -> pymupdf.Pixmap:
    """A raster 'diagram' with callout text rendered into it (for OCR tests)."""
    tmp = pymupdf.open()
    p = tmp.new_page(width=300, height=200)
    p.draw_rect(pymupdf.Rect(20, 20, 280, 180), color=(0, 0, 0), width=2)
    p.insert_text((40, 100), text, fontname="helv", fontsize=22)
    return p.get_pixmap(dpi=150)


def build_synthetic_manual(path: Path) -> Path:
    doc = pymupdf.open()
    # --- procedure A: two pages
    p = doc.new_page(width=612, height=792)
    _header(p, "Widget Bracket", "999-01: Widget System", "REMOVAL AND INSTALLATION")
    p.insert_text((66, 150), "Removal", fontname="hebo", fontsize=10)
    p.insert_text((66, 175), "WARNING:", fontname="hebo", fontsize=10)
    p.insert_text((120, 175), "Support the widget before removal.", fontname="hebo", fontsize=10)
    p.insert_text((66, 200), "All vehicles", fontname="hebo", fontsize=10)
    p.insert_text((72, 225), "1.", fontname="helv", fontsize=10)
    p.insert_text((95, 225), "Disconnect the widget connector.", fontname="helv", fontsize=10)
    p.insert_text((72, 250), "2.", fontname="helv", fontsize=10)
    p.insert_text((95, 250), "NOTICE:", fontname="hebo", fontsize=10)
    p.insert_text((150, 250), "Do not reuse the widget nut.", fontname="hebo", fontsize=10)
    p.insert_text((95, 275), "Remove the widget nut.", fontname="helv", fontsize=10)
    p.insert_text((66, 300), "Vehicles with 9.9L engine", fontname="hebo", fontsize=10)
    p.insert_text((72, 325), "3.", fontname="helv", fontsize=10)
    p.insert_text((95, 325), "Remove the 9.9L heat shield.", fontname="helv", fontsize=10)
    p.insert_image(pymupdf.Rect(100, 360, 400, 560), pixmap=_diagram("77 Nm (57 lb-ft)"))
    _footer(p, 1, 2, "SEB99001")

    p = doc.new_page(width=612, height=792)
    p.insert_text((66, 80), "Torque Specifications", fontname="hebo", fontsize=10)
    # ruled table
    x = [66, 266, 336, 406]
    ys = [95, 112, 129, 146]
    for yy in ys:
        p.draw_line((x[0], yy), (x[-1], yy))
    for xx in x:
        p.draw_line((xx, ys[0]), (xx, ys[-1]))
    for i, h in enumerate(["Description", "Nm", "lb-ft"]):
        p.insert_text((x[i] + 3, ys[0] + 12), h, fontname="hebo", fontsize=10)
    for r, row in enumerate([["Widget nut", "77", "57"], ["Bracket bolts", "25", "18"]]):
        for i, c in enumerate(row):
            p.insert_text((x[i] + 3, ys[r + 1] + 12), c, fontname="helv", fontsize=10)
    p.insert_text((66, 200), "Installation", fontname="hebo", fontsize=10)
    p.insert_text((72, 225), "1.", fontname="helv", fontsize=10)
    p.insert_text((95, 225), "Install the widget nut.", fontname="helv", fontsize=10)
    p.insert_text((110, 240), "\u2022", fontname="helv", fontsize=10)
    p.insert_text((122, 240), "Tighten to 77 Nm (57 lb-ft).", fontname="helv", fontsize=10)
    _footer(p, 2, 2, "SEB99001")

    # --- procedure B: one page, image-only body (OCR routing candidate is the figure)
    p = doc.new_page(width=612, height=792)
    _header(p, "Gizmo Location", "999-02: Gizmo", "DESCRIPTION AND OPERATION")
    p.insert_image(pymupdf.Rect(100, 150, 400, 350), pixmap=_diagram("N0999999"))
    _footer(p, 1, 1, "SEB99002")

    doc.set_toc(
        [[1, "9. Test Systems", 1], [2, "999-01 Widget System", 1], [3, "Widget Bracket", 1], [2, "999-02 Gizmo", 3]]
    )
    doc.save(path)
    return path


@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_synthetic_manual(tmp_path_factory.mktemp("manual") / "synthetic_manual.pdf")
