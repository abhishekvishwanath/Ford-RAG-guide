"""Extraction and persistence of original manual visuals.

Images are stored once per unique content hash (the same special-tool icon appears
on hundreds of pages). Each *placement* on a page becomes a ``FigureRef`` pointing
at the stored asset, with its page bbox preserved for association and cropping.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pymupdf

from app.ingestion.boilerplate import is_banner
from app.schemas.document import FigureRef

logger = logging.getLogger(__name__)

MIN_DISPLAY_PT = 20.0  # smaller placements are glyph-like decorations
ICON_MAX_PT = 90.0  # near-square placements up to this size are icons (special tools, symbols)
RENDER_DPI = 150


def _save_pixmap(pix: pymupdf.Pixmap, images_dir: Path) -> tuple[str, str, int, int]:
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)  # drop alpha
    if pix.colorspace and pix.colorspace.n not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    data = pix.tobytes("png")
    digest = hashlib.sha256(data).hexdigest()[:20]
    asset_id = f"img_{digest}"
    rel = f"{digest[:2]}/{asset_id}.png"
    path = images_dir / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return asset_id, rel, pix.width, pix.height


def extract_figures(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    pdf_page_index: int,
    images_dir: Path,
    xref_cache: dict[int, tuple[str, str, int, int]],
) -> tuple[list[FigureRef], int]:
    """Return content figures on the page and the number of banners removed."""
    figures: list[FigureRef] = []
    banners = 0
    pw, ph = page.rect.width, page.rect.height
    for idx, info in enumerate(page.get_image_info(xrefs=True)):
        bbox = tuple(round(v, 2) for v in info["bbox"])
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if is_banner(bbox, pw, ph):  # type: ignore[arg-type]
            banners += 1
            continue
        if min(w, h) < MIN_DISPLAY_PT:
            continue
        xref = info.get("xref") or 0
        method = "embedded_image"
        try:
            if xref and xref in xref_cache:
                asset_id, rel, wpx, hpx = xref_cache[xref]
            elif xref:
                asset_id, rel, wpx, hpx = _save_pixmap(pymupdf.Pixmap(doc, xref), images_dir)
                xref_cache[xref] = (asset_id, rel, wpx, hpx)
            else:
                raise ValueError("inline image without xref")
        except Exception:
            method = "page_region_render"
            pix = page.get_pixmap(clip=pymupdf.Rect(bbox), dpi=RENDER_DPI)
            asset_id, rel, wpx, hpx = _save_pixmap(pix, images_dir)
        is_icon = max(w, h) <= ICON_MAX_PT and 0.6 <= (w / h if h else 0) <= 1.6
        figures.append(
            FigureRef(
                figure_id=f"fig_p{pdf_page_index}_{idx}",
                asset_id=asset_id,
                pdf_page_index=pdf_page_index,
                bbox=bbox,  # type: ignore[arg-type]
                width_px=wpx,
                height_px=hpx,
                image_type="tool_icon" if is_icon else "illustration",
                extraction_method=method,
                asset_relpath=rel,
            )
        )
    return figures, banners
