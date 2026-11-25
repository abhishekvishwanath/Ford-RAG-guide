"""Recognition and removal of print boilerplate (footers, watermarks, banners).

Patterns describe the generic print layout of browser-printed workshop manuals
(``Page X of N`` / manual label / print date / source URL). They contain no
procedure-specific headings.
"""

from __future__ import annotations

import re

PAGE_OF_RE = re.compile(r"^Page (\d+) of (\d+)$")
MANUAL_LABEL_RE = re.compile(r"^((\d{4}) (.+?) Workshop Manual)$")
PRINT_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SOURCE_URL_RE = re.compile(r"^file://\S*")
SEB_ID_RE = re.compile(r"file=(SEB[0-9A-Z]+)\.HTM", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
SECTION_HEADER_RE = re.compile(r"^SECTION\s+(\d{3}-\d{2,3}[A-Z]?)\s*:\s*(.+)$")
REVISION_RE = re.compile(r"^Procedure revision date:\s*(\S+)")


def strip_emails(text: str) -> str:
    """Remove e-mail addresses (the source carries a personal watermark)."""
    return EMAIL_RE.sub("", text)


def is_footer_line(text: str) -> bool:
    t = text.strip()
    return bool(PAGE_OF_RE.match(t) or PRINT_DATE_RE.match(t) or SOURCE_URL_RE.match(t) or EMAIL_RE.fullmatch(t))


def is_banner(bbox: tuple[float, float, float, float], page_w: float, page_h: float) -> bool:
    """Wide, short images hugging the top/bottom page edge are print banners."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    near_edge = y0 < 0.05 * page_h or y1 > 0.92 * page_h
    return near_edge and w >= 0.3 * page_w and h <= 0.1 * page_h
