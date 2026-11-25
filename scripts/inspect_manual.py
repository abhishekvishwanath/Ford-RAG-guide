"""Inspect the source manual and print the facts used in docs/source_manual_assessment.md.

Usage: python scripts/inspect_manual.py [path/to/manual.pdf] [--json out.json]
Read-only: never modifies the PDF.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pymupdf  # noqa: E402

from app.core.config import get_settings  # noqa: E402

BOILERPLATE = re.compile(
    r"Page \d+ of \d+|\d{4} \w+ Workshop Manual|\d{4}-\d{2}-\d{2}|file://\S+|[\w.+-]+@[\w-]+\.[\w.]+"
)


def inspect(path: Path) -> dict:
    doc = pymupdf.open(path)
    toc = doc.get_toc()
    stats: dict = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "pages": doc.page_count,
        "needs_pass": bool(doc.needs_pass),
        "metadata": {k: v for k, v in doc.metadata.items() if k != "author"},
        "toc_entries": len(toc),
        "toc_depth": dict(collections.Counter(t[0] for t in toc)),
    }
    years: collections.Counter = collections.Counter()
    seb: set[str] = set()
    sections: set[str] = set()
    engines: collections.Counter = collections.Counter()
    low_text: list[int] = []
    content_images = 0
    for i, page in enumerate(doc):
        text = page.get_text()
        years.update(re.findall(r"(\d{4}) \w+ Workshop Manual", text))
        seb.update(re.findall(r"(SEB[0-9A-Z]+)\.HTM", text))
        sections.update(re.findall(r"SECTION (\d{3}-\d{2,3}[A-Z]?):", text))
        engines.update(re.findall(r"\b(\d\.\dL)\b", text))
        if len(BOILERPLATE.sub("", text).strip()) < 40:
            low_text.append(i + 1)
        for info in page.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            if x1 - x0 > 80 and y1 - y0 > 80:
                content_images += 1
    stats.update(
        {
            "model_years_in_footer": dict(years),
            "procedure_documents": len(seb),
            "sections": len(sections),
            "engine_mentions": dict(engines.most_common(10)),
            "low_text_pages": len(low_text),
            "content_images": content_images,
        }
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=str(get_settings().manual_pdf_path))
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()
    path = Path(args.pdf)
    if not path.exists():
        sys.exit(f"Manual not found: {path}")
    stats = inspect(path)
    print(json.dumps(stats, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
