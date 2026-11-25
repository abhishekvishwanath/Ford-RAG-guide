"""Resumable OCR of unique illustration assets.

Ford illustrations carry callout text (torque values, figure ids) only in the raster.
OCR output is stored per asset and is always labelled as OCR-derived downstream; the
original image remains the authority.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from app.core.logging import log_event
from app.ingestion.ocr_service import find_figure_numbers, ocr_file

logger = logging.getLogger(__name__)


def _ocr_one(args: tuple[str, str]) -> dict:
    asset_id, path = args
    try:
        res = ocr_file(Path(path))
        return {
            "asset_id": asset_id,
            "text": res.text,
            "confidence": res.confidence,
            "figure_numbers": find_figure_numbers(res.text),
        }
    except Exception as exc:
        return {"asset_id": asset_id, "text": "", "confidence": None, "figure_numbers": [], "error": str(exc)}


def load_figure_ocr(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                out[rec["asset_id"]] = rec
    return out


def run_figure_ocr(assets: Iterable[tuple[str, Path]], out_path: Path, workers: int) -> dict:
    """OCR assets not yet present in ``out_path`` (append-only JSONL = natural checkpoint)."""
    done = load_figure_ocr(out_path)
    todo = [(a, str(p)) for a, p in dict(assets).items() if a not in done]
    log_event(logger, "figure ocr start", todo=len(todo), already_done=len(done))
    errors = 0
    with ProcessPoolExecutor(max_workers=workers) as pool, out_path.open("a", encoding="utf-8") as fh:
        for i, rec in enumerate(pool.map(_ocr_one, todo, chunksize=8)):
            errors += bool(rec.get("error"))
            fh.write(json.dumps(rec) + "\n")
            if i % 200 == 0:
                fh.flush()
                log_event(logger, "figure ocr progress", done=i, total=len(todo))
    return {"ocr_new": len(todo), "ocr_cached": len(done), "ocr_errors": errors}
