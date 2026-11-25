"""OCR via Tesseract for pages without a text layer and for figure callouts."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Figure identifiers printed inside Ford illustrations, e.g. "N0133215", "E146523"
FIGURE_NUMBER_RE = re.compile(r"\b([A-Z]\d{6,7})\b")


@dataclass
class OcrResult:
    text: str
    confidence: float | None  # mean word confidence 0..100
    engine: str = "tesseract"


@lru_cache
def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_image(image: Image.Image, *, psm: int = 11) -> OcrResult:
    """OCR an image. psm 11 = sparse text, suited to diagram callouts."""
    import pytesseract

    if not tesseract_available():
        raise RuntimeError("tesseract binary not found; install it or disable OCR")
    data = pytesseract.image_to_data(image, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for txt, conf in zip(data["text"], data["conf"], strict=False):
        c = float(conf)
        if txt.strip() and c >= 0:
            words.append(txt.strip())
            confs.append(c)
    text = " ".join(words)
    return OcrResult(text=text, confidence=round(sum(confs) / len(confs), 1) if confs else None)


def ocr_file(path: Path, *, psm: int = 11) -> OcrResult:
    with Image.open(path) as im:
        im = im.convert("L")
        # small figures OCR better when upscaled
        if max(im.size) < 1200:
            scale = 1200 / max(im.size)
            im = im.resize((int(im.width * scale), int(im.height * scale)))
        return ocr_image(im, psm=psm)


def find_figure_numbers(text: str) -> list[str]:
    return list(dict.fromkeys(FIGURE_NUMBER_RE.findall(text)))


def needs_page_ocr(text_layer_chars: int, min_chars: int) -> bool:
    """Route a page to OCR only when it has effectively no text layer."""
    return text_layer_chars < min_chars
