"""Deterministic test doubles."""

from __future__ import annotations

import hashlib
import math

from app.retrieval.bm25_search import tokenize
from app.schemas.chunk import Applicability, Chunk


class HashEmbedder:
    """Bag-of-words hashing embedder: similar token sets -> similar vectors. Offline and fast."""

    def __init__(self, dim: int = 128, model_name: str = "fake-hash-embedder") -> None:
        self.dim = dim
        self.model_name = model_name
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in tokenize(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def make_chunk(cid: str, content: str, *, title: str = "Proc", parent: str = "proc_A", seq: int = 0,
               engines: list[str] | None = None, chunk_type: str = "procedure", figure_ids: list[str] | None = None,
               subsection: str | None = None, steps: tuple[int, int] | None = None) -> Chunk:  # fmt: skip
    return Chunk(
        chunk_id=cid,
        document_id="doc",
        document_title="Testcar 2099 Workshop Manual",
        source_hash="hash",
        parent_id=parent,
        procedure_id=parent.removeprefix("proc_"),
        source_procedure_id=parent.removeprefix("proc_"),
        procedure_title=title,
        section_code="999-01",
        section_name="Widget System",
        doc_type="REMOVAL AND INSTALLATION",
        subsection=subsection,
        chunk_type=chunk_type,
        sequence_index=seq,
        split_reason="heading",
        step_range=steps,
        page_start=10 + seq,
        page_end=10 + seq,
        pdf_page_start=9 + seq,
        pdf_page_end=9 + seq,
        printed_page_start=f"Page {seq + 1} of 3",
        printed_page_end=f"Page {seq + 1} of 3",
        figure_ids=figure_ids or [],
        applicability=Applicability(model_years=[2099], engines=engines or []),
        context_header=f"Testcar 2099 Workshop Manual > 999-01 Widget System > {title}",
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
        token_count=len(content.split()),
    )


def make_profile():
    """A small invented catalogue mirroring the manual's *shape* (not its content)."""
    from app.services.manual_profile import ManualProfile

    def proc(pid, title, section, doc_type, engines=(), trans=(), body=()):
        return {"parent_id": f"proc_{pid}", "procedure_id": pid, "title": title, "section_code": section[:6],
                "section_name": section[7:], "doc_type": doc_type, "toc_path": [], "subsections": [],
                "engines": list(engines), "transmissions": list(trans), "body_styles": list(body), "variants": [],
                "equipment_conditions": [], "chunk_types": {}}  # fmt: skip

    return ManualProfile({
        "model_years": [2014], "model_name": "Mustang",
        "covered": {"engines": ["3.7L", "5.0L", "5.8L"], "transmissions": ["automatic", "manual"],
                    "body_styles": ["convertible", "coupe"]},
        "procedures": [
            proc("CP37", "Coolant Pump — 3.7L", "303-03A Engine Cooling", "REMOVAL AND INSTALLATION", engines=["3.7L"]),
            proc("CP50", "Coolant Pump — 5.0L", "303-03A Engine Cooling", "REMOVAL AND INSTALLATION", engines=["5.0L"]),
            proc("DTP", "Door Trim Panel", "501-05 Interior Trim and Ornamentation", "REMOVAL AND INSTALLATION"),
            proc("CLU", "Clutch Disc and Pressure Plate", "308-01 Clutch", "REMOVAL AND INSTALLATION", trans=["manual"]),
            proc("QTP", "Quarter Trim Panel — Convertible", "501-05 Interior Trim and Ornamentation",
                 "REMOVAL AND INSTALLATION", body=["convertible"]),
            proc("QTC", "Quarter Trim Panel — Coupe", "501-05 Interior Trim and Ornamentation",
                 "REMOVAL AND INSTALLATION", body=["coupe"]),
            proc("FDB", "Brake Pads", "206-03 Front Disc Brake", "REMOVAL AND INSTALLATION"),
        ],
    })  # fmt: skip
