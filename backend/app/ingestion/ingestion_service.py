"""Ingestion orchestration: stages, status reporting, idempotency.

Stages (each resumable and individually re-runnable):
  parse     -> processed/pages/*.jsonl            (PDF layout parsing + OCR routing + image extraction)
  structure -> processed/documents.jsonl          (procedure documents with hierarchy)
  figures   -> processed/figure_ocr.jsonl         (OCR of unique illustrations)
  chunk     -> processed/chunks.jsonl             (hierarchical chunks + metadata)   [phase 3]
  index     -> chroma_db + indexes/bm25           (embeddings + keyword index)       [phase 4]
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from app.core.config import Settings
from app.core.exceptions import IngestionError
from app.core.logging import log_event
from app.ingestion.boilerplate import MANUAL_LABEL_RE
from app.ingestion.figure_ocr import run_figure_ocr
from app.ingestion.pdf_parser import PageParseRunner, ParseOptions, file_sha256
from app.ingestion.structure_detector import iter_procedures
from app.schemas.document import DocumentMetadata, ElementKind, ProcedureDocument

logger = logging.getLogger(__name__)

ALL_STAGES = ("parse", "structure", "figures", "chunk", "index")
StageFn = Callable[[], dict]


class IngestionService:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.processed = settings.processed_dir
        self.status_path = self.processed / "ingestion_status.json"
        self.documents_path = self.processed / "documents.jsonl"
        self.metadata_path = self.processed / "document_metadata.json"
        self.figure_ocr_path = self.processed / "figure_ocr.jsonl"
        self._lock = threading.Lock()
        self.page_range: tuple[int, int] | None = None
        self._source_hash: str | None = None

    # ------------------------------------------------------------ helpers
    @property
    def runner(self) -> PageParseRunner:
        opts = ParseOptions(
            batch_size=self.s.ingestion_batch_size,
            ocr_enabled=self.s.ocr_enabled,
            ocr_min_chars=self.s.ocr_min_text_chars,
            retries=self.s.page_retry_attempts,
        )
        return PageParseRunner(self.s.manual_pdf_path, self.processed, self.s.images_dir, opts)

    def source_hash(self) -> str:
        if self._source_hash is None:
            self._source_hash = file_sha256(self.s.manual_pdf_path)
        return self._source_hash

    def validate_source(self) -> None:
        path = self.s.manual_pdf_path
        if not path.exists():
            raise IngestionError(f"Manual PDF not found at configured path ({path.name}).")
        if path.suffix.lower() != ".pdf":
            raise IngestionError("Source must be a .pdf file.")
        if path.stat().st_size > self.s.max_pdf_size_mb * 1024 * 1024:
            raise IngestionError("Source PDF exceeds the configured size limit.")
        with path.open("rb") as fh:
            if fh.read(5) != b"%PDF-":
                raise IngestionError("Source file is not a valid PDF.")
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                raise IngestionError("Source PDF requires a password.")

    def read_status(self) -> dict:
        if self.status_path.exists():
            return json.loads(self.status_path.read_text())
        return {"state": "never_run", "stages": {}}

    def _write_status(self, **update: object) -> None:
        with self._lock:
            status = self.read_status()
            status.update(update)
            status["updated_at"] = datetime.now(UTC).isoformat()
            self.processed.mkdir(parents=True, exist_ok=True)
            tmp = self.status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(status, indent=1, default=str))
            tmp.replace(self.status_path)

    def _stage_status(self, stage: str, **info: object) -> None:
        status = self.read_status()
        stages = status.get("stages", {})
        stages[stage] = {**stages.get(stage, {}), **info}
        self._write_status(stages=stages)

    def iter_documents(self) -> Iterator[ProcedureDocument]:
        if not self.documents_path.exists():
            return
        with self.documents_path.open(encoding="utf-8") as fh:
            for line in fh:
                yield ProcedureDocument.model_validate_json(line)

    # ------------------------------------------------------------ stages
    def stage_parse(self) -> dict:
        def progress(done: int, total: int) -> None:
            self._stage_status("parse", progress=f"{done}/{total} batches")

        report = self.runner.run(self.page_range, source_hash=self.source_hash(), progress=progress)
        if report.cancelled:
            raise IngestionError("Parsing cancelled; completed batches are checkpointed.")
        return {
            "pages_parsed": report.pages_parsed,
            "pages_failed": report.pages_failed,
            "pages_ocr": report.pages_ocr,
            "pages_with_warnings": report.pages_with_warnings,
            "batches_total": report.batches_total,
            "batches_from_checkpoint": report.batches_skipped_from_checkpoint,
        }

    def stage_structure(self) -> dict:
        with pymupdf.open(self.s.manual_pdf_path) as doc:
            toc = doc.get_toc()
            page_count = doc.page_count
        labels: Counter[str] = Counter()
        dates: set[str] = set()
        kinds: Counter[str] = Counter()
        n_docs = 0
        warnings: Counter[str] = Counter()

        def pages() -> Iterator:
            for rec in self.runner.iter_pages(self.page_range):
                if rec.manual_label:
                    labels[rec.manual_label] += 1
                if rec.print_date:
                    dates.add(rec.print_date)
                yield rec

        tmp = self.documents_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for pdoc in iter_procedures(pages(), toc):
                n_docs += 1
                kinds.update(e.kind.value for e in pdoc.elements)
                warnings.update(w.split("_at_")[0] for w in pdoc.warnings)
                fh.write(pdoc.model_dump_json(exclude_none=True) + "\n")
        tmp.replace(self.documents_path)

        years = sorted({int(m.group(2)) for lbl in labels if (m := MANUAL_LABEL_RE.match(lbl))})
        names = Counter(m.group(3) for lbl in labels if (m := MANUAL_LABEL_RE.match(lbl)))
        meta = DocumentMetadata(
            document_id=self.s.document_id,
            file_name=self.s.manual_pdf_path.name,
            source_hash=self.source_hash(),
            page_count=page_count,
            manual_labels=dict(labels),
            model_years=years,
            model_name=names.most_common(1)[0][0] if names else None,
            print_dates=sorted(dates),
            toc_entries=len(toc),
        )
        self.metadata_path.write_text(meta.model_dump_json(indent=1))
        return {"documents": n_docs, "element_kinds": dict(kinds), "document_warnings": dict(warnings)}

    def stage_figures(self) -> dict:
        if not self.s.figure_ocr_enabled:
            return {"skipped": "FIGURE_OCR_ENABLED=false"}
        assets: dict[str, Path] = {}
        for pdoc in self.iter_documents():
            for el in pdoc.elements:
                f = el.figure
                if f and f.image_type == "illustration" and min(f.width_px, f.height_px) >= self.s.figure_min_side_px:
                    assets[f.asset_id] = self.s.images_dir / f.asset_relpath
        result = run_figure_ocr(assets.items(), self.figure_ocr_path, workers=ParseOptions().workers)
        return {"unique_illustrations": len(assets), **result}

    def stages(self) -> dict[str, StageFn]:
        from app.ingestion import later_stages  # chunk/index registered in later phases

        return {
            "parse": self.stage_parse,
            "structure": self.stage_structure,
            "figures": self.stage_figures,
            **later_stages.stage_functions(self),
        }

    # ------------------------------------------------------------ run
    def run(self, stages: list[str] | None = None, page_range: tuple[int, int] | None = None) -> dict:
        self.validate_source()
        self.page_range = page_range
        available = self.stages()
        selected = [s for s in (stages or list(ALL_STAGES)) if s in available]
        unknown = set(stages or []) - set(available)
        if unknown:
            raise IngestionError(f"Unknown or unavailable stages: {sorted(unknown)}")
        self._write_status(
            state="running",
            started_at=datetime.now(UTC).isoformat(),
            source_hash=self.source_hash(),
            page_range=page_range,
            requested_stages=selected,
        )
        results: dict[str, dict] = {}
        for name in selected:
            t0 = time.perf_counter()
            self._stage_status(name, state="running")
            try:
                out = available[name]()
            except Exception as exc:
                self._stage_status(name, state="failed", error=str(exc))
                self._write_status(state="failed")
                log_event(logger, "ingestion stage failed", logging.ERROR, stage=name, error=str(exc))
                raise
            out["seconds"] = round(time.perf_counter() - t0, 1)
            results[name] = out
            self._stage_status(name, state="done", result=out)
            log_event(logger, "ingestion stage done", stage=name, seconds=out["seconds"])
        self._write_status(state="done", finished_at=datetime.now(UTC).isoformat())
        return results


def figure_count(doc: ProcedureDocument) -> int:
    return sum(1 for e in doc.elements if e.kind == ElementKind.FIGURE)
