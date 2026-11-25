"""Batched, resumable, parallel page parsing.

* Pages are parsed in bounded batches by worker processes, each holding one open
  document handle; nothing is accumulated in memory beyond one batch.
* Every finished batch is written atomically to ``processed/pages/batch_XXXXX.jsonl``
  and recorded in a checkpoint keyed by source hash + parser version, so an
  interrupted run resumes where it stopped and a changed PDF/parser forces a re-parse.
* Page failures are retried, then recorded (never silently dropped).
* Cancellation: create ``processed/CANCEL`` (or Ctrl-C); finished batches are kept.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app.core.logging import log_event
from app.schemas.document import PageDiagnostics, PageRecord

logger = logging.getLogger(__name__)

PARSER_VERSION = "1.2.0"


def file_sha256(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class ParseOptions:
    batch_size: int = 25
    workers: int = max(1, (os.cpu_count() or 2) - 2)
    ocr_enabled: bool = True
    ocr_min_chars: int = 40
    retries: int = 2


@dataclass
class ParseReport:
    source_hash: str
    page_count: int
    pages_parsed: int = 0
    pages_failed: list[int] = field(default_factory=list)
    pages_ocr: list[int] = field(default_factory=list)
    pages_with_warnings: int = 0
    batches_total: int = 0
    batches_skipped_from_checkpoint: int = 0
    cancelled: bool = False
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------- worker side
_WORKER_DOC: pymupdf.Document | None = None
_WORKER_XREF_CACHE: dict = {}


def _worker_doc(pdf_path: str) -> pymupdf.Document:
    global _WORKER_DOC
    if _WORKER_DOC is None:
        _WORKER_DOC = pymupdf.open(pdf_path)
    return _WORKER_DOC


def parse_batch(pdf_path: str, start: int, end: int, out_file: str, images_dir: str, opts: ParseOptions) -> dict:
    """Parse pages [start, end) and write one JSONL file. Runs in a worker process."""
    from app.ingestion.layout_parser import parse_page

    doc = _worker_doc(pdf_path)
    failed, ocr_pages, warned = [], [], 0
    tmp = Path(out_file).with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for idx in range(start, end):
            record: PageRecord | None = None
            last_err = ""
            for _attempt in range(opts.retries + 1):
                try:
                    record = parse_page(
                        doc,
                        idx,
                        Path(images_dir),
                        _WORKER_XREF_CACHE,
                        ocr_enabled=opts.ocr_enabled,
                        ocr_min_chars=opts.ocr_min_chars,
                    )
                    break
                except Exception as exc:  # recorded below; never silently skipped
                    last_err = f"{type(exc).__name__}: {exc}"
            if record is None:
                record = PageRecord(pdf_page_index=idx, diagnostics=PageDiagnostics(error=last_err))
                failed.append(idx)
            if record.diagnostics.ocr_used:
                ocr_pages.append(idx)
            if record.diagnostics.warnings:
                warned += 1
            fh.write(record.model_dump_json(exclude_none=True) + "\n")
    tmp.replace(out_file)
    pymupdf.TOOLS.store_shrink(100)
    return {"start": start, "end": end, "file": out_file, "failed": failed, "ocr": ocr_pages, "warned": warned}


# ---------------------------------------------------------------- driver side
class PageParseRunner:
    def __init__(self, pdf_path: Path, processed_dir: Path, images_dir: Path, opts: ParseOptions) -> None:
        self.pdf_path = pdf_path
        self.pages_dir = processed_dir / "pages"
        self.checkpoint_path = processed_dir / "parse_checkpoint.json"
        self.cancel_path = processed_dir / "CANCEL"
        self.images_dir = images_dir
        self.opts = opts

    def _load_checkpoint(self, source_hash: str) -> dict:
        if self.checkpoint_path.exists():
            cp = json.loads(self.checkpoint_path.read_text())
            if cp.get("source_hash") == source_hash and cp.get("parser_version") == PARSER_VERSION:
                return cp
            log_event(logger, "checkpoint invalidated (source or parser changed); re-parsing")
        return {"source_hash": source_hash, "parser_version": PARSER_VERSION, "batches": {}}

    def _save_checkpoint(self, cp: dict) -> None:
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cp, indent=1))
        tmp.replace(self.checkpoint_path)

    def run(
        self,
        page_range: tuple[int, int] | None = None,
        source_hash: str | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> ParseReport:
        t0 = time.perf_counter()
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.cancel_path.unlink(missing_ok=True)
        source_hash = source_hash or file_sha256(self.pdf_path)
        with pymupdf.open(self.pdf_path) as doc:
            n_pages = doc.page_count
        lo, hi = page_range or (0, n_pages)
        hi = min(hi, n_pages)
        cp = self._load_checkpoint(source_hash)
        report = ParseReport(source_hash=source_hash, page_count=n_pages)
        bs = self.opts.batch_size
        todo: list[tuple[int, int]] = []
        # boundaries are aligned to multiples of batch_size so partial and full runs share batches
        for start in range((lo // bs) * bs, hi, bs):
            end = min(start + bs, n_pages)
            key = f"{start}-{end}"
            report.batches_total += 1
            if key in cp["batches"] and Path(cp["batches"][key]["file"]).exists():
                report.batches_skipped_from_checkpoint += 1
                self._accumulate(report, cp["batches"][key])
            else:
                todo.append((start, end))
        done = report.batches_skipped_from_checkpoint
        log_event(logger, "parse start", batches=len(todo), skipped=done, workers=self.opts.workers)
        try:
            with ProcessPoolExecutor(max_workers=self.opts.workers) as pool:
                pending: dict[Future, tuple[int, int]] = {}
                queue = iter(todo)
                self._fill(pool, pending, queue)
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        start, end = pending.pop(fut)
                        result = fut.result()
                        cp["batches"][f"{start}-{end}"] = result
                        self._save_checkpoint(cp)
                        self._accumulate(report, result)
                        done += 1
                        if progress:
                            progress(done, report.batches_total)
                    if self.cancel_path.exists():
                        report.cancelled = True
                        for f in pending:
                            f.cancel()
                        break
                    self._fill(pool, pending, queue)
        except KeyboardInterrupt:
            report.cancelled = True
        report.seconds = round(time.perf_counter() - t0, 1)
        log_event(logger, "parse finished", **{k: v for k, v in report.to_dict().items() if not isinstance(v, list)})
        return report

    def _fill(self, pool: ProcessPoolExecutor, pending: dict, queue: Iterator[tuple[int, int]]) -> None:
        while len(pending) < self.opts.workers * 2:
            try:
                start, end = next(queue)
            except StopIteration:
                return
            out = self.pages_dir / f"batch_{start:05d}_{end:05d}.jsonl"
            fut = pool.submit(parse_batch, str(self.pdf_path), start, end, str(out), str(self.images_dir), self.opts)
            pending[fut] = (start, end)

    @staticmethod
    def _accumulate(report: ParseReport, result: dict) -> None:
        report.pages_parsed += result["end"] - result["start"]
        report.pages_failed.extend(result["failed"])
        report.pages_ocr.extend(result["ocr"])
        report.pages_with_warnings += result["warned"]

    def iter_pages(self, page_range: tuple[int, int] | None = None) -> Iterator[PageRecord]:
        """Stream parsed pages in page order (one batch file in memory at a time)."""
        if not self.checkpoint_path.exists():
            return
        cp = json.loads(self.checkpoint_path.read_text())
        batches = sorted(cp["batches"].values(), key=lambda b: b["start"])
        lo, hi = page_range or (0, 10**9)
        seen: set[int] = set()
        for b in batches:
            if b["end"] <= lo or b["start"] >= hi:
                continue
            with Path(b["file"]).open(encoding="utf-8") as fh:
                for line in fh:
                    rec = PageRecord.model_validate_json(line)
                    if lo <= rec.pdf_page_index < hi and rec.pdf_page_index not in seen:
                        seen.add(rec.pdf_page_index)
                        yield rec
