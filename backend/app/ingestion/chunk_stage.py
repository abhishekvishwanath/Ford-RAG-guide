"""Chunk stage: procedure documents -> chunks, parent records, figure records, manual profile."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from app.embeddings.tokenizer import get_token_counter
from app.ingestion.chunker import ChunkingConfig, HierarchicalChunker
from app.ingestion.figure_ocr import load_figure_ocr
from app.ingestion.metadata import load_vocabulary
from app.schemas.chunk import Chunk, FigureRecord, ProcedureRecord
from app.schemas.document import DocumentMetadata

if TYPE_CHECKING:
    from app.ingestion.ingestion_service import IngestionService

MIN_PROCEDURES_FOR_COVERAGE = 5


def _write_jsonl(path: Path, items: list) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json(exclude_none=True) + "\n")
    tmp.replace(path)


def build_manual_profile(meta: DocumentMetadata, procedures: list[ProcedureRecord], chunks: list[Chunk]) -> dict:
    """Discovered applicability vocabulary + a procedure catalogue used by the readiness gate."""
    eng, trans, body, variants = Counter(), Counter(), Counter(), Counter()
    for p in procedures:
        eng.update(p.applicability.engines)
        trans.update(t for t in p.applicability.transmissions if t in ("automatic", "manual"))
        body.update(p.applicability.body_styles)
        variants.update(p.applicability.variants)
    chunk_types: dict[str, Counter] = {}
    for c in chunks:
        chunk_types.setdefault(c.parent_id, Counter())[c.chunk_type] += 1
    vocab = load_vocabulary()
    return {
        "document_id": meta.document_id,
        "source_hash": meta.source_hash,
        "model_name": meta.model_name,
        "model_years": meta.model_years,
        "covered": {
            "engines": sorted(e for e, n in eng.items() if n >= MIN_PROCEDURES_FOR_COVERAGE),
            "transmissions": sorted(t for t, n in trans.items() if n >= MIN_PROCEDURES_FOR_COVERAGE),
            "transmission_codes": [
                c
                for c in vocab.transmission_codes
                if trans.get(c) or any(c in p.applicability.transmissions for p in procedures)
            ],
            "body_styles": sorted(b for b, n in body.items() if n >= 1),
            "variants": sorted(variants),
        },
        "variant_procedure_counts": {
            "engines": dict(eng),
            "transmissions": dict(trans),
            "body_styles": dict(body),
            "variants": dict(variants),
        },
        "procedures": [
            {
                "parent_id": p.parent_id,
                "procedure_id": p.procedure_id,
                "title": p.title,
                "section_code": p.section_code,
                "section_name": p.section_name,
                "doc_type": p.doc_type,
                "toc_path": p.toc_path,
                "subsections": p.subsections,
                "engines": p.applicability.engines,
                "transmissions": p.applicability.transmissions,
                "body_styles": p.applicability.body_styles,
                "variants": p.applicability.variants,
                "equipment_conditions": p.applicability.equipment_conditions,
                "chunk_types": dict(chunk_types.get(p.parent_id, {})),
            }
            for p in procedures
        ],
    }


def run_chunk_stage(service: IngestionService) -> dict:
    s = service.s
    meta = DocumentMetadata.model_validate_json(service.metadata_path.read_text())
    counter = get_token_counter(s.embedding_model)
    title = (
        f"{meta.model_name or 'Vehicle'} {'/'.join(map(str, meta.model_years))} Workshop Manual"
        if meta.model_years
        else "Workshop Manual"
    )
    chunker = HierarchicalChunker(
        ChunkingConfig(s.chunk_target_tokens, s.chunk_max_tokens, s.chunk_overlap, s.heading_split_word_threshold),
        counter,
        document_id=s.document_id,
        document_title=title,
        source_hash=meta.source_hash,
        model_years=meta.model_years,
    )
    ocr = load_figure_ocr(service.figure_ocr_path)
    chunks: list[Chunk] = []
    procedures: list[ProcedureRecord] = []
    figures: list[FigureRecord] = []
    for doc in service.iter_documents():
        c, p, f = chunker.chunk_document(doc, ocr)
        chunks.extend(c)
        procedures.append(p)
        figures.extend(f)

    processed = s.processed_dir
    _write_jsonl(processed / "chunks.jsonl", chunks)
    _write_jsonl(processed / "procedures.jsonl", procedures)
    _write_jsonl(processed / "figures.jsonl", figures)
    profile = build_manual_profile(meta, procedures, chunks)
    s.indexes_dir.mkdir(parents=True, exist_ok=True)
    s.manual_profile_path.write_text(json.dumps(profile, indent=1))

    tokens = [c.token_count for c in chunks] or [0]
    report = {
        "chunks": len(chunks),
        "procedures": len(procedures),
        "figures_associated": len(figures),
        "figures_with_ocr_text": sum(1 for f in figures if f.ocr_text),
        "tokenizer_exact": counter.exact,
        "token_stats": {
            "min": min(tokens),
            "median": statistics.median(tokens),
            "mean": round(statistics.mean(tokens), 1),
            "p95": sorted(tokens)[int(len(tokens) * 0.95) - 1] if len(tokens) > 1 else tokens[0],
            "max": max(tokens),
            "over_max": sum(1 for t in tokens if t > s.chunk_max_tokens),
        },
        "by_type": dict(Counter(c.chunk_type for c in chunks)),
        "by_split_reason": dict(Counter(c.split_reason for c in chunks)),
        "continuations": sum(c.is_continuation for c in chunks),
        "with_engine_applicability": sum(bool(c.applicability.engines) for c in chunks),
        "with_transmission_applicability": sum(bool(c.applicability.transmissions) for c in chunks),
        "with_equipment_conditions": sum(bool(c.applicability.equipment_conditions) for c in chunks),
        "covered": profile["covered"],
    }
    (processed / "chunk_report.json").write_text(json.dumps(report, indent=1))
    return report
