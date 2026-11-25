# Implementation Log

## Phase 1 — Input inspection and project skeleton — DONE (2026-11-25)

**Objective:** inspect diagram + manual, write source assessment, create runnable skeleton.

**Inputs inspected**
- Architecture diagram (chat attachment): two-phase pipeline — structure chunking (## split, H3 gate 400 words),
  metadata, all-MiniLM-L6-v2, persistent Chroma, `get_or_create_collection` incremental ingestion;
  BM25 + vector + RRF, TOP_K=5, LangGraph retrieve→build_prompt→generate, grounded Claude answer with citation or refusal.
- Manual: see `docs/source_manual_assessment.md`. Key finding: MY2014 only, despite the filename.

**Decisions**
- Scope strictly MY2014 (asked user; confirmed).
- LLM provider: Groq (user request) instead of Claude API. Model `openai/gpt-oss-120b`
  (largest reasoning model in the account's Groq model list on 2026-11-25; 131k context).
  The LLM client is behind an interface so Claude can be added later.
- Embedding max length 256 → multi-window dense indexing (documented in architecture).

**Created**
- `backend/app/core/{config,logging,exceptions}.py`, `backend/app/main.py`, `/api/health`
- `scripts/inspect_manual.py`, `.env.example`, `.gitignore`, `pyproject.toml`, `backend/requirements.txt` (pinned)
- `frontend/` Vite + React 19 + TS + Tailwind v4 + lucide-react + Vitest

**Tests**
- `pytest -q` → 1 passed
- `ruff check` → clean
- `frontend: vitest run` → 1 passed; `npm run build` → OK

**Known issues**
- None blocking.

## Phase 2 — Large PDF parsing — DONE (2026-11-25)

**Objective:** page-batched, resumable parsing into a normalized document model.

**Modules:** `ingestion/{boilerplate,layout_parser,table_extractor,visual_extractor,ocr_service,pdf_parser,structure_detector,figure_ocr,ingestion_service}.py`, `schemas/document.py`, `scripts/ingest_manual.py`.

**Design highlights**
- Worker processes parse 25-page batches; each finished batch is written atomically and checkpointed
  (keyed by source SHA-256 + `PARSER_VERSION`). Interrupted runs resume; a changed PDF or parser re-parses.
  Cancellation via Ctrl-C or `data/processed/CANCEL`. Page failures are retried, then recorded (never dropped).
- Typography-based paragraph classification: title, section header, doc type, revision, heading,
  applicability qualifier, step, sub-step, bullet, admonition (WARNING/CAUTION/NOTICE/NOTE), paragraph.
- Tables: ruled (`find_tables` + per-cell clipped re-extraction), un-ruled with bold header (text strategy),
  and a column-assignment fallback for headers printed as one span; rows cut by a page break are re-joined.
- Boilerplate (footer, print banners) and the third-party e-mail watermark are removed.
- Images saved once per content hash; placements become figure refs with bbox; banners filtered;
  tool icons vs illustrations distinguished. Figure OCR (Tesseract, psm 11) runs as a separate resumable stage.
- Document level: pages grouped by the source procedure id; admonitions that open a step are moved before
  it and the step absorbs its action text; indented renumbered items become sub-steps; lower-case
  continuations are joined; header-less tables at a page top continue the previous table.

**Full-manual results (parser 1.1.0):** 5,007 pages parsed in 488 s with 8 workers; **0 failed pages**,
**0 pages needed OCR** (no scanned pages); 446 pages with warnings (blank procedure tail pages);
942 procedure documents; 3,261 unique illustrations OCR'd in 105 s (0 errors).

**Tests:** synthetic-PDF unit tests for footer/page numbering, watermark removal, heading/qualifier/step/
admonition detection, step-number regex, table row integrity, merged-span legend tables, figure extraction,
banner removal, asset dedup, OCR routing and callout OCR, structure grouping and warning ordering,
TOC tolerance, checkpoint resume, page-failure reporting.

## Phase 3 — Hierarchical chunking and metadata — DONE (2026-11-25)

**Modules:** `ingestion/{chunker,metadata,chunk_stage}.py`, `schemas/chunk.py`, `embeddings/tokenizer.py`,
`config/applicability_vocabulary.json`, `config/synonyms.json`.

**Design highlights**
- Procedure ≤ 400 words → one chunk; otherwise split at sub-section headings; tiny sections merge forward;
  oversize sections are packed from atomic units (step + its warnings/bullets/figures; tables; qualifiers).
  Tables split by rows with the header repeated; token fallback only for a single oversize unit.
- Continuation chunks repeat procedure context, the active applicability qualifier and the section's
  WARNING/CAUTION preamble. Context header and repeated text are budgeted into the max size.
- Applicability from *structure only* (section, title, qualifier lines), normalised via a config vocabulary;
  unnormalised qualifiers kept verbatim as equipment conditions; "Vehicles without …" kept as exclusions.
- Parent procedure records, figure records (association method + confidence + OCR), manual profile
  (discovered years/variants + procedure catalogue for the readiness gate).

**Full-manual results (parser 1.1.0):** 4,134 chunks from 942 procedures; tokens median 294, p95 648,
max 897, **0 over max**; split reasons: heading 2,868 / whole procedure 597 / step group 663 / table rows 6.
Discovered coverage: engines 3.7L, 5.0L, 5.8L; transmissions automatic/manual; coupe/convertible; GT500.

**Known issues:** some un-ruled two-column "Item / Specification" material tables still fall back to
heading + paragraphs (text preserved, structure partial). Parser 1.2.0 addresses most; see Phase 4 run.
