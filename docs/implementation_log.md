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

## Phase 4 — Embeddings and persistent indexing — DONE (2026-11-25)

**Modules:** `embeddings/{embedding_service,tokenizer}.py`, `vectorstore/chroma_store.py`, `retrieval/bm25_search.py`, `ingestion/index_stage.py`.

- `all-MiniLM-L6-v2` truncates at 256 word-pieces, so each chunk is embedded as overlapping ~190-token windows
  prefixed with a short context label; dense hits are aggregated per chunk by max similarity (documented deviation).
- Persistent ChromaDB (`get_or_create_collection`, cosine). A manifest (chunk_id → content hash) makes indexing
  incremental: only new/changed chunks are embedded, stale chunks are deleted, an embedding-model change rebuilds.
- Chroma metadata is scalar: per-variant boolean flags (`eng_5_0l`, `trans_automatic`, …) plus `*_any` flags.
- BM25 corpus persisted as JSONL (no pickles); tokenizer keeps part/tool numbers, DTCs, displacements, units intact.

**Full-manual results (parser 1.2.0):** 5,007 pages / 0 failed / 0 OCR pages; 942 procedures; 4,039 chunks
(median 315 tokens, p95 665, max 924 — 1 chunk over 900); 9,847 vectors written in 59 s; tables 1,887.
Re-running chunk+index embeds 0 chunks (idempotent; verified in integration test).

## Phase 5 — Conversation context and readiness gate — DONE (2026-11-25)

**Modules:** `conversation/{state,context_extractor,readiness,question_planner,safety,session_manager}.py`,
`memory/{memory_models,session_store}.py`, `services/manual_profile.py`, `config/readiness_policy.json`, `graph/*`.

- Slots carry value, provenance (user_provided / confirmed / inferred / user_unsure), confidence, turn, basis,
  conflict. A restatement never downgrades provenance; trim names (GT, GT500, V6) only produce *inferred* engines.
- Rule extraction for explicit facts + optional Groq `gpt-oss-20b` JSON extraction for free text; LLM vehicle
  values not literally supported by the user's words are stored as inferred (must be confirmed).
- Readiness gate: safety → scope (manual years/engines from the profile) → conflicts → hard slots → inferred /
  low-confidence confirmation → weighted score ≥ 0.85. Engine/transmission/body become hard **dynamically**
  when the manual's catalogue procedures for the component differ by that variant (e.g. Coolant Pump — 3.7L /
  — 5.0L → engine required; Door Trim Panel → not).
- Unknown answers get manual-grounded hints (VIN 8th digit engine code, Section 100-01).
- LangGraph: 27 nodes per spec + insufficient-evidence exit; the readiness conditional edge is the only path to
  retrieval, and each retrieval node also refuses to run without READY (defense in depth).
- SQLite session store behind a repository protocol: persistence, isolation, 20-message cap, sliding expiry, purge.

**Tests:** extraction, readiness (spec example, hard-gate vs score, dynamic variants, inference confirmation,
scope, conflicts, corrections, task change, unsure, safety, follow-up), session store (restart, isolation,
truncation, expiry), graph (retrieval mocked to fail on every non-READY path).

## Phase 6 — Hybrid retrieval and visual retrieval — DONE (2026-11-25)

**Modules:** `retrieval/{pipeline,applicability_filter,fusion,reranker,context_expansion,visual_retrieval}.py`,
`services/manual_repository.py`, `config/synonyms.json`.

- Query construction keeps user wording, adds manual synonyms (water pump → coolant pump), DTCs and identifiers;
  vehicle variants are filters, not query terms (adding "5.0L" biased ranking toward engine tables — fixed).
- Applicability: Chroma `where` + BM25 mask exclude incompatible variants; non-specific chunks kept and labelled.
- Weighted RRF (k=60) with content-hash dedup → cross-encoder `ms-marco-MiniLM-L-6-v2` scored with two queries
  (synonym-rewritten message + "<manual term> <task>"), because manual terminology raised the correct chunk's
  logit from −2.9 to +8.4 → small task-type / follow-up / step boosts.
- Expansion: whole procedure for procedural tasks if it fits the budget, else preamble + hits + neighbours +
  referenced steps; other procedures unexpanded; manual order preserved.
- Visuals only from ingestion-time associations; main procedure first; only images tied to cited evidence are shown.
- Variant ambiguity (two compatible procedures differing only by an unspecified variant) → clarification.

## Phase 7 — LangGraph and generation — DONE (2026-11-25)

**Modules:** `generation/{llm_client,prompts,answer_generator,response_validator}.py`, `api/*`, `services/container.py`, `main.py`.

- Groq `openai/gpt-oss-120b` (reasoning effort configurable, default low). **The supplied key is on Groq's
  on_demand tier: 8,000 tokens/minute per model** (prompt + max output). Defaults were sized for it
  (evidence 4,500 tokens, output 1,800) and the prompt is trimmed (other procedures first) to fit.
- Validator: normalises Unicode hyphens/spaces; accepts `[ S2 ]`/bare `S2` markers for real ids; removes unknown
  ids; every value+unit and part/tool/DTC identifier must occur in evidence (including spec-table columns whose
  units are in the header); uncited technical content → rejected; dropped WARNING/CAUTION re-inserted verbatim;
  uncited claims about the vehicle's equipment are stripped. Rejected/unavailable → verbatim excerpt fallback.
- Citations are built from chunk metadata (section, procedure id, subsection, PDF pages, printed "Page X of N").
- API: `/api/chat`, `/api/sessions[/{id}][/context]`, `/api/admin/{ingest,reindex,ingestion/status}` (X-Admin-Key),
  `/api/manual-assets/{asset_id}` (id-validated, no paths), `/api/health`; rate limiting; per-session locks.

**Live checks against the real manual (Groq):**
- Spec example: "replace the water pump" → year? → engine? (manual has 3.7L/5.0L coolant pumps) → "5.0L V8" →
  READY → Coolant Pump — 5.0L removal + installation, 3.7L excluded, cited PDF pp. 1659–1661.
- "What torque do the pump bolts need?" → follow-up on the active procedure → Stage 1 20 Nm (177 lb-in),
  Stage 2 +60° [Installation, p. 1660–1661].
- "what is step 6 of the removal?" → "Remove the thermostat housing…" [p. 1660].
- Front caliper bolt torque → 115 Nm (85 lb-ft) for both listed caliper variants [Front Disc Brake specs, p. 430].

**Tests:** 104 passing (unit + integration: synthetic manual ingested end-to-end through Chroma, FastAPI flow,
diagram served, follow-up re-retrieval, honest insufficient-evidence, restart persistence, admin auth, path safety).

## Phase 8 — Frontend — DONE (2026-11-25)

**Modules:** `frontend/src/{App.tsx, hooks/useChat.ts, services/api.ts, types/api.ts}`,
`components/{MessageBubble,Markdown,markdownParse,DiagramGallery,SourcesList,SpecTables,VehiclePanel,Composer}`.

- Vehicle and task panel shows each slot with provenance badge (confirmed / provided / needs confirmation /
  unknown), marks required-but-missing fields, and has an editor (options from the manual profile via
  `/api/health`) that calls `/api/sessions/{id}/context` → readiness re-evaluated, active procedure invalidated.
- Readiness indicator (status + score bar + still-needed fields); "Checking details…" vs "Searching the manual…".
- Distinct message modes: clarifying question (with option chips), confirmation, safety escalation,
  out of scope, answer. No sources/diagrams are shown while context is being collected.
- Safe Markdown renderer (React elements only): manual step numbers preserved, WARNING/CAUTION as red alerts,
  NOTICE/NOTE as amber notes, pipe tables, `[S#]` citation chips that highlight the source entry.
- Original diagrams with captions, step, PDF + printed page and OCR-read figure id; lightbox with zoom.
- Sources list with section, procedure, PDF pages, printed "Page X of N", procedure reference; collapsible tables;
  copy answer; error bubble with retry; session id in localStorage (guarded) with restore on reload.

**Tests:** Vitest + Testing Library — 13 passing (renderer, XSS safety, modes, diagram lightbox, full
clarification→answer flow with session id reuse, error+retry, vehicle edit → context API). `tsc -b`, oxlint
and `vite build` clean.
**Smoke test:** uvicorn + Vite (proxy) → real chat turn returned a cited Thermostat — 3.7L answer and the
diagram asset streamed through `/api/manual-assets/{id}` (200, image/png). Screenshot verified the layout.
Note: port 5173 was occupied by another local project, so the dev server was run on 5174 (`npx vite --port 5174`).

## Phase 9 — Evaluation — DONE (2026-11-26)

**Modules:** `backend/app/evaluation/{metrics,datasets,baselines,runner}.py`, `scripts/evaluate_rag.py`,
`backend/tests/evaluation/data/{retrieval_cases_draft,retrieval_cases,conversation_cases}.json`.

- 48 retrieval cases resolved against the manual. Each one's key fact is verified to occur in the expected procedure.
- Full system: Recall@5 0.958, MRR 0.788, nDCG@5 0.823, 0 applicability violations, completeness 0.92,
  figure P/R 0.73. Ablations: no_rerank 0.778 MRR, dense_only 0.759, bm25_only 0.663, no_filter 0.701 (+1.02 violations@5).
- Reranker experiment (tuned on odd cases, checked on held-out even cases): the cross-encoder alone hurt, so it
  was replaced by RRF(hybrid, rerank_mean, k=10). The task-type boost was rejected.
- Chunking A/B/C: fixed-size MRR 0.671, recursive 0.625, structure-based 0.788. Baselines show ~1.3 wrong-variant
  chunks in the top 5.
- Context gate: 16 conversations / 29 turns. Decision accuracy 1.0, slot accuracy 1.0, 0 premature retrievals.
- Faithfulness proxy (live Groq, n=8): 7/7 answers passed the validator, 1 Groq failure (served the excerpt fallback).
- Bugs found by evaluation and fixed: mixed-qualifier applicability, metadata-blind index manifest, engine terms
  in the query, raw reranker, validator false positives (Unicode hyphens, wrapped identifiers, vehicle facts).
- `chunking --rebuild` now removes the throwaway `data/indexes/eval_chroma` dir. Opening a second Chroma
  client with different settings raised "instance already exists".

Report: `docs/evaluation_report.md`.

## Phase 10 — Hardening & documentation — DONE (2026-11-26)

- Docs: `architecture.md`, `safety_and_scope.md`, `operations.md`, `evaluation_report.md`, README rewrite.
- Ruff clean. Removed dead code. SQLite `-wal/-shm` files untracked and ignored.
- mypy: 146 → 49 errors. **Known issue:** the remaining errors are Optional narrowing and the untyped Chroma client
  API. They aren't runtime defects (all tests pass), but they're left for a typing pass.
- Disk: the machine ran out of space during evaluation (145 MB free). Only this project's eval index and pytest
  temp files were deleted. The user freed the rest.

**Tests:** backend 110 passing; frontend 13 passing; production build OK.

## Technician stress test — DONE (2026-11-26)

Ten hard live conversations (`scripts/run_technician_scenarios.py`); details, before/after table and
remaining issues in `docs/evaluation_report.md` §8. 13 defects fixed (variant handling for caliper type/side,
main-procedure choice, spec-table evidence, safety acknowledgement + LLM hazard echo, trim/engine conflict loop,
range values in the validator, follow-up vocabulary, look-alike variant questions, focused fallback, Groq daily
quota fallback model). Retrieval: completeness 0.97, citation page 0.75, figures 0.84 (from 0.92/0.65/0.73);
Recall/MRR unchanged; gate 1.0 with 0 premature retrievals. Tests: 120 passing.
