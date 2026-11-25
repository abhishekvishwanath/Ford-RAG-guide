# Architecture

The system follows the supplied two-phase RAG reference design (structure-based chunking →
metadata → `all-MiniLM-L6-v2` → persistent ChromaDB → incremental ingestion; BM25 + vector →
RRF → top-k → LangGraph → grounded LLM → cited answer or refusal), extended with a mandatory
conversational **readiness gate** before retrieval, vehicle **applicability filtering**, and
**original-diagram retrieval**.

```
INGESTION (scripts/ingest_manual.py, /api/admin/ingest)
 PDF ─► parse (8 workers, 25-page batches, checkpoints) ─► structure (procedures by source id)
     ─► figure OCR (unique images) ─► hierarchical chunks + metadata + manual profile
     ─► index: multi-window embeddings → ChromaDB (incremental manifest) + BM25 corpus

CONVERSATION (POST /api/chat → LangGraph, 27 nodes)
 load_session → append_user_message → extract_vehicle_context → extract_repair_context
 → update_context_slots → detect_conflicts → determine_required_context → evaluate_readiness
      ├─ NEEDS_INFORMATION  → ask_followup ──────────┐
      ├─ NEEDS_CONFIRMATION → request_confirmation ──┤
      ├─ SAFETY_ESCALATION  → handle_safety_escalation┤
      ├─ OUT_OF_SCOPE       → handle_out_of_scope ───┤
      └─ READY → build_retrieval_query → apply_applicability_filters → dense_retrieve
               → bm25_retrieve → rrf_fusion → rerank → expand_procedure_context
               → retrieve_visuals_and_tables → assess_evidence
                    ├─ insufficient → handle_insufficient_evidence ─┤
                    └─ sufficient → build_grounded_prompt → generate_answer
                                   → validate_answer → attach_citations ─┤
                                              update_session_state → return_response
```

## Repository layout

| Path | Responsibility |
|---|---|
| `backend/app/core` | typed settings (`.env`), JSON logging with VIN/e-mail masking, error handling |
| `backend/app/ingestion` | PDF parsing, OCR, tables, figures, structure, chunking, metadata, stages |
| `backend/app/embeddings`, `vectorstore` | MiniLM embedder + tokenizer, ChromaDB store |
| `backend/app/retrieval` | query building, applicability filter, BM25, RRF, reranker, expansion, visuals |
| `backend/app/conversation`, `memory` | slots/provenance, extraction, readiness, questions, safety, sessions |
| `backend/app/graph` | LangGraph state, nodes, edges, workflow |
| `backend/app/generation` | LLM client (Groq), prompts, generator, validator, citations |
| `backend/app/api`, `services` | FastAPI routes, dependency container, manual repository/profile |
| `backend/app/evaluation` | metrics, baselines, dataset resolution, evaluation suites |
| `config/` | applicability vocabulary, synonyms, readiness policy (no code changes needed to tune) |
| `frontend/` | React + TS + Tailwind workshop UI |

## Ingestion and structure

* **Procedure boundary** = the source procedure id printed in every page footer (941 ids); bookmark
  tree (1,140 entries) gives hierarchy labels and tolerates misplaced bookmarks.
* **Typography classifier** (no hardcoded headings): title (bold 12 pt), header block (italic 9 pt),
  sub-headings / applicability qualifiers (bold 10 pt; qualifier = vehicle-applicability phrasing),
  steps, sub-steps (indented renumbering), bullets (symbol-font glyphs), admonitions
  (WARNING/CAUTION/NOTICE/NOTE), pinpoint answer labels.
* **Sequence fixes:** an admonition that opens a step is moved before the step; the step absorbs its
  action text; lower-case continuations are joined across page breaks.
* **Tables:** ruled (line strategy, per-cell clipped re-extraction), un-ruled with bold header (text
  strategy), and a column-assignment fallback (columns from data cells, header split by word boxes);
  rows cut by a page break are re-joined; header-less continuation tables inherit the header.
* **Figures:** stored once per content hash; placement bbox kept; banners/glyphs filtered; tool icons
  distinguished; OCR (Tesseract psm 11) extracts callouts such as torque values and figure ids, always
  labelled as OCR.

## Chunking (hierarchy L1 group → L2 section → L3 procedure → L4 child chunk)

* ≤ 400 words (`HEADING_SPLIT_WORD_THRESHOLD`) → one chunk per procedure.
* Otherwise split at sub-section headings; small sections merge forward; oversize sections are packed
  from atomic units (step + its warnings, bullets and figures; table; qualifier block). Tables split by
  rows with header repeated; token fallback only for a single oversize unit. Target 550 / max 900 tokens
  (embedding-model tokenizer); header and repeated warnings are budgeted.
* Continuations repeat procedure context, the active qualifier and the section's WARNING/CAUTION preamble.
* `split_reason`, `parent_id`, `sequence_index`, `procedure_id`, `is_continuation`, `step_range` recorded.

## Metadata schema (chunk)

```json
{
  "chunk_id": "chk_SEB33A09_001", "document_id": "ford_mustang_workshop_manual",
  "document_title": "Mustang 2014 Workshop Manual", "source_hash": "sha256…", "manual_revision": null,
  "toc_path": ["3. Powertrain", "03 Engine", "303-03A Engine Cooling", "Coolant Pump — 5.0L"],
  "section_code": "303-03A", "section_name": "Engine Cooling",
  "parent_id": "proc_SEB33A09", "procedure_id": "SEB33A09", "source_procedure_id": "SEB33A09",
  "procedure_title": "Coolant Pump — 5.0L", "doc_type": "REMOVAL AND INSTALLATION", "subsection": "Removal",
  "chunk_type": "removal", "sequence_index": 1, "is_continuation": false, "split_reason": "heading",
  "step_range": [1, 11], "page_start": 1660, "page_end": 1660, "pdf_page_start": 1659, "pdf_page_end": 1659,
  "printed_page_start": "Page 2 of 3", "printed_page_end": "Page 2 of 3",
  "figure_ids": [], "table_ids": [],
  "applicability": {"model_years": [2014], "engines": ["5.0L"], "transmissions": [], "body_styles": [],
                    "variants": [], "equipment_conditions": [], "exclusions": [],
                    "sources": ["document", "title"], "raw_qualifiers": []},
  "flags": ["has_notice"], "context_header": "…", "content": "…", "figure_ocr": "",
  "content_hash": "…", "token_count": 247
}
```

* Applicability comes **only from structure** (section, title, qualifier lines), normalised through
  `config/applicability_vocabulary.json`. Empty list = not variant-specific (never "confirmed for all").
* A qualifier constrains a chunk only if *all* its content is under that qualifier; mixed chunks stay
  general with the qualifier lines inline. The procedure record keeps the union (drives the gate).
* ChromaDB metadata is scalar: `engine_any`, `eng_5_0l`, `trans_any`, `trans_automatic`, `body_any`, …
* Parent records, figure records (association method + confidence + OCR) and the **manual profile**
  (discovered years/variants + procedure catalogue) are written beside the chunks.

## Embeddings and indexing

`all-MiniLM-L6-v2` truncates at 256 word-pieces, so each chunk is embedded as ~190-token windows with
40-token overlap, each prefixed with section › procedure › subsection; dense hits aggregate per chunk by
max similarity. **Documented deviation** from one-vector-per-chunk; chunk sizes for the LLM are unchanged.
A manifest of `chunk_id → hash(text + filter metadata)` makes re-indexing incremental and catches
metadata-only changes; changing the embedding model rebuilds the collection.

## Readiness gate (see `config/readiness_policy.json`)

1. Unacknowledged immediate hazard → `SAFETY_ESCALATION`.
2. Other vehicle or uncovered model year/engine → `OUT_OF_SCOPE` (supported years come from the profile).
3. Conflicting values → `NEEDS_CONFIRMATION`.
4. Hard slots: `model_year`, `component` (or symptom/DTC for diagnosis), `task_type`, plus
   **dynamic variant slots**: engine / transmission / body become hard when the manual's catalogue
   procedures for the component differ by that variant. Missing → `NEEDS_INFORMATION`.
5. Inferred (e.g. "GT" → 5.0L) or low-confidence hard slots → `NEEDS_CONFIRMATION`.
6. Weighted score ≥ `READINESS_THRESHOLD` (0.85) → `READY`. A score never overrides a missing hard slot.

| Example | Decision |
|---|---|
| "replace the water pump" | NEEDS_INFORMATION → model year |
| "2014" | NEEDS_INFORMATION → engine (Coolant Pump — 3.7L / — 5.0L exist) |
| "5.0L V8" | READY (transmission not asked: not discriminating for this job) |
| "2014 Mustang GT, replace the spark plugs" | NEEDS_CONFIRMATION (engine inferred from "GT") |
| "2014 mustang, remove the door trim panel" | READY immediately (no variant differences) |
| "2010 mustang water pump" | OUT_OF_SCOPE (manual covers 2014 only) |
| "fuel is leaking from the rail" | SAFETY_ESCALATION |

## Retrieval

Query = user wording + manual synonyms (`config/synonyms.json`, e.g. water pump → coolant pump) +
DTCs/identifiers + task terms; vehicle variants are **filters, not query terms**. Dense (Chroma `where`)
and BM25 (masked) both apply the applicability filter → weighted RRF (k=60) with content-hash dedup →
cross-encoder scored with two queries and **rank-fused** with the hybrid order (k=10) — chosen by
evaluation (see `docs/evaluation_report.md`) → expansion (whole procedure for procedural tasks within a
4,500-token budget; else preamble + hits + neighbours + referenced steps) → figures/tables from
ingestion-time associations (main procedure first) → sufficiency (cross-encoder relevance gate, variant
ambiguity check).

## Generation and validation

Groq `openai/gpt-oss-120b` with a rules-first system prompt; evidence blocks carry section, procedure,
part, PDF + printed pages and an applicability note. The prompt is trimmed to the provider's per-request
token cap. The validator enforces citations, values/identifiers present in evidence (including spec-table
columns), re-inserts dropped WARNING/CAUTION verbatim and strips uncited claims about the vehicle's
equipment; failures fall back to verbatim manual excerpts. Citations are built from metadata only.

## Session memory

SQLite via SQLAlchemy behind a `SessionRepository` protocol (swap for PostgreSQL/Redis): slot state with
provenance, readiness, pending question, active procedure, recent evidence ids; last 20 messages; sliding
7-day expiry; purge on startup; per-session locks; no cross-session profiling.
