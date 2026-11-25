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
