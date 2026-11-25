# Mustang Workshop Assistant — working notes for Claude

Conversational RAG assistant for qualified technicians over the **2014** Ford Mustang
Workshop Manual (5,007-page PDF in `data/raw/`, not in git). Build follows a 10-phase spec;
progress, commands and test results live in `docs/implementation_log.md` — read it first
and resume from the last completed phase.

## Non-negotiables
- The manual is the only technical source. Never invent steps, torque values, specs, tools, warnings.
- **No retrieval before the readiness gate returns READY.** Tests must prove retrieval is not called otherwise.
- Never mix model years / engines / transmissions / body variants. Scope is MY2014 only (user decision);
  supported years come from `data/indexes/manual_profile.json`, not code.
- Original diagrams only; citations are built from chunk metadata, never from LLM text.
- Preserve warnings before the steps they govern.
- Strip the footer e-mail watermark from all extracted text (privacy).

## Decisions
- LLM: **Groq** (`openai/gpt-oss-120b` generation, `openai/gpt-oss-20b` extraction), configured via `.env`.
- Embeddings: `all-MiniLM-L6-v2` (256-token max) → chunks embedded as multiple windows.
- Procedure boundary = Ford SEB document id from page footers; bookmarks give hierarchy labels.

## Commands
- Backend tests: `.venv/bin/pytest -q` (from repo root)
- Lint: `.venv/bin/ruff check backend scripts`
- Frontend: `cd frontend && npm test && npm run build`
- Push to GitHub at the end of every phase.
