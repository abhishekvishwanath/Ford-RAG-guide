# Mustang Workshop Assistant

A conversational, safety-conscious workshop assistant for technicians working on the
**2014 Ford Mustang**. It gathers the vehicle and repair context first, then retrieves
the exact procedure, specifications, warnings and original diagrams from the workshop
manual, with traceable citations.

> Status: under active development — see [docs/implementation_log.md](docs/implementation_log.md).

## Architecture (summary)

```
Ingestion:  PDF → layout parse (+OCR) → structure detection → hierarchical chunks
            → metadata/applicability → all-MiniLM-L6-v2 → ChromaDB (persistent) + BM25 index
Chat:       message → session state → context extraction → readiness gate
            → (ask follow-up | confirm | safety | out-of-scope | READY)
            → query → dense + BM25 → RRF → rerank → procedure expansion → figures/tables
            → grounded prompt → LLM (Groq) → validated answer + citations + original diagrams
```

## Requirements

- Python 3.11, Node 20+, Tesseract 5 (`brew install tesseract`)
- The manual PDF placed at `data/raw/` (not distributed with this repo)
- A Groq API key

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
cp .env.example .env        # fill GROQ_API_KEY and ADMIN_API_KEY
cd frontend && npm install
```

## Run

```bash
# backend
cd backend && ../.venv/bin/uvicorn app.main:app --reload --port 8000
# frontend
cd frontend && npm run dev   # http://localhost:5173
```

## Tests

```bash
.venv/bin/pytest -q
cd frontend && npm test
```

## Docs

- [Source manual assessment](docs/source_manual_assessment.md)
- [Implementation log](docs/implementation_log.md)
