# Mustang Workshop Assistant

A conversational, safety-conscious workshop assistant for technicians working on the
**2014 Ford Mustang**. It gathers the vehicle and repair context first — and refuses to search
until the manual's variant-relevant details are known — then retrieves the exact procedure,
specifications, warnings and **original diagrams** from the 5,007-page workshop manual, with
citations to section, procedure, PDF page and printed page.

## What it does

* **Intake before retrieval.** A readiness gate (LangGraph) asks one targeted question at a time.
  Engine / transmission / body style are only asked when the manual actually has different
  procedures for that job (e.g. *Coolant Pump — 3.7L* vs *— 5.0L*). Trim-name guesses ("GT" → 5.0L)
  are confirmed; conflicts are resolved; other years/vehicles are out of scope; immediate hazards
  (fuel leak, fire, unstable lift…) get a stop-work response first.
* **Structure-aware ingestion.** Procedures are split by the manual's own boundaries; warnings stay
  before the steps they govern; spec/legend tables are reconstructed; diagrams are extracted and
  linked to steps; callouts are OCR'd for search.
* **Hybrid retrieval with applicability filtering.** MiniLM multi-window vectors in ChromaDB + BM25,
  RRF, cross-encoder rank fusion, procedure expansion; incompatible variants are excluded.
* **Grounded answers.** Groq `gpt-oss-120b` answers only from evidence; a validator checks every value,
  part/tool number and citation, re-inserts dropped warnings, and falls back to verbatim excerpts.

## Quick start

```bash
brew install tesseract
python3.11 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
cp .env.example .env                     # GROQ_API_KEY, ADMIN_API_KEY
cp <manual>.pdf data/raw/                # the PDF is not distributed with this repo
.venv/bin/python scripts/ingest_manual.py            # ~10 min, resumable
cd backend && ../.venv/bin/uvicorn app.main:app --port 8000
cd frontend && npm install && npm run dev            # http://localhost:5173
```

## Results (measured, see `docs/evaluation_report.md`)

| Measure | Result |
|---|---|
| Retrieval Recall@5 / MRR / nDCG@5 (48 verified cases) | 0.958 / 0.788 / 0.823 |
| Procedure completeness / diagram precision | 0.97 / 0.84 |
| Live technician stress test (10 hard conversations) | after fixes: 8 fully answered; 2 ended in verbatim manual excerpts because a model answer was rejected (report §8) |
| Wrong-variant chunks in top 5 | 0.00 (1.02 without the applicability filter) |
| Structure-based vs fixed-size vs recursive chunking (MRR) | 0.788 vs 0.671 vs 0.625 |
| Readiness gate decision accuracy (29 turns) / premature retrievals | 1.00 / 0 |
| Live answers passing the value/citation validator | 7 / 7 (1 Groq call failed → excerpt fallback) |

Faithfulness is checked lexically (values, part numbers and citations must appear in the evidence). Semantic
correctness was not judged.

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | pipeline, chunking, metadata schema, readiness gate, retrieval, generation |
| [docs/source_manual_assessment.md](docs/source_manual_assessment.md) | what the PDF actually contains (MY2014 only) |
| [docs/evaluation_report.md](docs/evaluation_report.md) | benchmarks, chunking comparison, ablations, limitations |
| [docs/safety_and_scope.md](docs/safety_and_scope.md) | safety behaviour, scope, privacy |
| [docs/operations.md](docs/operations.md) | setup, ingestion, configuration, troubleshooting |
| [docs/implementation_log.md](docs/implementation_log.md) | phase-by-phase log, commands, test results |

## Tests

```bash
.venv/bin/pytest -q                          # backend unit + integration
cd frontend && npm test && npm run build
.venv/bin/python scripts/evaluate_rag.py retrieval   # and: chunking | context | faithfulness
```
