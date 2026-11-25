# Operations

## Setup from a clean checkout

```bash
brew install tesseract                      # OCR (pages without a text layer, diagram callouts)
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
cp .env.example .env                        # set GROQ_API_KEY and ADMIN_API_KEY
cp /path/to/319-Ford-Mustang-Workshop-Service-Manual-_-2004-2014.pdf data/raw/
cd frontend && npm install && cd ..
```

## Ingestion

```bash
.venv/bin/python scripts/ingest_manual.py --dry-run          # validate + inspect, writes nothing
.venv/bin/python scripts/ingest_manual.py                    # all stages (~10 min on 8 cores)
.venv/bin/python scripts/ingest_manual.py --stages chunk index   # re-chunk / re-index only
.venv/bin/python scripts/ingest_manual.py --pages 68 140     # page range (0-based, end exclusive)
```

| Stage | Output | Typical time (M-series, 8 workers) |
|---|---|---|
| parse | `data/processed/pages/*.jsonl`, `data/extracted_images/` | 7–9 min for 5,007 pages |
| structure | `data/processed/documents.jsonl`, `document_metadata.json` | < 1 s |
| figures | `data/processed/figure_ocr.jsonl` | ~2 min first time, cached after |
| chunk | `chunks.jsonl`, `procedures.jsonl`, `figures.jsonl`, `chunk_report.json`, `data/indexes/manual_profile.json` | ~6 s |
| index | `data/chroma_db/`, `data/indexes/bm25_corpus.jsonl`, `index_manifest.json` | ~1 min full, seconds incremental |

* **Resume / cancel:** Ctrl-C or `touch data/processed/CANCEL`; re-running resumes from
  `parse_checkpoint.json` (keyed by PDF SHA-256 and parser version).
* **Status:** `data/processed/ingestion_status.json`, or `GET /api/admin/ingestion/status` with `X-Admin-Key`.
* **Via API:** `POST /api/admin/ingest` (`{"stages": ["chunk","index"]}`) runs in a background thread and
  hot-reloads indexes; `POST /api/admin/reindex` = chunk + index.
* Changing `PARSER_VERSION`, the PDF, or chunking settings and re-running only redoes affected work;
  changing `EMBEDDING_MODEL` rebuilds the vector collection.

## Running

```bash
cd backend && ../.venv/bin/uvicorn app.main:app --port 8000      # API + docs at /docs
cd frontend && npm run dev                                       # http://localhost:5173 (proxy /api → :8000)
```

If port 5173 is taken: `npx vite --port 5174`.

## Configuration highlights (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL` / `LLM_EXTRACTION_MODEL` | `openai/gpt-oss-120b` / `openai/gpt-oss-20b` | Groq |
| `LLM_TPM_LIMIT` | 8000 | Groq on_demand tier cap; raise with a higher tier |
| `EVIDENCE_TOKEN_BUDGET` / `LLM_MAX_OUTPUT_TOKENS` | 4500 / 1800 | sized for 8k TPM |
| `LLM_REASONING_EFFORT` | low | low / medium / high |
| `READINESS_THRESHOLD`, `MIN_REQUIRED_SLOT_CONFIDENCE` | 0.85, 0.80 | gate tuning |
| `RETRIEVAL_TOP_K`, `DENSE_CANDIDATES`, `BM25_CANDIDATES`, `RRF_K` | 5, 30, 30, 60 | |
| `RERANKER_ENABLED`, `RERANKER_MODEL`, `RERANK_FUSION_K` | true, ms-marco-MiniLM-L-6-v2, 10 | |
| `APPLICABILITY_STRICTNESS` | strict | |
| `SESSION_MESSAGE_LIMIT`, `SESSION_EXPIRY_DAYS` | 20, 7 | |

Behavioural configuration lives in `config/` (vocabulary, synonyms, readiness policy).

## Tests and evaluation

```bash
.venv/bin/pytest -q                         # backend unit + integration
cd frontend && npm test && npm run build    # frontend
.venv/bin/python scripts/evaluate_rag.py resolve|retrieval|chunking|context|faithfulness
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| UI banner "manual index is not built" | run ingestion; check `/api/health` → `index_ready` |
| Answers show "verbatim manual excerpts" | LLM unavailable, rate-limited (8k TPM) or answer failed validation; see `validation.warnings` in the response |
| `413 Request too large … tokens per minute` in logs | lower `EVIDENCE_TOKEN_BUDGET`/`LLM_MAX_OUTPUT_TOKENS` or upgrade the Groq tier |
| Slow first answer | embedding + reranker models load on first use; the server warms them up at startup |
| `401` on admin endpoints | set `ADMIN_API_KEY` and send it as `X-Admin-Key` |
| Parsing restarts from zero | PDF hash or `PARSER_VERSION` changed (by design) |
