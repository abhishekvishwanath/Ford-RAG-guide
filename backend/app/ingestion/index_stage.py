"""Index stage: chunks -> ChromaDB (dense, persistent) + BM25 corpus, incrementally.

A manifest maps chunk_id -> content_hash for what is currently in the vector store.
Re-running ingestion only embeds new/changed chunks and deletes stale ones
(the diagram's "get_or_create_collection - avoid full rebuild"). Changing the
embedding model forces a full rebuild of the collection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from app.embeddings.embedding_service import Embedder, SentenceTransformerEmbedder
from app.embeddings.tokenizer import get_token_counter
from app.retrieval.bm25_search import BM25Index, bm25_text, tokenize
from app.schemas.chunk import Chunk
from app.vectorstore.chroma_store import ChromaStore

if TYPE_CHECKING:
    from app.ingestion.ingestion_service import IngestionService


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as fh:
        return [Chunk.model_validate_json(line) for line in fh]


def plan_index_changes(chunks: list[Chunk], manifest: dict[str, str]) -> tuple[list[Chunk], list[str], list[str]]:
    """Return (to_upsert, stale_ids_to_delete, changed_ids) given the current manifest."""
    current = {c.chunk_id: c.content_hash for c in chunks}
    upsert = [c for c in chunks if manifest.get(c.chunk_id) != c.content_hash]
    changed = [c.chunk_id for c in upsert if c.chunk_id in manifest]
    stale = [cid for cid in manifest if cid not in current]
    return upsert, stale, changed


def run_index_stage(service: IngestionService, embedder: Embedder | None = None) -> dict:
    s = service.s
    chunks = load_chunks(s.processed_dir / "chunks.jsonl")
    embedder = embedder or SentenceTransformerEmbedder(s.embedding_model, s.embedding_batch_size)
    counter = get_token_counter(s.embedding_model)
    manifest_path = s.indexes_dir / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    store = ChromaStore(s.chroma_persist_directory, s.chroma_collection_name, embedder)
    rebuilt = False
    if manifest.get("embedding_model") not in (None, embedder.model_name) or (
        store.embedding_model() not in (None, embedder.model_name)
    ):
        store.client.delete_collection(s.chroma_collection_name)
        store = ChromaStore(s.chroma_persist_directory, s.chroma_collection_name, embedder)
        manifest, rebuilt = {}, True
    if manifest.get("chunks") and store.count() == 0:  # manifest without data (e.g. store wiped)
        manifest = {}

    upsert, stale, changed = plan_index_changes(chunks, manifest.get("chunks", {}))
    if stale or changed:
        store.delete_chunks(stale + changed)
    windows = store.upsert_chunks(upsert, counter) if upsert else 0

    tokenized = [tokenize(bm25_text(c)) for c in chunks]
    bm25 = BM25Index([c.chunk_id for c in chunks], tokenized)
    s.indexes_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(s.indexes_dir / "bm25_corpus.jsonl", tokenized)

    new_manifest = {
        "embedding_model": embedder.model_name,
        "source_hash": chunks[0].source_hash if chunks else None,
        "chunks": {c.chunk_id: c.content_hash for c in chunks},
    }
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(new_manifest))
    tmp.replace(manifest_path)
    return {
        "chunks_total": len(chunks),
        "chunks_embedded": len(upsert),
        "chunks_unchanged": len(chunks) - len(upsert),
        "stale_deleted": len(stale),
        "windows_written": windows,
        "vector_count": store.count(),
        "full_rebuild": rebuilt,
        "embedding_model": embedder.model_name,
    }
