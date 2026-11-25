"""Persistent ChromaDB store with multi-window chunk embeddings.

all-MiniLM-L6-v2 truncates input at 256 word-pieces, so a 550-token chunk is stored as
several overlapping windows (each prefixed with a short context label). Search results
are aggregated back to the chunk by maximum similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.embeddings.embedding_service import Embedder
from app.embeddings.tokenizer import TokenCounter
from app.ingestion.metadata import flatten_for_vector_store
from app.schemas.chunk import Chunk

logger = logging.getLogger(__name__)

WINDOW_TOKENS = 190
WINDOW_OVERLAP = 40


@dataclass
class DenseHit:
    chunk_id: str
    score: float  # cosine similarity of best window
    window: int


def window_label(c: Chunk) -> str:
    parts = [c.section_name, c.procedure_title, c.subsection]
    return " > ".join(p for p in parts if p)


def chunk_windows(c: Chunk, counter: TokenCounter) -> list[str]:
    body = c.content + (f"\nDiagram text (OCR): {c.figure_ocr}" if c.figure_ocr else "")
    label = window_label(c)
    return [f"{label}\n{w}" for w in counter.split_by_tokens(body, WINDOW_TOKENS, WINDOW_OVERLAP)]


def chunk_metadata(c: Chunk) -> dict:
    meta: dict = {
        "chunk_id": c.chunk_id,
        "parent_id": c.parent_id,
        "procedure_id": c.procedure_id,
        "source_procedure_id": c.source_procedure_id,
        "document_id": c.document_id,
        "source_hash": c.source_hash,
        "content_hash": c.content_hash,
        "chunk_type": c.chunk_type,
        "section_code": c.section_code or "",
        "procedure_title": c.procedure_title or "",
        "page_start": c.page_start,
        "page_end": c.page_end,
    }
    meta.update(flatten_for_vector_store(c.applicability))
    return meta


class ChromaStore:
    def __init__(self, persist_dir: Path, collection_name: str, embedder: Embedder) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False)
        )
        self.embedder = embedder
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedder.model_name},
        )

    def count(self) -> int:
        return self.collection.count()

    def embedding_model(self) -> str | None:
        return (self.collection.metadata or {}).get("embedding_model")

    def upsert_chunks(self, chunks: list[Chunk], counter: TokenCounter, batch: int = 256) -> int:
        ids, docs, metas = [], [], []
        for c in chunks:
            meta = chunk_metadata(c)
            for i, w in enumerate(chunk_windows(c, counter)):
                ids.append(f"{c.chunk_id}#w{i}")
                docs.append(w)
                metas.append({**meta, "window": i})
        for i in range(0, len(ids), batch):
            vecs = self.embedder.embed(docs[i : i + batch])
            self.collection.upsert(
                ids=ids[i : i + batch], documents=docs[i : i + batch], metadatas=metas[i : i + batch], embeddings=vecs
            )
        return len(ids)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        for i in range(0, len(chunk_ids), 500):
            self.collection.delete(where={"chunk_id": {"$in": chunk_ids[i : i + 500]}})

    def search(self, query: str, n_chunks: int, where: dict | None = None) -> list[DenseHit]:
        if self.count() == 0:
            return []
        vec = self.embedder.embed([query])[0]
        n_windows = min(self.count(), n_chunks * 4)
        res = self.collection.query(
            query_embeddings=[vec], n_results=n_windows, where=where or None, include=["metadatas", "distances"]
        )
        best: dict[str, DenseHit] = {}
        for meta, dist in zip(res["metadatas"][0], res["distances"][0], strict=True):
            cid = str(meta["chunk_id"])
            score = 1.0 - float(dist)
            if cid not in best or score > best[cid].score:
                best[cid] = DenseHit(cid, score, int(meta.get("window", 0)))
        return sorted(best.values(), key=lambda h: -h.score)[:n_chunks]
