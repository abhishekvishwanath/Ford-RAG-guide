"""Weighted Reciprocal Rank Fusion with chunk/content de-duplication."""

from __future__ import annotations

from app.schemas.retrieval import Candidate


def rrf_fuse(
    dense: list[Candidate],
    bm25: list[Candidate],
    k: int = 60,
    w_dense: float = 1.0,
    w_bm25: float = 1.0,
    content_hash: dict[str, str] | None = None,
) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for c in dense:
        m = merged.setdefault(c.chunk_id, Candidate(chunk_id=c.chunk_id))
        m.dense_score, m.dense_rank = c.dense_score, c.dense_rank
        m.rrf_score += w_dense / (k + (c.dense_rank or 1))
    for c in bm25:
        m = merged.setdefault(c.chunk_id, Candidate(chunk_id=c.chunk_id))
        m.bm25_score, m.bm25_rank = c.bm25_score, c.bm25_rank
        m.rrf_score += w_bm25 / (k + (c.bm25_rank or 1))
    ranked = sorted(merged.values(), key=lambda c: (-c.rrf_score, c.chunk_id))
    if content_hash:
        seen: set[str] = set()
        unique = []
        for c in ranked:
            h = content_hash.get(c.chunk_id, c.chunk_id)
            if h in seen:
                continue
            seen.add(h)
            unique.append(c)
        ranked = unique
    return ranked
