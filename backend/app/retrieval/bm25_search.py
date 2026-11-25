"""BM25 keyword retrieval with a tokenizer that preserves technical identifiers.

Part numbers (W710141), DTCs (B11E8:16), tool numbers (303-D087), displacements (5.0L)
and units (lb-ft) are kept as single tokens; sub-parts are also indexed so partial
queries still match.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.schemas.chunk import Chunk

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.:/\-][A-Za-z0-9]+)*")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "my",
        "me",
        "we",
        "our",
        "you",
        "your",
        "do",
        "does",
        "how",
        "what",
        "which",
    ]
)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        out.append(tok)
        if any(ch in tok for ch in ".:/-"):
            out.extend(p for p in re.split(r"[.:/\-]", tok) if p and p not in STOPWORDS and len(p) > 1)
    return out


def bm25_text(c: Chunk) -> str:
    return "\n".join(filter(None, [c.context_header, c.content, c.figure_ocr]))


@dataclass
class BM25Hit:
    chunk_id: str
    score: float


class BM25Index:
    def __init__(self, chunk_ids: list[str], tokenized: list[list[str]]) -> None:
        self.chunk_ids = chunk_ids
        self._pos = {cid: i for i, cid in enumerate(chunk_ids)}
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    @classmethod
    def from_chunks(cls, chunks: Iterable[Chunk]) -> BM25Index:
        ids, toks = [], []
        for c in chunks:
            ids.append(c.chunk_id)
            toks.append(tokenize(bm25_text(c)))
        return cls(ids, toks)

    def save(self, path: Path, tokenized: list[list[str]]) -> None:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for cid, toks in zip(self.chunk_ids, tokenized, strict=True):
                fh.write(json.dumps({"chunk_id": cid, "tokens": toks}) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        ids, toks = [], []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                ids.append(rec["chunk_id"])
                toks.append(rec["tokens"])
        return cls(ids, toks)

    def search(self, query: str, n: int, allow: Callable[[str], bool] | None = None) -> list[BM25Hit]:
        if self.bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        order = np.argsort(-scores)
        hits: list[BM25Hit] = []
        for i in order:
            if scores[i] <= 0:
                break
            cid = self.chunk_ids[i]
            if allow is not None and not allow(cid):
                continue
            hits.append(BM25Hit(cid, float(scores[i])))
            if len(hits) >= n:
                break
        return hits
