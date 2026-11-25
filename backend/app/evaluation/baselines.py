"""Comparison baselines: fixed-size and recursive chunking over the same parsed text.

Both use the same cleaned page text as the structure-based chunker (boilerplate and
watermark already removed) so the comparison isolates the chunking strategy. Baseline
chunks carry only page numbers — no structure, procedure ids or applicability metadata.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from app.embeddings.tokenizer import TokenCounter
from app.ingestion.chunker import render_element
from app.schemas.chunk import Applicability, Chunk
from app.schemas.document import PageRecord


def page_texts(pages: Iterable[PageRecord]) -> list[tuple[int, str]]:
    out = []
    for p in pages:
        text = "\n".join(t for t in (render_element(e) for e in p.elements) if t and not t.startswith("[Figure"))
        if text.strip():
            out.append((p.pdf_page_index, text))
    return out


def _mk(strategy: str, i: int, text: str, p0: int, p1: int, counter: TokenCounter) -> Chunk:
    return Chunk(
        chunk_id=f"{strategy}_{i:05d}",
        document_id="baseline",
        document_title="baseline",
        source_hash="baseline",
        parent_id=f"{strategy}_{i:05d}",
        procedure_id=f"{strategy}_{i:05d}",
        source_procedure_id="",
        chunk_type="baseline",
        sequence_index=i,
        split_reason=strategy,
        page_start=p0 + 1,
        page_end=p1 + 1,
        pdf_page_start=p0,
        pdf_page_end=p1,
        applicability=Applicability(),
        context_header="",
        content=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
        token_count=counter.count(text),
    )


def fixed_size_chunks(pages: list[tuple[int, str]], counter: TokenCounter, size: int = 400) -> list[Chunk]:
    """Baseline A: concatenate all text and cut every `size` tokens, ignoring structure."""
    words: list[tuple[str, int]] = [(w, pno) for pno, text in pages for w in text.split()]
    chunks, buf, i = [], [], 0
    approx = max(1, int(size / 1.3))  # words per chunk ~ size tokens (exact count recorded below)
    for w in words:
        buf.append(w)
        if len(buf) >= approx:
            chunks.append(_mk("fixed", i, " ".join(x for x, _ in buf), buf[0][1], buf[-1][1], counter))
            buf, i = [], i + 1
    if buf:
        chunks.append(_mk("fixed", i, " ".join(x for x, _ in buf), buf[0][1], buf[-1][1], counter))
    return chunks


def recursive_chunks(
    pages: list[tuple[int, str]], counter: TokenCounter, size: int = 400, overlap: int = 50
) -> list[Chunk]:
    """Baseline B: recursive splitting (paragraph -> line -> sentence) packed to `size` tokens with overlap."""
    pieces: list[tuple[str, int]] = []
    for pno, text in pages:
        for para in re.split(r"\n{2,}", text):
            for line in para.split("\n"):
                if counter.count(line) <= size:
                    pieces.append((line, pno))
                else:
                    for sent in (x for x in re.split(r"(?<=[.!?])\s+", line) if x):
                        if counter.count(sent) <= size:
                            pieces.append((sent, pno))
                            continue
                        words = sent.split()  # last resort: word windows
                        step = max(1, int(size / 1.3))
                        pieces.extend((" ".join(words[i : i + step]), pno) for i in range(0, len(words), step))
    chunks, cur, cur_tok, i = [], [], 0, 0
    for text, pno in pieces:
        t = counter.count(text)
        if cur and cur_tok + t > size:
            chunks.append(_mk("recursive", i, "\n".join(x for x, _ in cur), cur[0][1], cur[-1][1], counter))
            i += 1
            keep, kept = [], 0
            for x in reversed(cur):  # overlap tail
                kept += counter.count(x[0])
                if kept > overlap:
                    break
                keep.insert(0, x)
            cur, cur_tok = keep, sum(counter.count(x[0]) for x in keep)
        cur.append((text, pno))
        cur_tok += t
    if cur:
        chunks.append(_mk("recursive", i, "\n".join(x for x, _ in cur), cur[0][1], cur[-1][1], counter))
    return chunks
