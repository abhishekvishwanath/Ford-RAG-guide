"""Read-only access to ingested artifacts: chunks, procedure parents, figures, tables.

Loaded once into memory (a few tens of MB for the full manual); keyed for O(1) lookups.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.schemas.chunk import Chunk, FigureRecord, ProcedureRecord
from app.schemas.document import ElementKind, ProcedureDocument, TableData


class ManualRepository:
    def __init__(self, chunks: list[Chunk], procedures: list[ProcedureRecord], figures: list[FigureRecord],
                 tables: dict[str, TableData] | None = None) -> None:  # fmt: skip
        self.chunks: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self.procedures: dict[str, ProcedureRecord] = {p.parent_id: p for p in procedures}
        self.figures: dict[str, FigureRecord] = {f.figure_id: f for f in figures}
        self.tables: dict[str, TableData] = tables or {}
        self.assets: dict[str, FigureRecord] = {}
        for f in figures:
            self.assets.setdefault(f.asset_id, f)
        self._by_parent: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            self._by_parent[c.parent_id].append(c)
        for lst in self._by_parent.values():
            lst.sort(key=lambda c: c.sequence_index)

    @classmethod
    def load(cls, processed_dir: Path) -> ManualRepository:
        def read(name: str, model):  # type: ignore[no-untyped-def]
            path = processed_dir / name
            if not path.exists():
                return []
            with path.open(encoding="utf-8") as fh:
                return [model.model_validate_json(line) for line in fh]

        tables: dict[str, TableData] = {}
        docs_path = processed_dir / "documents.jsonl"
        if docs_path.exists():
            with docs_path.open(encoding="utf-8") as fh:
                for line in fh:
                    if '"table"' not in line:
                        continue
                    for el in ProcedureDocument.model_validate_json(line).elements:
                        if el.kind == ElementKind.TABLE and el.table is not None:
                            tables[el.table.table_id] = el.table
        return cls(read("chunks.jsonl", Chunk), read("procedures.jsonl", ProcedureRecord),
                   read("figures.jsonl", FigureRecord), tables)  # fmt: skip

    @property
    def ready(self) -> bool:
        return bool(self.chunks)

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def parent_chunks(self, parent_id: str) -> list[Chunk]:
        return self._by_parent.get(parent_id, [])

    def figures_for_chunk(self, chunk: Chunk) -> list[FigureRecord]:
        return [self.figures[f] for f in chunk.figure_ids if f in self.figures]

    def stats(self) -> dict:
        return {"chunks": len(self.chunks), "procedures": len(self.procedures), "figures": len(self.figures),
                "tables": len(self.tables)}  # fmt: skip


def dump_json(obj: object) -> str:
    return json.dumps(obj, default=str)
