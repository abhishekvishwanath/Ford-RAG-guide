"""Registry for ingestion stages implemented outside the core service (chunking, indexing)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ingestion.ingestion_service import IngestionService, StageFn


def stage_functions(service: IngestionService) -> dict[str, StageFn]:
    from app.ingestion.chunk_stage import run_chunk_stage
    from app.ingestion.index_stage import run_index_stage

    return {"chunk": lambda: run_chunk_stage(service), "index": lambda: run_index_stage(service)}
