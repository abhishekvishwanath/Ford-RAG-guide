"""Protected ingestion/administration endpoints."""

from __future__ import annotations

import threading

from fastapi import APIRouter, Depends

from app.api.dependencies import get_container, require_admin
from app.core.exceptions import AppError
from app.schemas.api import IngestRequest
from app.services.container import Container

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])
_job_lock = threading.Lock()


class IngestionBusy(AppError):
    status_code = 409
    code = "ingestion_running"


def _start(c: Container, stages: list[str] | None, page_range: tuple[int, int] | None) -> dict:
    if not _job_lock.acquire(blocking=False):
        raise IngestionBusy("An ingestion job is already running.")

    def job() -> None:
        try:
            c.ingestion.run(stages, page_range)
            c.reload_indexes()
        finally:
            _job_lock.release()

    threading.Thread(target=job, name="ingestion", daemon=True).start()
    return {"accepted": True, "stages": stages or "all", "page_range": page_range}


@router.post("/ingest", status_code=202)
def ingest(body: IngestRequest, c: Container = Depends(get_container)) -> dict:
    c.ingestion.validate_source()
    return _start(c, body.stages, body.page_range)


@router.post("/reindex", status_code=202)
def reindex(c: Container = Depends(get_container)) -> dict:
    return _start(c, ["chunk", "index"], None)


@router.get("/ingestion/status")
def ingestion_status(c: Container = Depends(get_container)) -> dict:
    status = c.ingestion.read_status()
    status["job_running"] = _job_lock.locked()
    status["index_ready"] = c.index_ready
    status["manual"] = c.manual.stats()
    return status
