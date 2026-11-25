"""Application exception hierarchy and FastAPI handlers."""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import log_event

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ValidationFailedError(AppError):
    status_code = 422
    code = "validation_failed"


class AuthError(AppError):
    status_code = 401
    code = "unauthorized"


class RateLimitError(AppError):
    status_code = 429
    code = "rate_limited"


class LLMUnavailableError(AppError):
    status_code = 503
    code = "llm_unavailable"


class IndexNotReadyError(AppError):
    status_code = 503
    code = "index_not_ready"


class IngestionError(AppError):
    code = "ingestion_error"


class RetrievalBlockedError(AppError):
    """Raised if retrieval is attempted while the readiness gate is not READY."""

    status_code = 409
    code = "retrieval_blocked"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        error_id = uuid.uuid4().hex[:12]
        log_event(logger, exc.message, logging.WARNING, error_id=error_id, code=exc.code)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "error_id": error_id, **exc.details}},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        error_id = uuid.uuid4().hex[:12]
        logger.exception("unhandled error", extra={"extra_fields": {"error_id": error_id}})
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Unexpected server error.", "error_id": error_id}},
        )
