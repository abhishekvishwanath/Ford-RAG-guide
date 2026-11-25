from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.services.container import Container

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health(c: Container = Depends(get_container)) -> dict:
    s = c.settings
    return {
        "status": "ok" if c.index_ready else "degraded",
        "app": s.app_name,
        "manual_present": s.manual_pdf_path.exists(),
        "index_ready": c.index_ready,
        "manual": c.manual.stats(),
        "model_years": c.profile.model_years,
        "llm_configured": c.llm is not None,
        "llm_model": s.llm_model,
    }
