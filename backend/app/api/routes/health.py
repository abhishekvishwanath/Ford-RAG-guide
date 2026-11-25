from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "app": s.app_name,
        "manual_present": s.manual_pdf_path.exists(),
        "manual_profile_present": s.manual_profile_path.exists(),
        "llm_configured": bool(s.groq_api_key.get_secret_value()),
        "llm_model": s.llm_model,
    }
