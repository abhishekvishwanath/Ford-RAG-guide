"""Secure serving of original manual images by asset id (never by path)."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.api.dependencies import get_container
from app.core.exceptions import NotFoundError
from app.services.container import Container

router = APIRouter(prefix="/api", tags=["assets"])
ASSET_RE = re.compile(r"^img_[0-9a-f]{20}$")


@router.get("/manual-assets/{asset_id}")
def manual_asset(asset_id: str, c: Container = Depends(get_container)) -> FileResponse:
    if not ASSET_RE.fullmatch(asset_id):
        raise NotFoundError("Asset not found.")
    rec = c.manual.assets.get(asset_id)
    if rec is None:
        raise NotFoundError("Asset not found.")
    root = c.settings.images_dir.resolve()
    path = (root / rec.asset_relpath).resolve()
    if root not in path.parents or not path.is_file():
        raise NotFoundError("Asset not found.")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=86400",
                                                               "X-Content-Type-Options": "nosniff"})  # fmt: skip
