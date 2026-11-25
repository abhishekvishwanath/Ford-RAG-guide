"""FastAPI application factory."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin, assets, chat, health, sessions
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.services.container import Container


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        if getattr(app.state, "container", None) is None:
            app.state.container = Container(settings)
            threading.Thread(target=app.state.container.warmup, daemon=True).start()
        app.state.container.sessions.purge_expired()
        yield

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.container = container
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Admin-Key"],
    )
    register_exception_handlers(app)
    for r in (health.router, chat.router, sessions.router, admin.router, assets.router):
        app.include_router(r)
    return app


app = create_app()
