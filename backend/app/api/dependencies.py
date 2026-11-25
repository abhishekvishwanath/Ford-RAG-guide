"""FastAPI dependencies: container access, admin authentication, rate limiting."""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, Request

from app.core.exceptions import AuthError, RateLimitError
from app.services.container import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


def require_admin(request: Request, x_admin_key: str | None = Header(default=None)) -> None:
    expected = get_container(request).settings.admin_api_key.get_secret_value()
    if not expected:
        raise AuthError("Admin API is disabled (ADMIN_API_KEY not configured).")
    if not x_admin_key or not hmac.compare_digest(x_admin_key, expected):
        raise AuthError("Invalid admin credentials.")


class RateLimiter:
    """Sliding-window limiter per client (prototype: in-process memory)."""

    def __init__(self) -> None:
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window: float = 60.0) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                raise RateLimitError("Too many requests; please wait a moment.")
            q.append(now)


_limiter = RateLimiter()


def rate_limit(request: Request) -> None:
    limit = get_container(request).settings.rate_limit_per_minute
    client = request.client.host if request.client else "unknown"
    _limiter.check(client, limit)
