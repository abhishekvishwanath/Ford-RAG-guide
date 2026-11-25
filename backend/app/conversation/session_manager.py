"""Session lifecycle on top of the repository (create / resume / reset / expire)."""

from __future__ import annotations

from app.conversation.state import SessionState
from app.core.exceptions import NotFoundError
from app.memory.session_store import SessionRepository


class SessionManager:
    def __init__(self, repo: SessionRepository) -> None:
        self.repo = repo

    def get_or_create(self, session_id: str | None) -> SessionState:
        if session_id:
            state = self.repo.get(session_id)
            if state is None:
                raise NotFoundError("Session not found or expired. Start a new conversation.")
            return state
        return self.repo.create()

    def get(self, session_id: str) -> SessionState:
        state = self.repo.get(session_id)
        if state is None:
            raise NotFoundError("Session not found or expired.")
        return state

    def reset(self, session_id: str) -> SessionState:
        self.repo.delete(session_id)
        return self.repo.create()

    def delete(self, session_id: str) -> None:
        if not self.repo.delete(session_id):
            raise NotFoundError("Session not found.")
