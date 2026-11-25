"""Configurable cross-encoder reranking (with a no-op implementation)."""

from __future__ import annotations

import logging
import math
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class NoopReranker:
    name = "none"

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.0] * len(passages)


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):  # type: ignore[no-untyped-def]
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.name, max_length=512)
                logger.info("reranker loaded: %s", self.name)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw = self._load().predict([(query, p) for p in passages], show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]  # logits -> (0, 1)
