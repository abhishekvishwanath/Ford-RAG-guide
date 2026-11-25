"""LLM client abstraction. Groq is the configured provider; tests inject fakes."""

from __future__ import annotations

import json
import logging
import time
from typing import Protocol

from app.core.config import Settings
from app.core.exceptions import LLMUnavailableError
from app.core.logging import log_event

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str = "low",
    ) -> str: ...


class GroqLLMClient:
    def __init__(self, settings: Settings) -> None:
        from groq import Groq

        key = settings.groq_api_key.get_secret_value()
        if not key:
            raise LLMUnavailableError("GROQ_API_KEY is not configured.")
        self.s = settings
        self.client = Groq(api_key=key, timeout=settings.llm_timeout_seconds, max_retries=settings.llm_max_retries)

    def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str = "low",
    ) -> str:
        model = model or self.s.llm_model
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens or self.s.llm_max_output_tokens,
            "temperature": self.s.llm_temperature if temperature is None else temperature,
        }
        if model.startswith("openai/gpt-oss"):
            kwargs.update(reasoning_effort=reasoning_effort, include_reasoning=False)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            log_event(logger, "llm call failed", logging.WARNING, model=model, error=type(exc).__name__)
            raise LLMUnavailableError("The language model is currently unavailable.") from exc
        content = resp.choices[0].message.content or ""
        log_event(
            logger,
            "llm call",
            model=model,
            ms=round((time.perf_counter() - t0) * 1000),
            tokens=getattr(resp.usage, "total_tokens", None),
        )
        return content


def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating code fences or leading prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])
