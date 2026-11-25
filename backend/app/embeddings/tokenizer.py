"""Token counting with the embedding model's own tokenizer (word-based fallback)."""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def hf_model_name(name: str) -> str:
    return name if "/" in name else f"sentence-transformers/{name}"


class TokenCounter:
    def __init__(self, model_name: str) -> None:
        self._tok = None
        try:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(hf_model_name(model_name))
            self._tok.model_max_length = 10**9  # counting only; never truncate
        except Exception as exc:  # offline / missing model -> approximate
            logger.warning("tokenizer unavailable (%s); using word-based estimate", exc)

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tok is None:
            return int(len(text.split()) * 1.35) + 1
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def split_by_tokens(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Token windows over text (used for dense multi-window indexing and last-resort splitting)."""
        if self._tok is None:
            words = text.split()
            step = max(1, int(max_tokens / 1.35) - int(overlap / 1.35))
            size = int(max_tokens / 1.35)
            return [" ".join(words[i : i + size]) for i in range(0, max(1, len(words)), step) if words[i : i + size]]
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        if len(offs) <= max_tokens:
            return [text]
        out, start = [], 0
        while start < len(offs):
            end = min(start + max_tokens, len(offs))
            out.append(text[offs[start][0] : offs[end - 1][1]])
            if end == len(offs):
                break
            start = end - overlap
        return out


@lru_cache
def get_token_counter(model_name: str) -> TokenCounter:
    return TokenCounter(model_name)
