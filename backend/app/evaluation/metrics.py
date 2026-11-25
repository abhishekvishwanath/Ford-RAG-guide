"""Standard IR metrics over ranked lists with binary relevance."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(relevant_flags: Sequence[bool], k: int) -> float:
    """Hit-rate style recall: 1 if any relevant item is in the top-k (one target procedure per query)."""
    return 1.0 if any(relevant_flags[:k]) else 0.0


def precision_at_k(relevant_flags: Sequence[bool], k: int) -> float:
    top = list(relevant_flags[:k])
    return sum(top) / k if k else 0.0


def reciprocal_rank(relevant_flags: Sequence[bool]) -> float:
    for i, rel in enumerate(relevant_flags, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant_flags: Sequence[bool], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(relevant_flags[:k]) if rel)
    ideal_hits = min(k, sum(relevant_flags)) or (1 if any(relevant_flags) else 0)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
