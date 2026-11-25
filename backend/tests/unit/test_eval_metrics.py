import math

import pytest

from app.evaluation import metrics as M
from app.evaluation.baselines import fixed_size_chunks, recursive_chunks
from tests.unit.test_indexing import WordCounter


def test_metrics():
    rel = [False, True, False, True, False]
    assert M.recall_at_k(rel, 5) == 1.0 and M.recall_at_k(rel, 1) == 0.0
    assert M.precision_at_k(rel, 5) == pytest.approx(0.4)
    assert M.reciprocal_rank(rel) == 0.5 and M.reciprocal_rank([False] * 3) == 0.0
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert M.ndcg_at_k(rel, 5) == pytest.approx(expected)


class Counter(WordCounter):
    def count(self, text: str) -> int:
        return len(text.split())


def test_fixed_and_recursive_baselines_keep_page_ranges():
    pages = [(0, "alpha " * 300), (1, "beta gamma.\n\n" + "delta " * 300)]
    fixed = fixed_size_chunks(pages, Counter(), size=130)
    assert all(c.page_start <= c.page_end for c in fixed)
    assert fixed[0].page_start == 1 and fixed[-1].page_end == 2
    rec = recursive_chunks(pages, Counter(), size=100, overlap=10)
    assert all(c.token_count <= 120 for c in rec)
    assert {c.page_start for c in rec} == {1, 2}
    assert all(c.applicability.engines == [] for c in fixed + rec)  # baselines carry no metadata
