"""Run the evaluation suites against the ingested manual.

  python scripts/evaluate_rag.py resolve        # verify draft cases against the manual -> retrieval_cases.json
  python scripts/evaluate_rag.py retrieval      # structure-based system + ablations
  python scripts/evaluate_rag.py chunking       # fixed-size / recursive baselines vs structure-based
  python scripts/evaluate_rag.py context        # conversational readiness gate
  python scripts/evaluate_rag.py faithfulness --n 8 --sleep 50   # live LLM (rate-limited) validator proxy
Results are written to docs/evaluation/*.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

from app.core.config import get_settings  # noqa: E402
from app.evaluation import runner as R  # noqa: E402
from app.evaluation.datasets import resolve_cases  # noqa: E402

DATA = ROOT / "backend/tests/evaluation/data"
OUT = ROOT / "docs/evaluation"


def container():  # type: ignore[no-untyped-def]
    from app.services.container import Container

    return Container(get_settings())


def load_cases() -> list[dict]:
    return json.loads((DATA / "retrieval_cases.json").read_text())["cases"]


def cmd_resolve() -> None:
    c = container()
    cases, problems = resolve_cases(DATA / "retrieval_cases_draft.json", c.manual)
    (DATA / "retrieval_cases.json").write_text(json.dumps({"cases": cases}, indent=1))
    print(f"resolved {len(cases)} cases; problems: {len(problems)}")
    for p in problems:
        print("  -", p)


def cmd_retrieval() -> None:
    c = container()
    cases = load_cases()
    results = {}
    for variant in ("full", "no_rerank", "dense_only", "bm25_only", "no_filter"):
        results[variant] = R.run_structure(c.retrieval, c.manual, cases, variant)
        print(variant, results[variant]["summary"])
    R.save(OUT / "retrieval_results.json", results)


def cmd_chunking(rebuild: bool = False) -> None:
    c = container()
    s = get_settings()
    if rebuild:
        import shutil

        # eval_chroma holds only the throwaway baseline collections
        shutil.rmtree(s.indexes_dir / "eval_chroma", ignore_errors=True)
    cases = load_cases()
    out = {"C_structure": R.run_structure(c.retrieval, c.manual, cases, "full")}
    out["C_structure"]["chunk_stats"] = json.loads((s.processed_dir / "chunk_report.json").read_text())["token_stats"]
    for name, chunks in R.build_baselines(s).items():
        out[name] = R.run_baseline(name.split("_")[0].lower(), chunks, s, cases, c.manual, c.retrieval.reranker)
        print(name, out[name]["summary"], out[name]["chunk_stats"])
    print("C_structure", out["C_structure"]["summary"])
    R.save(OUT / "chunking_comparison.json", out)


def cmd_context() -> None:
    from app.graph.state import GraphDeps
    from app.graph.workflow import build_workflow
    from app.memory.session_store import SqliteSessionRepository
    from app.services.manual_profile import load_profile

    sys.path.insert(0, str(ROOT / "backend"))
    from tests.unit.test_graph_gate import FakeGenerator, RecordingRetrieval

    s = get_settings().model_copy(update={"llm_context_extraction_enabled": False})
    profile = load_profile(str(s.manual_profile_path))
    tmp = OUT / "_ctx_sessions.db"

    def factory():  # type: ignore[no-untyped-def]
        rec = RecordingRetrieval()
        repo = SqliteSessionRepository(f"sqlite:///{tmp}")
        deps = GraphDeps(settings=s, repo=repo, profile=profile, retrieval=rec, generator=FakeGenerator())
        return build_workflow(deps), rec

    cases = json.loads((DATA / "conversation_cases.json").read_text())["conversations"]
    res = R.run_context(cases, factory)
    tmp.unlink(missing_ok=True)
    print(json.dumps(res["summary"], indent=1))
    for conv in res["conversations"]:
        bad = [t for t in conv["turns"] if not t["ok"]]
        if bad:
            print(" mismatch", conv["id"], [(t["user"], t["expected"], t["got"], t["asked"]) for t in bad])
    R.save(OUT / "context_results.json", res)


def cmd_faithfulness(n: int, sleep_s: float) -> None:
    from app.generation.answer_generator import GroundedAnswerGenerator

    c = container()
    gen = GroundedAnswerGenerator(c.settings, c.llm, c.sessions, c.document_title)
    cases = load_cases()
    step = max(1, len(cases) // n)
    subset = cases[::step][:n]
    res = R.run_faithfulness(c.retrieval, gen, subset, sleep_s)
    print(json.dumps(res["summary"], indent=1))
    R.save(OUT / "faithfulness_results.json", res)


def main() -> None:
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["resolve", "retrieval", "chunking", "context", "faithfulness"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=50.0)
    ap.add_argument("--rebuild", action="store_true", help="re-embed baseline corpora")
    a = ap.parse_args()
    {
        "resolve": cmd_resolve,
        "retrieval": cmd_retrieval,
        "chunking": lambda: cmd_chunking(a.rebuild),
        "context": cmd_context,
    }.get(a.suite, lambda: cmd_faithfulness(a.n, a.sleep))()


if __name__ == "__main__":
    main()
