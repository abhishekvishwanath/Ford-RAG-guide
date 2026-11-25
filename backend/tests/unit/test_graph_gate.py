"""The readiness gate must make retrieval unreachable unless the decision is READY."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.graph.state import GraphDeps
from app.graph.workflow import RETRIEVAL, build_workflow, run_turn
from app.memory.session_store import SqliteSessionRepository
from app.schemas.retrieval import Candidate, EvidenceItem, EvidencePackage, GeneratedAnswer, RetrievalRequest
from tests.fakes import make_profile


class ExplodingRetrieval:
    """Any call fails the test."""

    def __getattr__(self, name):
        def boom(*a, **k):
            raise AssertionError(f"retrieval.{name} called before READY")

        return boom


class RecordingRetrieval:
    def __init__(self, sufficient: bool = True):
        self.calls: list[str] = []
        self.sufficient = sufficient

    def build_request(self, state, message, extraction):
        self.calls.append("build_request")
        return RetrievalRequest(query_text=message, keyword_query=message, original_message=message,
                                engine=state.value("engine"))  # fmt: skip

    def applicability_filter(self, request):
        self.calls.append("filter")
        return {}

    def dense(self, request, flt):
        self.calls.append("dense")
        return [Candidate(chunk_id="c1", dense_rank=1)]

    def bm25(self, request, flt):
        self.calls.append("bm25")
        return [Candidate(chunk_id="c1", bm25_rank=1)]

    def fuse(self, d, b):
        self.calls.append("fuse")
        return d

    def rerank(self, request, c):
        self.calls.append("rerank")
        return c

    def expand(self, request, c):
        self.calls.append("expand")
        item = EvidenceItem(source_id="S1", chunk_id="c1", parent_id="proc_CP50", procedure_title="Coolant Pump — 5.0L",
                            section="303-03A Engine Cooling", doc_type="REMOVAL AND INSTALLATION", subsection="Removal",
                            page_start=1, page_end=1, printed_page_start=None, printed_page_end=None,
                            applicability_note="5.0L", content="1. Drain the cooling system.")  # fmt: skip
        return EvidencePackage(request=request, items=[item])

    def visuals_and_tables(self, pkg):
        self.calls.append("visuals")
        return pkg

    def assess(self, pkg):
        self.calls.append("assess")
        pkg.sufficient = self.sufficient
        pkg.insufficiency_reason = None if self.sufficient else "no matching procedure"
        return pkg


class FakeGenerator:
    def __init__(self):
        self.generated = 0

    def build_prompt(self, state, message, pkg):
        return [{"role": "user", "content": message}]

    def generate(self, messages):
        self.generated += 1
        return "1. Drain the cooling system. [S1]"

    def validate(self, text, pkg):
        return GeneratedAnswer(text=text, cited_source_ids=["S1"])

    def citations(self, answer, pkg):
        from app.schemas.retrieval import Citation

        return [Citation(source_id="S1", chunk_id="c1", document_title="T", section=None, procedure_title=None,
                         procedure_id="CP50", subsection=None, pdf_pages="1", printed_pages=None)]  # fmt: skip

    def fallback_answer(self, pkg):
        return GeneratedAnswer(text="fallback", cited_source_ids=["S1"])


@pytest.fixture
def make(tmp_path):
    def _make(retrieval, generator=None):
        settings = Settings(database_url=f"sqlite:///{tmp_path / 's.db'}", llm_context_extraction_enabled=False)
        repo = SqliteSessionRepository(settings.database_url)
        deps = GraphDeps(settings=settings, repo=repo, profile=make_profile(), retrieval=retrieval,
                         generator=generator or FakeGenerator())  # fmt: skip
        return build_workflow(deps), repo

    return _make


def test_incomplete_context_asks_and_never_retrieves(make):
    wf, _ = make(ExplodingRetrieval())
    r = run_turn(wf, None, "I need to replace the water pump.")
    assert r["interaction_mode"] == "asking_context"
    assert r["readiness"]["status"] == "NEEDS_INFORMATION"
    assert "model year" in r["assistant_message"].lower()
    assert not set(RETRIEVAL) & set(r["conversation_state"]["trace"])
    r = run_turn(wf, r["session_id"], "2014")
    assert r["interaction_mode"] == "asking_context" and "engine" in r["assistant_message"].lower()


def test_conflict_confirmation_safety_and_scope_never_retrieve(make):
    wf, _ = make(ExplodingRetrieval())
    r = run_turn(wf, None, "2013 or 2014 mustang, 5.0L, replace coolant pump")
    assert r["readiness"]["status"] == "NEEDS_CONFIRMATION"
    r = run_turn(wf, None, "2014 mustang GT replace coolant pump")
    assert r["interaction_mode"] == "awaiting_confirmation"
    r = run_turn(wf, None, "2014 mustang 5.0L, fuel leaking from the rail while replacing the coolant pump")
    assert r["interaction_mode"] == "safety_escalation"
    r = run_turn(wf, None, "2019 mustang coolant pump")
    assert r["interaction_mode"] == "out_of_scope"


def test_ready_path_runs_full_retrieval_and_generation(make):
    rec, gen = RecordingRetrieval(), FakeGenerator()
    wf, repo = make(rec, gen)
    r = run_turn(wf, None, "I need to replace the water pump.")
    sid = r["session_id"]
    run_turn(wf, sid, "2014")
    assert rec.calls == []  # still nothing retrieved during intake
    r = run_turn(wf, sid, "5.0L")
    assert r["interaction_mode"] == "answer" and r["readiness"]["status"] == "READY"
    assert rec.calls == ["build_request", "filter", "dense", "bm25", "fuse", "rerank", "expand", "visuals", "assess"]
    assert gen.generated == 1 and r["citations"][0]["source_id"] == "S1"
    assert "Vehicle and task" in r["assistant_message"]
    # persisted
    s = repo.get(sid)
    assert s.active_procedure.parent_id == "proc_CP50" and s.pending_question is None


def test_followup_triggers_fresh_retrieval(make):
    rec = RecordingRetrieval()
    wf, _ = make(rec)
    sid = run_turn(wf, None, "2014 mustang 5.0L replace the coolant pump")["session_id"]
    n = len(rec.calls)
    r = run_turn(wf, sid, "what torque for those bolts?")
    assert r["interaction_mode"] == "answer"
    assert rec.calls[n:].count("dense") == 1  # fresh evidence, not reused generated text


def test_insufficient_evidence_is_honest(make):
    wf, _ = make(RecordingRetrieval(sufficient=False))
    r = run_turn(wf, None, "2014 mustang remove the door trim panel")
    assert r["interaction_mode"] == "answer" and r["citations"] == []
    assert "could not find evidence" in r["assistant_message"]


def test_retrieval_node_guard_blocks_direct_invocation():
    from app.conversation.state import ReadinessDecision, ReadinessStatus
    from app.core.exceptions import RetrievalBlockedError
    from app.graph import nodes

    state = {"decision": ReadinessDecision(status=ReadinessStatus.NEEDS_INFORMATION)}
    for name in RETRIEVAL:
        with pytest.raises(RetrievalBlockedError):
            getattr(nodes, name)(state, deps=None)
