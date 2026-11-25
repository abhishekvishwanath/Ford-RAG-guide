"""End-to-end: synthetic manual ingestion -> Chroma/BM25 -> FastAPI -> LangGraph -> grounded answer.

The LLM is a deterministic echo of evidence lines (external API calls are mocked in tests);
everything else (parser, chunker, embeddings, Chroma, BM25, cross-encoder, graph, SQLite) is real.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.ingestion.ingestion_service import IngestionService
from app.main import create_app
from app.services.container import Container
from tests.conftest import build_synthetic_manual
from tests.unit.test_graph_gate import ExplodingRetrieval

ADMIN = "test-admin-key"


class EchoLLM:
    """Answers by quoting evidence lines with their [S#] marker (or torque lines for torque questions)."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, **kw):  # type: ignore[no-untyped-def]
        if kw.get("json_mode"):
            return "{}"
        self.calls += 1
        user = messages[-1]["content"]
        q = re.search(r"Question: (.*)", user).group(1).lower()
        blocks = re.findall(
            r"\[(S\d+)\] Section:[^\n]*\n(.*?)(?=\n\n\[S\d+\] Section:|\n\nAnswer the question)", user, re.S
        )
        out = []
        for sid, body in blocks:
            for ln in body.splitlines():
                if ln.startswith("Diagram text"):
                    continue
                if "torque" in q and "Nm" in ln or "torque" not in q and re.match(r"\s*(\d+\.|WARNING|NOTICE)", ln):
                    out.append(f"{ln.strip()} [{sid}]")
        return "\n".join(out) or "The manual evidence does not contain this."


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    pdf = build_synthetic_manual(root / "manual.pdf")
    s = Settings(
        manual_pdf_path=pdf,
        processed_dir=root / "processed",
        images_dir=root / "images",
        indexes_dir=root / "indexes",
        chroma_persist_directory=root / "chroma",
        chroma_collection_name="e2e_manual",
        database_url=f"sqlite:///{root / 'sessions.db'}",
        ingestion_batch_size=2,
        llm_context_extraction_enabled=False,
        reranker_enabled=True,
        admin_api_key=ADMIN,
        rate_limit_per_minute=1000,
    )
    report = IngestionService(s).run()
    llm = EchoLLM()
    container = Container(s, llm=llm)
    client = TestClient(create_app(s, container))
    with client:
        yield {"settings": s, "report": report, "client": client, "container": container, "llm": llm}


def chat(client, message, sid=None):
    r = client.post("/api/chat", json={"session_id": sid, "message": message})
    assert r.status_code == 200, r.text
    return r.json()


def test_ingestion_through_chroma(env):
    rep = env["report"]
    assert rep["parse"]["pages_parsed"] == 3 and rep["parse"]["pages_failed"] == []
    assert rep["structure"]["documents"] == 2
    assert rep["index"]["vector_count"] >= rep["chunk"]["chunks"] >= 2
    c = env["container"]
    assert c.index_ready and c.profile.model_years == [2041]
    # re-running is idempotent: nothing re-embedded
    again = IngestionService(env["settings"]).run(["chunk", "index"])
    assert again["index"]["chunks_embedded"] == 0


def test_full_conversation_to_grounded_answer_with_diagram(env):
    client = env["client"]
    r = chat(client, "I need to remove the widget bracket")
    assert r["interaction_mode"] == "asking_context" and "model year" in r["assistant_message"].lower()
    sid = r["session_id"]
    r = chat(client, "2041", sid)
    assert r["interaction_mode"] == "asking_context" and r["readiness"]["status"] == "NEEDS_INFORMATION"
    assert "engine" in r["readiness"]["missing_fields"]  # procedure has 9.9L-specific steps
    r = chat(client, "9.9L", sid)
    assert r["interaction_mode"] == "answer", r["assistant_message"]
    assert r["readiness"]["status"] == "READY"
    msg = r["assistant_message"]
    assert "Disconnect the widget connector" in msg
    assert "WARNING: Support the widget before removal." in msg  # safety statement kept
    assert r["citations"] and r["citations"][0]["procedure_id"] == "SEB99001"
    assert r["citations"][0]["pdf_pages"].startswith("1")
    assert r["images"], "original diagram should be returned"
    img = client.get(r["images"][0]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert "file://" not in str(r) and str(env["settings"].images_dir) not in str(r)


def test_followup_retrieves_fresh_evidence(env):
    client = env["client"]
    sid = chat(client, "2041 testcar 9.9L, remove the widget bracket")["session_id"]
    r = chat(client, "what is the torque for the widget nut?", sid)
    assert r["interaction_mode"] == "answer"
    assert "77 Nm (57 lb-ft)" in r["assistant_message"]
    assert "dense_retrieve" in r["conversation_state"]["trace"]


def test_unsupported_question_is_honest(env):
    client = env["client"]
    r = chat(client, "2041 testcar, what is the firing order of the quantum flux capacitor?")
    sid = r["session_id"]
    if r["interaction_mode"] == "asking_context":
        r = chat(client, "the quantum flux capacitor", sid)
    if r["interaction_mode"] == "asking_context":
        r = chat(client, "specification", sid)
    assert r["interaction_mode"] == "answer"
    assert "could not find evidence" in r["assistant_message"] and r["citations"] == []


def test_incomplete_and_conflicting_context_never_retrieve(env):
    c = env["container"]
    saved = c.retrieval
    c.retrieval = ExplodingRetrieval()
    c.workflow = c._build_workflow()
    try:
        r = chat(env["client"], "I need to remove the widget bracket")
        assert r["interaction_mode"] == "asking_context"
        r = chat(env["client"], "2040 or 2041 testcar widget bracket removal")
        assert r["interaction_mode"] in ("awaiting_confirmation", "asking_context")
        assert r["readiness"]["status"] == "NEEDS_CONFIRMATION"
    finally:
        c.retrieval = saved
        c.workflow = c._build_workflow()


def test_session_persists_across_backend_restart(env):
    s = env["settings"]
    sid = chat(env["client"], "2041 testcar, remove the widget bracket")["session_id"]
    fresh = Container(s, llm=EchoLLM(), load_indexes=False)  # new process-equivalent: new engine/connection
    with TestClient(create_app(s, fresh)) as client2:
        r = client2.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["vehicle_context"]["model_year"]["value"] == 2041
    assert body["repair_context"]["component"]["value"] == "widget bracket"
    assert [m["role"] for m in body["messages"]][:2] == ["user", "assistant"]


def test_vehicle_panel_update_reevaluates_readiness(env):
    client = env["client"]
    sid = chat(client, "remove the widget bracket")["session_id"]
    r = client.post(f"/api/sessions/{sid}/context", json={"vehicle": {"model_year": 2041, "engine": "9.9L"}})
    assert r.status_code == 200
    assert r.json()["readiness"]["status"] == "READY"
    assert r.json()["vehicle_context"]["engine"]["status"] == "confirmed"


def test_admin_requires_key_and_assets_are_safe(env):
    client = env["client"]
    assert client.get("/api/admin/ingestion/status").status_code == 401
    assert client.get("/api/admin/ingestion/status", headers={"X-Admin-Key": "wrong"}).status_code == 401
    ok = client.get("/api/admin/ingestion/status", headers={"X-Admin-Key": ADMIN})
    assert ok.status_code == 200 and ok.json()["state"] == "done"
    assert client.get("/api/manual-assets/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/api/manual-assets/img_ffffffffffffffffffff").status_code == 404
    assert client.get("/api/sessions/does-not-exist").status_code == 404


def test_validation_errors(env):
    client = env["client"]
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 3000}).status_code == 422
    assert client.post("/api/chat", json={"message": "hi", "vehicle": {"engine": "big"}}).status_code == 422
