from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.container import Container


def test_health_reports_degraded_without_index(tmp_path) -> None:
    s = Settings(database_url=f"sqlite:///{tmp_path / 's.db'}", processed_dir=tmp_path, indexes_dir=tmp_path)
    c = Container(s, llm=None, load_indexes=True)
    with TestClient(create_app(s, c)) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded" and body["index_ready"] is False
    assert "gsk_" not in r.text  # never leak secrets
