from fastapi.testclient import TestClient

from app.main import app


def test_healthz_returns_ok_shape() -> None:
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in {"ok", "degraded"}
        assert "model_loaded" in body
        assert "model_version" in body
