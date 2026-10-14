"""API smoke tests against the committed IEEE-CIS model artifacts."""
from fastapi.testclient import TestClient

from api.main import app


def test_health_and_ieee_scoring():
    with TestClient(app) as client:  # the context manager runs the lifespan (model loading)
        health = client.get("/health")
        assert health.status_code == 200
        ieee = health.json()["datasets"]["ieee_cis"]
        assert "xgboost" in ieee["models_loaded"]

        features = client.get("/features/ieee").json()["features"]
        resp = client.post("/predict/ieee", json={"features": [0.0] * len(features)})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert 0.0 <= body["fraud_score"] <= 1.0
        assert body["dataset"] == "ieee_cis"


def test_rejects_wrong_feature_count_and_non_finite():
    with TestClient(app) as client:
        assert client.post("/predict/ieee", json={"features": [0.0, 1.0]}).status_code == 422
        assert client.post("/predict/cc", json={"features": [0.0] * 29}).status_code == 422
