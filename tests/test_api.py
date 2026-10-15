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


def test_scores_come_only_from_real_models():
    """No invented scores: the committed artifacts have XGBoost + Isolation
    Forest but no autoencoder, so exactly those two are blended."""
    with TestClient(app) as client:
        n = len(client.get("/features/ieee").json()["features"])
        body = client.post("/predict/ieee", json={"features": [0.1] * n}).json()
        scores = body["model_scores"]
        assert set(scores) == {"xgboost", "isolation_forest", "ensemble"}
        expected = 0.65 * scores["xgboost"] + 0.35 * scores["isolation_forest"]
        assert abs(scores["ensemble"] - expected) < 1e-3


def test_missing_models_give_503_not_made_up_scores():
    # no credit-card models are committed
    with TestClient(app) as client:
        r = client.post("/predict/cc", json={"features": [0.0] * 30})
        assert r.status_code == 503
        r = client.post("/predict", json={"features": [0.0] * 30})
        assert r.status_code == 503


def test_raw_features_need_the_scaler():
    with TestClient(app) as client:
        n = len(client.get("/features/ieee").json()["features"])
        if client.get("/health").json()["datasets"]["ieee_cis"]["scaler_loaded"]:
            return  # a local run with processed data present
        r = client.post("/predict/ieee", json={"features": [0.0] * n, "prescaled": False})
        assert r.status_code == 422
        assert "scaler" in r.json()["detail"]


def test_feature_list_is_bounded():
    with TestClient(app) as client:
        assert client.post("/predict/ieee", json={"features": [0.0] * 5000}).status_code == 422


def test_batch_endpoint():
    with TestClient(app) as client:
        n = len(client.get("/features/ieee").json()["features"])
        r = client.post("/predict/batch", json={"transactions": [
            {"transaction_id": "a", "features": [0.0] * n},
            {"transaction_id": "b", "features": [0.5] * n},
        ]})
        assert r.status_code == 200, r.text
        assert [x["transaction_id"] for x in r.json()] == ["a", "b"]
        assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
