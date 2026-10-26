"""Integration tests for the FastAPI service.

They run against the real ASGI app through httpx's ASGITransport -- no network,
no uvicorn process -- with the session-trained bundle injected into the model
holder. That exercises routing, validation, middleware, headers and the
prediction path exactly as deployed.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from churn.api.main import app
from churn.api.model_registry import ChurnModel, holder

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client(trained_bundle) -> Iterator[TestClient]:
    holder.set(
        ChurnModel(
            trained_bundle["pipeline"], trained_bundle["threshold"], trained_bundle["metadata"]
        )
    )
    with TestClient(app) as c:
        yield c


def test_health_reports_ok_with_a_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-0.0.1"
    assert 0.0 < body["threshold"] < 1.0


def test_every_response_carries_the_model_version_header(client):
    response = client.get("/health")
    assert response.headers["X-Model-Version"] == "test-0.0.1"
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Response-Time-Ms"]) >= 0


def test_request_id_is_echoed_when_supplied(client):
    rid = "abc-123-trace"
    response = client.get("/health", headers={"X-Request-ID": rid})
    assert response.headers["X-Request-ID"] == rid


def test_predict_returns_a_calibrated_probability_and_a_decision(client, valid_customer):
    response = client.post("/predict", json={"customer": valid_customer, "customer_id": "c-1"})
    assert response.status_code == 200
    body = response.json()
    pred = body["prediction"]
    assert 0.0 <= pred["churn_probability"] <= 1.0
    assert pred["customer_id"] == "c-1"
    assert pred["churn_label"] == (pred["churn_probability"] >= pred["threshold"])
    assert pred["recommended_action"] in {"retention_offer", "no_action"}
    assert body["model_version"] == "test-0.0.1"
    assert body["request_id"]


def test_predict_is_deterministic(client, valid_customer):
    payload = {"customer": valid_customer}
    first = client.post("/predict", json=payload).json()["prediction"]["churn_probability"]
    second = client.post("/predict", json=payload).json()["prediction"]["churn_probability"]
    assert first == second


def test_month_to_month_short_tenure_scores_above_a_two_year_contract(client, valid_customer):
    """Sanity check on direction: the model should not be inverted."""
    risky = client.post("/predict", json={"customer": valid_customer}).json()
    safe_customer = {**valid_customer, "Contract": "Two year", "tenure": 70, "TotalCharges": 6699.0}
    safe = client.post("/predict", json={"customer": safe_customer}).json()
    assert risky["prediction"]["churn_probability"] > safe["prediction"]["churn_probability"]


def test_batch_predict_preserves_order_and_ids(client, valid_customer):
    other = {**valid_customer, "Contract": "Two year", "tenure": 60}
    response = client.post(
        "/predict_batch",
        json={"customers": [valid_customer, other], "customer_ids": ["a", "b"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["n"] == 2
    assert [p["customer_id"] for p in body["predictions"]] == ["a", "b"]


def test_batch_matches_single_predictions(client, valid_customer):
    single = client.post("/predict", json={"customer": valid_customer}).json()
    batch = client.post("/predict_batch", json={"customers": [valid_customer]}).json()
    assert batch["predictions"][0]["churn_probability"] == pytest.approx(
        single["prediction"]["churn_probability"]
    )


def test_mismatched_id_length_is_rejected(client, valid_customer):
    response = client.post(
        "/predict_batch", json={"customers": [valid_customer], "customer_ids": ["a", "b"]}
    )
    assert response.status_code == 422


def test_unknown_field_is_rejected(client, valid_customer):
    response = client.post(
        "/predict", json={"customer": {**valid_customer, "favourite_colour": "blue"}}
    )
    assert response.status_code == 422


def test_invalid_category_level_is_rejected(client, valid_customer):
    response = client.post("/predict", json={"customer": {**valid_customer, "Contract": "Decade"}})
    assert response.status_code == 422


def test_out_of_range_numeric_is_rejected(client, valid_customer):
    response = client.post("/predict", json={"customer": {**valid_customer, "tenure": -4}})
    assert response.status_code == 422


def test_missing_required_field_is_rejected(client, valid_customer):
    payload = {k: v for k, v in valid_customer.items() if k != "Contract"}
    assert client.post("/predict", json={"customer": payload}).status_code == 422


def test_null_total_charges_is_accepted_and_imputed(client, valid_customer):
    response = client.post("/predict", json={"customer": {**valid_customer, "TotalCharges": None}})
    assert response.status_code == 200
    assert 0.0 <= response.json()["prediction"]["churn_probability"] <= 1.0


def test_empty_batch_is_rejected(client):
    assert client.post("/predict_batch", json={"customers": []}).status_code == 422


def test_metrics_endpoint_exposes_the_instruments(client, valid_customer):
    client.post("/predict", json={"customer": valid_customer})
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    for name in (
        "churn_requests_total",
        "churn_request_latency_seconds",
        "churn_predictions_total",
        "churn_predicted_probability",
    ):
        assert name in body


def test_service_card_lists_the_endpoints(client):
    body = client.get("/").json()
    assert "/predict" in body["endpoints"]
    assert body["model_loaded"] is True


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/predict_batch" in schema["paths"]
