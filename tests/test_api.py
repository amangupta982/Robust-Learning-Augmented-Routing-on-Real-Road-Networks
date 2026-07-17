"""Smoke tests for the Phase 7 routing API (roar/api/app.py). Uses the real
graph, real trained model, and real instrumented-edge endpoints (same
approach as the Phase 4/5 real-data tests) -- skipped if `make data` /
`make train` haven't been run yet.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from roar.api.app import app
from roar.eval.metrics import resolve_model_dir
from roar.graph.load_graph import EDGES_PATH, GRAPH_PATH

from tests.robust_astar_fixtures import instrumented_edge_endpoints

pytestmark = pytest.mark.skipif(
    not (GRAPH_PATH.exists() and EDGES_PATH.exists() and resolve_model_dir().exists()),
    reason="real graph/trained model not built yet; run `make data && make train` first",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_edge():
    return instrumented_edge_endpoints()[0]


def test_predictor_health_reports_ok_after_startup(client):
    response = client.get("/predictor/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["predictor_type"] == "lightgbm"
    assert body["model_version"]
    assert body["loaded_at"]


def test_route_returns_a_valid_path_for_a_real_instrumented_edge(client, sample_edge):
    origin, dest = sample_edge
    response = client.post(
        "/route",
        json={"origin": origin, "destination": dest, "depart_time": "2012-06-04T08:00:00"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["path"] is not None
    assert body["path"][0] == origin
    assert body["path"][-1] == dest
    assert body["cost"] > 0
    assert body["eta"] is not None
    assert body["node_expansions"] >= 1
    assert body["latency_ms"] >= 0
    bound = body["robustness_bound"]
    assert bound["alpha"] == pytest.approx(0.3)
    assert bound["ratio"] <= 1.3 + 1e-6


def test_route_respects_a_custom_alpha_and_lambda(client, sample_edge):
    origin, dest = sample_edge
    response = client.post(
        "/route",
        json={
            "origin": origin,
            "destination": dest,
            "depart_time": "2012-06-04T08:00:00",
            "lambda": 0.0,
            "alpha": 0.1,
        },
    )
    assert response.status_code == 200
    assert response.json()["robustness_bound"]["alpha"] == pytest.approx(0.1)


def test_route_rejects_lambda_out_of_range(client, sample_edge):
    origin, dest = sample_edge
    response = client.post(
        "/route",
        json={
            "origin": origin,
            "destination": dest,
            "depart_time": "2012-06-04T08:00:00",
            "lambda": 1.5,
        },
    )
    assert response.status_code == 422


def test_route_rejects_unknown_origin_node(client, sample_edge):
    _, dest = sample_edge
    response = client.post(
        "/route",
        json={
            "origin": "not-a-real-node-id",
            "destination": dest,
            "depart_time": "2012-06-04T08:00:00",
        },
    )
    assert response.status_code == 422
    assert "unknown origin" in response.json()["detail"]


def test_route_rejects_unknown_destination_node(client, sample_edge):
    origin, _ = sample_edge
    response = client.post(
        "/route",
        json={
            "origin": origin,
            "destination": "not-a-real-node-id",
            "depart_time": "2012-06-04T08:00:00",
        },
    )
    assert response.status_code == 422
    assert "unknown destination" in response.json()["detail"]


def test_route_is_missing_required_fields_returns_422(client):
    response = client.post("/route", json={"origin": "1"})
    assert response.status_code == 422


def test_origin_equals_destination_is_a_zero_cost_route(client, sample_edge):
    origin, _ = sample_edge
    response = client.post(
        "/route",
        json={"origin": origin, "destination": origin, "depart_time": "2012-06-04T08:00:00"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["path"] == [origin]
    assert body["cost"] == 0.0
