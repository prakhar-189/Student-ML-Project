"""Integration tests for the Flask app: routes, validation, and JSON prediction.

These load the serialized model/preprocessor from artifacts/ and exercise the
real prediction pipeline -- but require no training.
"""
import pytest

from app import app as flask_app

VALID_PAYLOAD = {
    "gender": "male",
    "race_ethnicity": "group C",
    "parental_level_of_education": "some college",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 74,
    "writing_score": 71,
}


@pytest.fixture
def client():
    flask_app.config.update(TESTING=True)
    return flask_app.test_client()


def test_home_page_renders(client):
    assert client.get("/").status_code == 200


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_predict_api_returns_score_in_range(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    score = resp.get_json()["math_score"]
    assert isinstance(score, (int, float))
    assert 0 <= score <= 100


def test_predict_api_rejects_bad_category(client):
    bad = {**VALID_PAYLOAD, "gender": "unknown"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_api_rejects_out_of_range_score(client):
    bad = {**VALID_PAYLOAD, "reading_score": 250}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 400


def test_predict_api_rejects_missing_field(client):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "writing_score"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 400
