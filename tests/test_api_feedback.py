"""POST /api/feedback —— 满意度反馈（本章预留，不落库）。"""

from fastapi.testclient import TestClient

from app.main import app


def test_post_feedback():
    with TestClient(app) as client:
        resp = client.post(
            "/api/feedback",
            json={"message_id": 123, "feedback": "thumbs_up"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_post_feedback_accepts_thumbs_down():
    with TestClient(app) as client:
        resp = client.post(
            "/api/feedback",
            json={"message_id": 456, "feedback": "thumbs_down"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_post_feedback_rejects_unknown_value():
    with TestClient(app) as client:
        resp = client.post(
            "/api/feedback",
            json={"message_id": 123, "feedback": "maybe"},
        )
    assert resp.status_code == 422
