from types import SimpleNamespace

import pytest

import app.web


class FakeAgent:
    def answer(self, message, session=None):
        return SimpleNamespace(
            text=f"Answered: {message}",
            sources=["01-returns-policy-current.md — Standard return window"],
            tool_name=None,
            tool_arguments=None,
            handoff=False,
        )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        app.web,
        "create_agent",
        lambda: FakeAgent(),
    )

    application = app.web.create_app()
    application.config["TESTING"] = True

    return application.test_client()


def test_index_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Aster & Row Support Agent" in response.data


def test_chat_requires_message(client):
    response = client.post(
        "/api/chat",
        json={
            "session_id": "test-session",
        },
    )

    assert response.status_code == 400


def test_chat_requires_session_id(client):
    response = client.post(
        "/api/chat",
        json={
            "message": "Hello",
        },
    )

    assert response.status_code == 400


def test_chat_returns_structured_response(client):
    response = client.post(
        "/api/chat",
        json={
            "message": "What is the return window?",
            "session_id": "test-session",
        },
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["answer"] == (
        "Answered: What is the return window?"
    )

    assert data["sources"] == [
        "01-returns-policy-current.md — Standard return window"
    ]

    assert data["handoff"] is False


def test_same_session_can_be_used_for_multiple_requests(client):
    first = client.post(
        "/api/chat",
        json={
            "message": "Where is ORD-1007?",
            "session_id": "same-session",
        },
    )

    second = client.post(
        "/api/chat",
        json={
            "message": "When will it arrive?",
            "session_id": "same-session",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200