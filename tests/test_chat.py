from unittest.mock import AsyncMock, patch
import json
from fastapi.testclient import TestClient
from app.main import app
from app.services.llm.resilience import LLMAllProvidersFailedError

client = TestClient(app, raise_server_exceptions=False)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "configured_providers" in data


@patch("app.api.routes.chat.chat_service.get_response", new_callable=AsyncMock)
def test_valid_chat_request(mock_get_response):
    mock_get_response.return_value = (
        "Hospitals provide emergency, inpatient, and outpatient care.",
        "groq",
        "llama-3.3-70b-versatile",
    )
    resp = client.post(
        "/api/v1/chat",
        json={"message": "What services does a hospital provide?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert data["response"] == "Hospitals provide emergency, inpatient, and outpatient care."
    assert data["provider"] == "groq"
    assert data["model"] == "llama-3.3-70b-versatile"


def test_empty_message_validation():
    resp = client.post("/api/v1/chat", json={"message": ""})
    assert resp.status_code == 422


def test_whitespace_message_validation():
    resp = client.post("/api/v1/chat", json={"message": "   "})
    assert resp.status_code == 422


def test_missing_message_validation():
    resp = client.post("/api/v1/chat", json={})
    assert resp.status_code == 422


def test_invalid_data_type_validation():
    resp = client.post("/api/v1/chat", json={"message": 12345})
    assert resp.status_code == 422


@patch("app.api.routes.chat.chat_service.get_response", new_callable=AsyncMock)
def test_all_llm_providers_failed(mock_get_response):
    mock_get_response.side_effect = LLMAllProvidersFailedError("All LLM providers in fallback chain failed")
    resp = client.post("/api/v1/chat", json={"message": "hello"})
    assert resp.status_code == 503
    assert "All configured LLM providers failed" in resp.json()["detail"]


@patch("app.api.routes.chat.chat_service.stream_response")
def test_chat_stream_endpoint(mock_stream_response):
    async def fake_stream(req):
        yield 'data: {"chunk": "Hello", "provider": "groq", "model": "llama-3.3-70b-versatile", "is_final": false}\n\n'
        yield 'data: {"chunk": " world", "provider": "groq", "model": "llama-3.3-70b-versatile", "is_final": false}\n\n'
        yield 'data: {"chunk": "", "provider": "groq", "model": "llama-3.3-70b-versatile", "is_final": true}\n\n'

    mock_stream_response.side_effect = fake_stream

    resp = client.post(
        "/api/v1/chat",
        json={"message": "Hello stream", "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body = resp.text
    assert "Hello" in body
    assert " world" in body
    assert '"is_final":true' in body or '"is_final": true' in body
