from fastapi.testclient import TestClient
from gradio_ui_full import app, parse_model_name
import os
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_frontend_serve():
    # Only passes if build exists
    if os.path.exists("frontend/dist/index.html"):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

def test_model_parse_logic():
    # parse_model_name returns (provider, tier, model_name)
    assert parse_model_name("LITELLM_SMART") == ("litellm", "smart", None)
    assert parse_model_name("OLLAMA_FAST") == ("ollama", "fast", None)
    assert parse_model_name("GEMINI") == ("litellm", "fast", None)
    # Test dynamic format: PROVIDER:model_name
    assert parse_model_name("LITELLM:gpt-4") == ("litellm", "smart", "gpt-4")
    assert parse_model_name("OLLAMA:llama2") == ("ollama", "smart", "llama2")

def test_reset_endpoint():
    response = client.post("/api/docqa/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "Database reset"

