"""Tests for the LlamaHome REST API interface."""

import pytest
from fastapi.testclient import TestClient
from src.interfaces.api import app, ModelConfig, ProcessRequest

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/api/health_check")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "API is healthy"
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["uptime"], (int, float))


def test_load_model():
    """Test loading a model."""
    config = ModelConfig(model_path="test/model.bin")
    response = client.post("/api/load_model", json=config.dict())
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Model loaded successfully"


def test_process_prompt():
    """Test prompt processing."""
    request = ProcessRequest(prompt="Test prompt")
    response = client.post("/api/process_prompt", json=request.dict())
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "tokens_used" in data
    assert "processing_time" in data
    assert "model_used" in data
    assert data["model_used"] == "llama3.3"


@pytest.mark.asyncio
async def test_websocket():
    """Test WebSocket connection and messaging."""
    with client.websocket_connect("/api/ws") as websocket:
        websocket.send_text("Test message")
        data = websocket.receive_text()
        assert data == "Message received: Test message"
