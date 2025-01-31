"""Tests for the LlamaHome REST API interface."""

from typing import Any, Dict, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient, Response

from src.interfaces.api import app, create_app

# Create test client
app = create_app()
client = TestClient(app)


@pytest.fixture
async def async_client() -> AsyncClient:
    """Create async test client.
    
    Returns:
        AsyncClient: Async test client
    """
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


def test_health_check() -> None:
    """Test the health check endpoint."""
    response = client.get("/api/health_check")
    assert response.status_code == 200
    data = cast(Dict[str, Any], response.json())
    assert data["status"] == "success"
    assert data["message"] == "API is healthy"
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["uptime"], float)


def test_load_model() -> None:
    """Test loading a model."""
    config = {"model_path": "test/model.bin"}
    response = client.post("/api/load_model", json=config)
    assert response.status_code == 200
    data = cast(Dict[str, Any], response.json())
    assert data["status"] == "success"
    assert data["message"] == "Model loaded successfully"


def test_process_prompt() -> None:
    """Test prompt processing."""
    request = {"prompt": "Test prompt"}
    response = client.post("/api/process_prompt", json=request)
    assert response.status_code == 200
    data = cast(Dict[str, Any], response.json())
    assert "response" in data
    assert "tokens_used" in data
    assert "processing_time" in data
    assert "model_used" in data
    assert data["model_used"] == "llama3.3"


@pytest.mark.asyncio
async def test_websocket(async_client: AsyncClient) -> None:
    """Test WebSocket connection and messaging.
    
    Args:
        async_client: Async test client
    """
    async with async_client.websocket_connect("/api/ws") as websocket:
        await websocket.send_text("Test message")
        data = await websocket.receive_text()
        assert data == "Message received: Test message"
