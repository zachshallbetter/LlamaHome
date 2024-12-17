"""Tests for the LlamaHome CLI interface."""

import asyncio
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from src.interfaces.cli import CLI, cli


@pytest.fixture
def mock_request_handler():
    """Mock request handler fixture."""
    handler = AsyncMock()
    handler.process_request = AsyncMock(return_value="Test response")
    handler.submit_request = AsyncMock(return_value="request-id")
    handler.stream_responses = AsyncMock(return_value=["Test ", "stream ", "response"])
    return handler


@pytest.fixture
def cli_interface(mock_request_handler):
    """CLI interface fixture."""
    return CLI(request_handler=mock_request_handler)


@pytest.mark.asyncio
async def test_cli_lifecycle(cli_interface):
    """Test CLI start/stop lifecycle."""
    assert not cli_interface._active
    
    await cli_interface.start()
    assert cli_interface._active
    assert cli_interface.handler.start.called
    
    await cli_interface.stop()
    assert not cli_interface._active
    assert cli_interface.handler.stop.called


@pytest.mark.asyncio
async def test_process_prompt(cli_interface):
    """Test prompt processing."""
    await cli_interface.start()
    
    # Test normal prompt
    await cli_interface.process_prompt("Test prompt", stream=False)
    cli_interface.handler.process_request.assert_called_once_with(
        "Test prompt", timeout=None, config=None
    )
    
    # Test streaming prompt
    await cli_interface.process_prompt("Test prompt", stream=True)
    cli_interface.handler.submit_request.assert_called_once_with("Test prompt", None)
    cli_interface.handler.stream_responses.assert_called_once()


def test_cli_command():
    """Test CLI command execution."""
    runner = CliRunner()
    
    with patch('src.interfaces.cli.CLI') as mock_cli:
        mock_instance = MagicMock()
        mock_cli.return_value = mock_instance
        mock_instance.start = AsyncMock()
        mock_instance.process_prompt = AsyncMock()
        mock_instance.stop = AsyncMock()
        
        result = runner.invoke(cli, ['run', '--no-stream', 'test prompt'])
        assert result.exit_code == 0


@pytest.mark.asyncio
async def test_cli_error_handling(cli_interface):
    """Test CLI error handling."""
    await cli_interface.start()
    
    cli_interface.handler.process_request.side_effect = TimeoutError()
    await cli_interface.process_prompt("Test prompt")
    
    cli_interface.handler.process_request.side_effect = Exception("Test error")
    await cli_interface.process_prompt("Test prompt")
