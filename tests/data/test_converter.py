"""Tests for format converter module."""

import json
import csv
from pathlib import Path

import pytest

from src.data.converter import FormatConverter, create_converter


@pytest.fixture
def test_data_dir(tmp_path_factory):
    """Create and return a temporary test data directory."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def converter():
    """Create a FormatConverter instance for testing."""
    return create_converter()


def test_converter_init():
    """Test FormatConverter initialization."""
    converter = FormatConverter()
    assert converter.config == {}
    assert len(converter.supported_formats) == 4
    assert all(fmt in converter.supported_formats for fmt in [".csv", ".json", ".txt", ".pdf"])

    config = {"test": "config"}
    converter = FormatConverter(config)
    assert converter.config == config


def test_process_csv(converter, test_data_dir):
    """Test CSV file processing."""
    input_file = test_data_dir / "test.csv"
    output_file = test_data_dir / "test.jsonl"
    
    test_data = [
        {"name": "John", "age": "30"},
        {"name": "Jane", "age": "25"}
    ]
    
    with open(input_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "age"])
        writer.writeheader()
        writer.writerows(test_data)
        
    assert converter.process_file(input_file, output_file)
    
    with open(output_file) as f:
        result = [json.loads(line) for line in f]
    assert result == test_data


def test_process_json(converter, test_data_dir):
    """Test JSON file processing."""
    input_file = test_data_dir / "test.json"
    output_file = test_data_dir / "test.jsonl"
    
    test_data = [
        {"id": 1, "value": "test1"},
        {"id": 2, "value": "test2"}
    ]
    
    with open(input_file, "w") as f:
        json.dump(test_data, f)
        
    assert converter.process_file(input_file, output_file)
    
    with open(output_file) as f:
        result = [json.loads(line) for line in f]
    assert result == test_data


def test_unsupported_format(converter, test_data_dir):
    """Test handling of unsupported file formats."""
    input_file = test_data_dir / "test.xyz"
    output_file = test_data_dir / "test.jsonl"
    
    with open(input_file, "w") as f:
        f.write("test")
        
    with pytest.raises(ValueError, match="Unsupported file format"):
        converter.process_file(input_file, output_file)


def test_missing_file(converter, test_data_dir):
    """Test handling of missing input files."""
    input_file = test_data_dir / "nonexistent.txt"
    output_file = test_data_dir / "test.jsonl"
    
    with pytest.raises(FileNotFoundError):
        converter.process_file(input_file, output_file)


def test_process_directory(converter, test_data_dir):
    """Test directory processing."""
    input_dir = test_data_dir / "input"
    output_dir = test_data_dir / "output"
    input_dir.mkdir()
    
    # Create test files
    (input_dir / "test1.txt").write_text("Test 1")
    (input_dir / "test2.txt").write_text("Test 2")
    (input_dir / "test.json").write_text('{"key": "value"}')
    
    results = converter.process_directory(input_dir, output_dir)
    
    assert len(results["successful"]) == 3
    assert len(results["failed"]) == 0
    assert (output_dir / "test1.jsonl").exists()
    assert (output_dir / "test2.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
