# LlamaHome API Documentation

This document describes the LlamaHome REST API which enables communication between components
and interaction with the Llama 3.3 model.

## Overview

The LlamaHome API enables seamless integration and interaction across all components of the
system. It includes:

- **Core API**: Manages requests and responses between user interfaces (CLI/GUI) and the
  Llama 3.3 model
- **Configuration API**: Handles dynamic adjustments to settings like model parameters
- **Utility API**: Provides helper functions for logging, error handling, and performance
  monitoring

## API Structure

### Core API

Responsible for handling user requests and model interactions.

| Endpoint | Method | Description | Stream |
|----------|--------|-------------|---------|
| `/api/process` | POST | Process user prompt and return response | Yes |
| `/api/stream` | POST | Stream responses | Yes |
| `/api/models` | GET | List available models | No |
| `/api/config` | POST | Get/update configuration | No |

### Configuration API

Manages runtime adjustments to settings and parameters.

| Endpoint | Method | Description | Stream |
|----------|--------|-------------|---------|
| `/api/configure_model` | POST | Update model parameters | No |
| `/api/get_config` | GET | Get current configuration | No |
| `/api/save_config` | POST | Save configuration to disk | No |
| `/api/reset_config` | POST | Reset to default configuration | No |

### Utility API

Provides auxiliary functions to support system operations.

| Endpoint | Method | Description | Stream |
|----------|--------|-------------|---------|
| `/api/health_check` | GET | Check API and model status | No |
| `/api/logs` | GET | Get recent debug logs | Yes |
| `/api/clear_cache` | POST | Clear temporary data | No |
| `/api/metrics` | GET | Get performance metrics | Yes |
| `/api/diagnostics` | GET | Get system diagnostics | Yes |
| `/api/shutdown` | POST | Shutdown API and release resources | No |

## Detailed API Reference

### Model Management

#### POST /api/load_model

Load a specific Llama 3.3 model into memory.

**Request Body:**

```json
{
    "model_path": "string (required)"
}
```

**Response:**

```json
{
    "status": "string ('success' or 'error')",
    "message": "string"
}
```

**Errors:**

- `400`: Bad Request - Invalid or missing fields
- `404`: Not Found - Model file not found
- `500`: Internal Server Error - Loading failed

##### POST /api/unload_model

Unload the currently active model.

**Response:**

```json
{
    "status": "string ('success' or 'error')",
    "message": "string"
}
```

**Errors:**

- `404`: Not Found - No model loaded
- `500`: Internal Server Error - Unloading failed

##### POST /api/process_prompt

Process a user prompt and generate a response.

**Request Body:**

```json
{
    "prompt": "string (required)",
    "model": "string (optional, default: 'llama3.3')",
    "temperature": "float (optional, default: 0.7)",
    "max_tokens": "integer (optional, default: 100)",
    "top_p": "float (optional, default: 0.9)",
    "frequency_penalty": "float (optional, default: 0.0)",
    "presence_penalty": "float (optional, default: 0.0)"
}
```

**Response:**

```json
{
    "response": "string",
    "tokens_used": "integer",
    "processing_time": "float",
    "model_used": "string"
}
```

**Errors:**

- `400`: Bad Request - Invalid request
- `401`: Unauthorized - Invalid credentials
- `403`: Forbidden - Unauthorized model access
- `404`: Not Found - Model not found
- `408`: Request Timeout - Processing timeout
- `413`: Payload Too Large - Request too large
- `415`: Unsupported Media Type
- `422`: Unprocessable Entity
- `429`: Too Many Requests
- `500`: Internal Server Error
- `503`: Service Unavailable

### Configuration

#### POST /api/configure_model

Update model-specific parameters.

**Request Body:**

```json
{
    "model": "string (required)",
    "temperature": "float (optional)",
    "max_tokens": "integer (optional)"
}
```

**Response:**

```json
{
    "status": "string ('success' or 'error')",
    "message": "string",
    "updated_params": {
        "temperature": "float",
        "max_tokens": "integer"
    }
}
```

**Errors:**

- `400`: Bad Request - Invalid parameters
- `404`: Not Found - Model not found
- `500`: Internal Server Error

### Utilities

#### GET /api/health_check

Check API and model health status.

**Response:**

```json
{
    "status": "string ('success' or 'error')",
    "message": "string",
    "model_loaded": "boolean",
    "uptime": "integer"
}
```

**Errors:**

- `500`: Internal Server Error

#### GET /api/logs

Fetch recent system logs.

**Query Parameters:**

- `level`: string (optional) - Log level filter
- `limit`: integer (optional) - Number of logs to return
- `start_time`: string (optional) - ISO timestamp
- `end_time`: string (optional) - ISO timestamp

**Response:**
Stream of log entries in JSON format.

## Error Handling

All API endpoints follow consistent error response format:

```json
{
    "error": {
        "code": "string",
        "message": "string",
        "details": "object (optional)"
    }
}
```

## Rate Limiting

- Default: 60 requests per minute
- Streaming endpoints: 10 concurrent connections
- Batch operations: 5 requests per minute

## Authentication

All API requests require authentication:

```http
Authorization: Bearer <api_token>
```

## Versioning

The API version is specified in the URL:

```http
https://api.llamahome.ai/v1/
```

## WebSocket Support

Real-time updates available via WebSocket connection:

```javascript
ws://api.llamahome.ai/v1/ws
```

## Best Practices

1. Use appropriate error handling
2. Implement request retries with exponential backoff
3. Cache responses when appropriate
4. Monitor rate limits
5. Keep authentication tokens secure

## Implementation Timeline

### Phase 1

- Core API endpoints setup and testing
- Basic model integration and validation
- Document preprocessing pipeline

### Phase 2

- Advanced model endpoints with error handling
- Parameter tuning API with validation
- Performance monitoring and alerting

### Phase 3

- Image model endpoints with format support
- Batch processing with rate limiting
- Advanced analytics dashboard
