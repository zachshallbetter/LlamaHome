# LlamaHome API Guide

## API Overview

### Core Concepts

1. **Architecture**

   ```mermaid
   graph TD
       A[Client] --> B[API]
       B --> C[Model]
   ```

2. **Key Features**
   - RESTful endpoints
   - WebSocket support
   - Streaming responses
   - Rate limiting
   - Authentication
   - Monitoring

## Quick Start

### Basic Usage

1. **Authentication**

   ```python
   from llamahome.client import APIClient
   
   client = APIClient(
       api_key="your_api_key",
       endpoint="https://api.llamahome.ai"
   )
   ```

2. **Simple Request**

   ```python
   response = await client.process_prompt(
       prompt="Summarize this text",
       model="llama3.3",
       max_tokens=100
   )
   print(response.text)
   ```

### Streaming Responses

1. **Async Stream**

   ```python
   async for chunk in client.stream_response(
       prompt="Generate a story",
       model="llama3.3"
   ):
       print(chunk.text, end="", flush=True)
   ```

2. **Batch Processing**

   ```python
   results = await client.process_batch(
       prompts=["Query 1", "Query 2", "Query 3"],
       model="llama3.3",
       batch_size=3
   )
   ```

## API Reference

### Core Endpoints

1. **Process Prompt**

   ```http
   POST /api/v1/process
   Content-Type: application/json
   Authorization: Bearer <api_token>
   
   {
     "prompt": "string (required)",
     "model": "string (optional, default: llama3.3)",
     "max_tokens": "integer (optional, default: 100)",
     "temperature": "float (optional, default: 0.7)",
     "stream": "boolean (optional, default: false)"
   }
   ```

   Response:

   ```json
   {
     "text": "Generated response",
     "usage": {
       "prompt_tokens": 10,
       "completion_tokens": 50,
       "total_tokens": 60
     },
     "model": "llama3.3",
     "created": "2024-03-15T12:00:00Z"
   }
   ```

2. **Stream Response**

   ```http
   POST /api/v1/stream
   Content-Type: application/json
   Authorization: Bearer <api_token>
   
   {
     "prompt": "string (required)",
     "model": "string (optional)",
     "max_tokens": "integer (optional)"
   }
   ```

   Response Stream:

   ```json
   {"chunk": "First", "index": 0}
   {"chunk": "part", "index": 1}
   {"chunk": "of response", "index": 2}
   ```

### Model Management

1. **List Models**

   ```http
   GET /api/v1/models
   Authorization: Bearer <api_token>
   ```

   Response:

   ```json
   {
     "models": [
       {
         "id": "llama3.3-7b",
         "name": "Llama 3.3 7B",
         "version": "3.3",
         "parameters": "7B",
         "context_length": 32768
       }
     ]
   }
   ```

2. **Model Information**

   ```http
   GET /api/v1/models/{model_id}
   Authorization: Bearer <api_token>
   ```

   Response:

   ```json
   {
     "id": "llama3.3-7b",
     "name": "Llama 3.3 7B",
     "version": "3.3",
     "parameters": "7B",
     "context_length": 32768,
     "capabilities": [
       "text-generation",
       "summarization",
       "translation"
     ],
     "performance": {
       "tokens_per_second": 100,
       "memory_required": "8GB"
     }
   }
   ```

### Configuration

1. **Update Settings**

   ```http
   POST /api/v1/config
   Content-Type: application/json
   Authorization: Bearer <api_token>
   
   {
     "model_settings": {
       "default_model": "llama3.3",
       "max_tokens": 2000,
       "temperature": 0.7
     },
     "system_settings": {
       "cache_size": "10GB",
       "max_requests_per_minute": 60
     }
   }
   ```

2. **Get Settings**

   ```http
   GET /api/v1/config
   Authorization: Bearer <api_token>
   ```

## Integration Patterns

### Client Integration

1. **Python Client**

   ```python
   from llamahome.client import LlamaClient
   
   class CustomClient:
       def __init__(self, api_key: str):
           self.client = LlamaClient(api_key=api_key)
           
       async def process_with_retry(
           self,
           prompt: str,
           max_retries: int = 3
       ) -> str:
           """Process prompt with retry logic."""
           for attempt in range(max_retries):
               try:
                   response = await self.client.process(prompt)
                   return response.text
               except Exception as e:
                   if attempt == max_retries - 1:
                       raise
                   await asyncio.sleep(2 ** attempt)
   ```

2. **JavaScript Client**

   ```javascript
   class LlamaClient {
     constructor(apiKey) {
       this.apiKey = apiKey;
       this.baseUrl = 'https://api.llamahome.ai';
     }
     
     async processPrompt(prompt, options = {}) {
       const response = await fetch(`${this.baseUrl}/api/v1/process`, {
         method: 'POST',
         headers: {
           'Authorization': `Bearer ${this.apiKey}`,
           'Content-Type': 'application/json'
         },
         body: JSON.stringify({
           prompt,
           ...options
         })
       });
       return response.json();
     }
   }
   ```

### Server Integration

1. **FastAPI Server**

   ```python
   from fastapi import FastAPI, Depends
   from llamahome.server import LlamaServer
   
   app = FastAPI()
   llama = LlamaServer()
   
   @app.post("/process")
   async def process_prompt(
       prompt: str,
       current_user = Depends(get_current_user)
   ):
       return await llama.process(prompt)
   ```

2. **Express Server**

   ```javascript
   const express = require('express');
   const { LlamaServer } = require('llamahome');
   
   const app = express();
   const llama = new LlamaServer();
   
   app.post('/process', async (req, res) => {
     const result = await llama.process(req.body.prompt);
     res.json(result);
   });
   ```

## Security

### Authentication

1. **Token Generation**

   ```python
   from llamahome.auth import TokenGenerator
   
   generator = TokenGenerator(secret_key="your-secret")
   token = generator.create_token(
       user_id="user123",
       expires_in=3600
   )
   ```

2. **Token Validation**

   ```python
   from llamahome.auth import TokenValidator
   
   validator = TokenValidator(secret_key="your-secret")
   is_valid = validator.validate_token(token)
   ```

### Rate Limiting

1. **Basic Rate Limiting**

   ```python
   from llamahome.security import RateLimiter
   
   limiter = RateLimiter(
       requests_per_minute=60,
       burst_size=10
   )
   ```

2. **Advanced Rate Limiting**

   ```python
   from llamahome.security import AdvancedRateLimiter
   
   limiter = AdvancedRateLimiter(
       tiers={
           "basic": {"rpm": 60, "burst": 10},
           "pro": {"rpm": 300, "burst": 50},
           "enterprise": {"rpm": 1000, "burst": 100}
       }
   )
   ```

## Monitoring

### Metrics Collection

1. **Basic Metrics**

   ```python
   from llamahome.monitoring import MetricsCollector
   
   collector = MetricsCollector()
   collector.record_request(
       endpoint="/api/v1/process",
       duration=0.123,
       status=200
   )
   ```

2. **Advanced Metrics**

   ```python
   from llamahome.monitoring import AdvancedMetrics
   
   metrics = AdvancedMetrics(
       enable_tracing=True,
       detailed_logging=True
   )
   ```

### Performance Monitoring

1. **Response Time Tracking**

   ```python
   from llamahome.monitoring import PerformanceMonitor
   
   monitor = PerformanceMonitor()
   with monitor.track_operation("process_prompt"):
       result = await process_prompt()
   ```

2. **Resource Usage Tracking**

   ```python
   from llamahome.monitoring import ResourceMonitor
   
   monitor = ResourceMonitor()
   monitor.track_resources(
       interval=60,
       metrics=["cpu", "memory", "gpu"]
   )
   ```

## Error Handling

### Error Types

1. **API Errors**

   ```python
   class APIError(Exception):
       def __init__(self, message: str, code: int):
           self.message = message
           self.code = code
   
   class RateLimitError(APIError):
       pass
   
   class AuthenticationError(APIError):
       pass
   ```

2. **Error Responses**

   ```json
   {
     "error": {
       "code": "rate_limit_exceeded",
       "message": "Too many requests",
       "details": {
         "retry_after": 60
       }
     }
   }
   ```

### Error Recovery

1. **Retry Logic**

   ```python
   from llamahome.error import RetryHandler
   
   handler = RetryHandler(
       max_retries=3,
       backoff_factor=2
   )
   ```

2. **Circuit Breaker**

   ```python
   from llamahome.error import CircuitBreaker
   
   breaker = CircuitBreaker(
       failure_threshold=5,
       reset_timeout=300
   )
   ```

## Best Practices

### API Usage

1. **Request Optimization**

   ```python
   # Good: Batch related requests
   results = await client.batch_process([
       "Query 1",
       "Query 2",
       "Query 3"
   ])
   
   # Bad: Multiple individual requests
   result1 = await client.process("Query 1")
   result2 = await client.process("Query 2")
   result3 = await client.process("Query 3")
   ```

2. **Resource Management**

   ```python
   # Good: Use context managers
   async with client.session() as session:
       result = await session.process(prompt)
   
   # Bad: Manual resource management
   session = await client.create_session()
   result = await session.process(prompt)
   await session.close()
   ```

### Performance

1. **Connection Pooling**

   ```python
   from llamahome.client import PooledClient
   
   client = PooledClient(
       pool_size=10,
       max_retries=3
   )
   ```

2. **Response Streaming**

   ```python
   async for chunk in client.stream_response(
       prompt,
       chunk_size=1000
   ):
       process_chunk(chunk)
   ```

## Next Steps

1. [API Examples](docs/Examples.md)
2. [Integration Guide](docs/Integration.md)
3. [Security Guide](docs/Security.md)
4. [Monitoring Guide](docs/Monitoring.md)
