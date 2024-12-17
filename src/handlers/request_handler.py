"""Request handling and queue management for LlamaHome."""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional
import time

from .model_handler import ModelHandler
from utils.log_manager import LogManager, LogTemplates

logger = LogManager(LogTemplates.SYSTEM_STARTUP).get_logger(__name__)


@dataclass
class Request:
    """Request data container."""

    id: str
    prompt: str
    config: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    priority: int = 0

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Response:
    """Response data container."""

    request_id: str
    content: str
    created_at: datetime = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()


class RequestHandler:
    """Handles request processing and queue management."""

    def __init__(
        self,
        model_handler: ModelHandler,
        max_queue_size: int = 100,
        request_timeout: float = 30.0,
    ) -> None:
        """Initialize request handler.

        Args:
            model_handler: ModelHandler instance
            max_queue_size: Maximum number of pending requests
            request_timeout: Default timeout for requests in seconds
        """
        self.model = model_handler
        self.max_queue_size = max_queue_size
        self.request_timeout = request_timeout

        self._request_queue: asyncio.Queue[Request] = asyncio.Queue(maxsize=max_queue_size)
        self._responses: Dict[str, Response] = {}
        self._active = False
        self._worker_task = None
        self._lock = asyncio.Lock()
        self._routes = {}
        self.logger = LogManager().get_logger("request_handler", "app", "access")

    async def start(self) -> None:
        """Start the request handler."""
        async with self._lock:
            if self._active:
                return

            logger.info("Starting request handler")
            self._active = True
            self._worker_task = asyncio.create_task(self._process_queue())

    async def stop(self) -> None:
        """Stop the request handler."""
        async with self._lock:
            if not self._active:
                return

            logger.info("Stopping request handler")
            self._active = False
            if self._worker_task:
                await self._worker_task
                self._worker_task = None

    async def submit_request(
        self, prompt: str, config: Optional[Dict[str, Any]] = None, priority: int = 0
    ) -> str:
        """Submit a new request.

        Args:
            prompt: Input prompt
            config: Optional configuration overrides
            priority: Request priority (higher values = higher priority)

        Returns:
            Request ID
        """
        if not self._active:
            raise RuntimeError("Request handler is not running")

        request = Request(id=str(uuid.uuid4()), prompt=prompt, config=config, priority=priority)

        try:
            await self._request_queue.put(request)
            logger.info(
                LogTemplates.REQUEST_RECEIVED.format(source="submit_request", request_id=request.id)
            )
            return request.id
        except asyncio.QueueFull:
            raise RuntimeError("Request queue is full")

    async def process_request(
        self, prompt: str, timeout: Optional[float] = None, config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a request and wait for response.

        Args:
            prompt: Input prompt
            timeout: Optional timeout override
            config: Optional configuration overrides

        Returns:
            Generated response text
        """
        request_id = await self.submit_request(prompt, config)
        timeout = timeout or self.request_timeout

        try:
            start_time = asyncio.get_event_loop().time()
            while True:
                if request_id in self._responses:
                    response = self._responses[request_id]
                    if response.error:
                        raise RuntimeError(response.error)
                    return response.content

                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(f"Request {request_id} timed out")

                await asyncio.sleep(0.1)

        finally:
            # Cleanup response
            self._responses.pop(request_id, None)

    async def stream_responses(
        self, request_id: str, timeout: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response chunks for a request.

        Args:
            request_id: Request ID to stream
            timeout: Optional timeout override

        Yields:
            Response text chunks
        """
        if not self._active:
            raise RuntimeError("Request handler is not running")

        timeout = timeout or self.request_timeout
        start_time = asyncio.get_event_loop().time()

        try:
            async for chunk in self.model.stream_response_async(
                self._find_request(request_id).prompt
            ):
                yield chunk

                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(f"Stream {request_id} timed out")

        except Exception as e:
            logger.exception(
                LogTemplates.REQUEST_FAILED.format(request_id=request_id, error=str(e))
            )
            raise

    def _find_request(self, request_id: str) -> Request:
        """Find a request by ID.

        Args:
            request_id: ID of the request to find

        Returns:
            Found request object

        Raises:
            ValueError: If request not found
        """
        for item in self._request_queue._queue:
            if isinstance(item, Request) and item.id == request_id:
                return item
        raise ValueError(f"Request {request_id} not found")

    async def _process_queue(self) -> None:
        """Process requests from the queue."""
        while self._active:
            try:
                request = await self._request_queue.get()

                try:
                    response = await self.model.generate_response_async(
                        request.prompt, request.config
                    )
                    self._responses[request.id] = Response(request_id=request.id, content=response)
                    logger.info(
                        LogTemplates.REQUEST_PROCESSED.format(
                            request_id=request.id, duration=0.0  # TODO: Add actual duration
                        )
                    )

                except Exception as e:
                    logger.exception(
                        LogTemplates.REQUEST_FAILED.format(request_id=request.id, error=str(e))
                    )
                    self._responses[request.id] = Response(
                        request_id=request.id, content="", error=str(e)
                    )

                finally:
                    self._request_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(LogTemplates.SYSTEM_ERROR.format(error=str(e)))
                continue

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._request_queue.qsize()

    @property
    def is_active(self) -> bool:
        """Check if handler is active."""
        return self._active

    def handle(self, request: Request) -> Response:
        """Handle a single request.
        
        Args:
            request: Request to process
            
        Returns:
            Response object
        """
        request_id = id(request)
        try:
            self.logger.info(LogTemplates.REQUEST_RECEIVED.format(
                source="RequestHandler",
                request_id=request_id
            ))
            
            start_time = time.time()
            response = Response(request_id=str(request_id), content="")
            
            duration = time.time() - start_time
            self.logger.info(LogTemplates.REQUEST_COMPLETED.format(
                source="RequestHandler", 
                request_id=request_id,
                duration=duration
            ))
            return response
            
        except Exception as e:
            self.logger.error(LogTemplates.REQUEST_FAILED.format(
                source="RequestHandler",
                request_id=request_id, 
                error=str(e)
            ))
            raise
