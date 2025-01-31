"""Security middleware for request processing."""

import logging
from datetime import datetime
from typing import Any, TypeVar

import jwt
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel
from ratelimit import RateLimit

T = TypeVar("T")


class SecurityConfig(BaseModel):
    """Security configuration."""

    jwt_secret: str
    rate_limit_requests: int = 100
    rate_limit_period: int = 60
    max_request_size: int = 1024 * 1024  # 1MB
    allowed_origins: list[str] = ["*"]


class AuditLog(BaseModel):
    """Audit log entry."""

    timestamp: datetime
    user_id: str | None
    action: str
    resource: str
    status: str
    details: dict[str, Any]


class SecurityMiddleware:
    """Security middleware for request processing."""

    def __init__(self, config: SecurityConfig) -> None:
        """Initialize security middleware.

        Args:
            config: Security configuration
        """
        self.config = config
        self.rate_limiter = RateLimit()
        self.logger = logging.getLogger("security")

    async def process_request(self, request: Request) -> Response:
        """Process and validate incoming requests.

        Args:
            request: FastAPI request object

        Returns:
            Processed response

        Raises:
            HTTPException: If request validation fails
        """
        try:
            await self._validate_request_size(request)
            await self._validate_origin(request)
            user = await self._authenticate_request(request)
            await self._check_rate_limit(request, user)
            await self._validate_input(request)
            await self._audit_log(request, user, "request_processed", "success")
            return await self._process_response(request)
        except HTTPException as e:
            await self._audit_log(
                request, None, "request_failed", "error", {"error": str(e)}
            )
            raise e from None

    async def _validate_request_size(self, request: Request) -> None:
        """Validate request size against configured limit.

        Args:
            request: FastAPI request object

        Raises:
            HTTPException: If request is too large
        """
        content_length = request.headers.get("content-length", 0)
        if int(content_length) > self.config.max_request_size:
            raise HTTPException(status_code=413, detail="Request too large")

    async def _validate_origin(self, request: Request) -> None:
        """Validate request origin against allowed origins.

        Args:
            request: FastAPI request object

        Raises:
            HTTPException: If origin is not allowed
        """
        origin = request.headers.get("origin")
        if origin and self.config.allowed_origins != ["*"]:
            if origin not in self.config.allowed_origins:
                raise HTTPException(status_code=403, detail="Origin not allowed")

    async def _authenticate_request(self, request: Request) -> dict[str, Any] | None:
        """Authenticate request using JWT.

        Args:
            request: FastAPI request object

        Returns:
            Decoded JWT payload if authentication succeeds

        Raises:
            HTTPException: If authentication fails
        """
        auth_header = request.headers.get("authorization")
        if not auth_header:
            return None

        try:
            token = auth_header.split(" ")[1]
            return jwt.decode(token, self.config.jwt_secret, algorithms=["HS256"])
        except Exception as e:
            raise HTTPException(
                status_code=401, detail="Invalid authentication token"
            ) from e

    async def _check_rate_limit(
        self, request: Request, user: dict[str, Any] | None
    ) -> None:
        """Check request against rate limits.

        Args:
            request: FastAPI request object
            user: Authenticated user information

        Raises:
            HTTPException: If rate limit is exceeded
        """
        key = user["id"] if user else request.client.host
        if not self.rate_limiter.is_allowed(
            key, self.config.rate_limit_requests, self.config.rate_limit_period
        ):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    async def _validate_input(self, request: Request) -> None:
        """Validate request input data.

        Args:
            request: FastAPI request object

        Raises:
            HTTPException: If input data is invalid
        """
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                await request.json()
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail="Invalid request data"
                ) from e

    async def _audit_log(
        self,
        request: Request,
        user: dict[str, Any] | None,
        action: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log security audit information.

        Args:
            request: FastAPI request object
            user: Authenticated user information
            action: Audit action
            status: Audit status
            details: Additional audit details
        """
        log = AuditLog(
            timestamp=datetime.utcnow(),
            user_id=user["id"] if user else None,
            action=action,
            resource=str(request.url),
            status=status,
            details=details or {},
        )
        self.logger.info(log.json())

    async def _process_response(self, request: Request) -> Response:
        """Process and enhance response with security headers.

        Args:
            request: FastAPI request object

        Returns:
            Enhanced response with security headers
        """
        response = Response()
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers[
            "Strict-Transport-Security"
        ] = "max-age=31536000; includeSubDomains"
        return response
