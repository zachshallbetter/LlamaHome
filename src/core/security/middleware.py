from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel
import jwt
from fastapi import Request, Response, HTTPException
import logging
from ratelimit import RateLimit

class SecurityConfig(BaseModel):
    jwt_secret: str
    rate_limit_requests: int = 100
    rate_limit_period: int = 60
    max_request_size: int = 1024 * 1024  # 1MB
    allowed_origins: list[str] = ["*"]

class AuditLog(BaseModel):
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource: str
    status: str
    details: Dict

class SecurityMiddleware:
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.rate_limiter = RateLimit()
        self.logger = logging.getLogger("security")

    async def process_request(self, request: Request) -> Response:
        """Process and validate incoming requests."""
        try:
            # Check request size
            await self._validate_request_size(request)
            
            # Validate origin
            await self._validate_origin(request)
            
            # Authenticate request
            user = await self._authenticate_request(request)
            
            # Check rate limit
            await self._check_rate_limit(request, user)
            
            # Validate input
            await self._validate_input(request)
            
            # Log request
            await self._audit_log(request, user, "request_processed", "success")
            
            return await self._process_response(request)

        except HTTPException as e:
            await self._audit_log(
                request, 
                None, 
                "request_failed", 
                "error",
                {"error": str(e)}
            )
            raise e

    async def _validate_request_size(self, request: Request) -> None:
        """Validate request size against configured limit."""
        content_length = request.headers.get("content-length", 0)
        if int(content_length) > self.config.max_request_size:
            raise HTTPException(
                status_code=413,
                detail="Request too large"
            )

    async def _validate_origin(self, request: Request) -> None:
        """Validate request origin against allowed origins."""
        origin = request.headers.get("origin")
        if origin and self.config.allowed_origins != ["*"]:
            if origin not in self.config.allowed_origins:
                raise HTTPException(
                    status_code=403,
                    detail="Origin not allowed"
                )

    async def _authenticate_request(self, request: Request) -> Optional[Dict]:
        """Authenticate request using JWT."""
        auth_header = request.headers.get("authorization")
        if not auth_header:
            return None

        try:
            token = auth_header.split(" ")[1]
            return jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=["HS256"]
            )
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )

    async def _check_rate_limit(
        self,
        request: Request,
        user: Optional[Dict]
    ) -> None:
        """Check request against rate limits."""
        key = user["id"] if user else request.client.host
        if not self.rate_limiter.is_allowed(
            key,
            self.config.rate_limit_requests,
            self.config.rate_limit_period
        ):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )

    async def _validate_input(self, request: Request) -> None:
        """Validate request input data."""
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.json()
                # Add specific validation rules here
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid request data"
                )

    async def _audit_log(
        self,
        request: Request,
        user: Optional[Dict],
        action: str,
        status: str,
        details: Optional[Dict] = None
    ) -> None:
        """Log security audit information."""
        log = AuditLog(
            timestamp=datetime.utcnow(),
            user_id=user["id"] if user else None,
            action=action,
            resource=str(request.url),
            status=status,
            details=details or {}
        )
        self.logger.info(log.json())

    async def _process_response(self, request: Request) -> Response:
        """Process and enhance response with security headers."""
        response = Response()
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response 