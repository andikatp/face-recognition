import jwt
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import (
    JWT_ALGORITHM,
    JWT_CHECK_REDIS,
    JWT_REDIS_BLACKLIST_PREFIX,
    JWT_SECRET_KEY,
    REDIS_URL,
    logger,
)
from app.services.redis_service import get_redis

# Paths that do not require JWT authentication
EXEMPT_PATHS = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that intercepts incoming HTTP requests, extracts the Bearer token,
    validates the JWT signature & expiration, and checks Redis for token revocation.
    If REDIS_URL is not set, auth is completely bypassed for offline testing.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # 1. Skip authentication if Redis is not configured (offline mode), or for exempt paths
        if not REDIS_URL or path in EXEMPT_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        # 2. Extract Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "message": "Authorization header missing or invalid format. Expected 'Bearer <token>'.",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.split(" ", 1)[1].strip()

        # 3. Decode & verify JWT signature
        try:
            payload = jwt.decode(
                token,
                JWT_SECRET_KEY,
                algorithms=[JWT_ALGORITHM],
            )
        except jwt.ExpiredSignatureError:
            logger.warning("🔒 [Middleware] Auth failed: Token has expired")
            return JSONResponse(
                status_code=401,
                content={"status": "error", "message": "Token expired. Silakan login kembali."},
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as err:
            logger.warning(f"🔒 [Middleware] Auth failed: Invalid token ({err})")
            return JSONResponse(
                status_code=401,
                content={"status": "error", "message": "Token autentikasi tidak valid."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 4. Check Redis for token blacklist/revocation
        if JWT_CHECK_REDIS:
            redis_client = get_redis()
            if redis_client:
                token_identifier = payload.get("jti") or token
                blacklist_key = f"{JWT_REDIS_BLACKLIST_PREFIX}{token_identifier}"
                try:
                    is_blacklisted = await redis_client.exists(blacklist_key)
                    if is_blacklisted:
                        logger.warning(
                            f"🔒 [Middleware] Rejected: Token {token_identifier} is blacklisted in Redis"
                        )
                        return JSONResponse(
                            status_code=401,
                            content={"status": "error", "message": "Token telah dicabut / logout."},
                            headers={"WWW-Authenticate": "Bearer"},
                        )
                except Exception as exc:
                    logger.error(f"⚠️ [Middleware] Redis check error: {exc}")

        # 5. Attach decoded payload to request state for downstream handlers
        request.state.user = payload

        return await call_next(request)
