import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import (
    JWT_ALGORITHM,
    JWT_CHECK_REDIS,
    JWT_REDIS_BLACKLIST_PREFIX,
    JWT_SECRET_KEY,
    REDIS_URL,
    logger,
)
from app.services.redis_service import get_redis

# Bearer security scheme for OpenAPI docs (auto_error=False so it doesn't block when offline)
security_scheme = HTTPBearer(auto_error=False)


async def get_current_user_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> dict:
    """
    Validates JWT signature, expiration, and checks Redis for revocation / blacklist.
    If REDIS_URL is not set, returns a local mock user for easy offline development.
    """
    if not REDIS_URL:
        return {"sub": "offline-dev-user", "mode": "standalone"}

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or invalid format. Expected 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # 1. Decode & verify JWT signature and claims
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError as err:
        logger.warning("🔒 Auth failed: Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired. Silakan login kembali.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err
    except jwt.InvalidTokenError as err:
        logger.warning(f"🔒 Auth failed: Invalid token format or signature: {err}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token autentikasi tidak valid.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err

    # 2. Check token revocation / blacklist in Redis (if enabled)
    if JWT_CHECK_REDIS:
        redis_client = get_redis()
        if redis_client:
            # Check by jti (JWT ID) if present, otherwise by the token hash/string
            token_identifier = payload.get("jti") or token
            blacklist_key = f"{JWT_REDIS_BLACKLIST_PREFIX}{token_identifier}"

            try:
                is_revoked = await redis_client.exists(blacklist_key)
                if is_revoked:
                    logger.warning(
                        f"🔒 Auth rejected: Revoked token detected in Redis ({token_identifier})"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token telah dicabut / logout.",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            except HTTPException:
                raise
            except Exception as exc:
                logger.error(f"⚠️ Redis check error during token validation: {exc}")
                # Depending on strictness, we continue or fail. Here we allow fail-open or log warning.

    return payload
