import redis.asyncio as aioredis

from app.config import REDIS_URL, logger

# Global asynchronous redis client instance
redis_client: aioredis.Redis | None = None


async def init_redis() -> None:
    """Initialize Redis connection pool if REDIS_URL is configured."""
    global redis_client
    if not REDIS_URL:
        logger.info("ℹ️ REDIS_URL is not set. Running in offline/standalone mode without Redis.")
        return

    try:
        redis_client = aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
        )
        await redis_client.ping()
        logger.info(f"✅ Connected to Redis at {REDIS_URL.split('@')[-1]}")
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}. Running without active Redis connection.")


async def close_redis() -> None:
    """Close Redis connection pool gracefully."""
    global redis_client
    if redis_client:
        await redis_client.aclose()
        logger.info("🔌 Redis connection closed.")


def get_redis() -> aioredis.Redis | None:
    """Dependency helper to get the active Redis client."""
    return redis_client
