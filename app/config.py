import logging
import os

# Suppress verbose TensorFlow logs and DeepFace deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPFACE_LOG_LEVEL"] = "40"  # ERROR level only
# Lower the default 0.9 cutoff so low-end phone cameras still detect a face
os.environ.setdefault("yunet_score_threshold", "0.5")

# --- CONFIGURE PROFESSIONAL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LivenessAPI")

# --- CORE TUNING CONSTANTS ---
APP_VERSION: str = os.getenv("APP_VERSION", "1.0.1")
# YuNet is more robust than Haar for off-center / slightly blurry faces.
DETECTOR_BACKEND: str = os.getenv("DETECTOR_BACKEND", "yunet")
MAX_DIMENSION: int = int(os.getenv("MAX_DIMENSION", "640"))
MIN_FACE_CONFIDENCE: float = float(os.getenv("MIN_FACE_CONFIDENCE", "0.35"))
MAX_PAYLOAD_BYTES: int = 5 * 1024 * 1024  # 5MB

# --- REDIS CONFIGURATION ---
# Optional: Leave unset or empty for offline development without Redis/Auth
REDIS_URL: str = os.getenv("REDIS_URL", "").strip()


# --- ENVIRONMENT & CORS CONFIGURATION ---
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
# Comma-separated list of allowed origins, or '*' for local development
ALLOWED_ORIGINS: list[str] = [
    origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()
]

# --- JWT AUTHENTICATION CONFIGURATION ---
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production-secret-key")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
# Check Redis for token blacklist/revocation or active session key
JWT_CHECK_REDIS: bool = os.getenv("JWT_CHECK_REDIS", "true").lower() in ("true", "1", "yes")
# Redis key prefix for revoked tokens or active sessions (e.g. "blacklist:<jti>" or "session:<user_id>")
JWT_REDIS_BLACKLIST_PREFIX: str = os.getenv("JWT_REDIS_BLACKLIST_PREFIX", "jwt:blacklist:")

if ENVIRONMENT == "production" and JWT_SECRET_KEY == "change-me-in-production-secret-key":
    logger.critical(
        "🚨 CRITICAL SECURITY WARNING: Running in production with default JWT_SECRET_KEY!"
    )
