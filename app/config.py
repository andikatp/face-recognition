import os
import logging

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
# YuNet is more robust than Haar for off-center / slightly blurry faces.
DETECTOR_BACKEND: str = os.getenv("DETECTOR_BACKEND", "yunet")
MAX_DIMENSION: int = int(os.getenv("MAX_DIMENSION", "640"))
BLUR_VARIANCE_THRESHOLD: float = float(os.getenv("BLUR_VARIANCE_THRESHOLD", "60.0"))
MIN_FACE_CONFIDENCE: float = float(os.getenv("MIN_FACE_CONFIDENCE", "0.4"))
MAX_PAYLOAD_BYTES: int = 5 * 1024 * 1024  # 5MB
