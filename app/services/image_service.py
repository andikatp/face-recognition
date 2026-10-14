from typing import Tuple, Optional
import cv2
import numpy as np

from app.config import logger, MAX_DIMENSION


def decode_image_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decodes raw image bytes to an OpenCV BGR numpy array.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def downscale_image(img: np.ndarray, max_dim: int = MAX_DIMENSION) -> np.ndarray:
    """
    Downscales large phone photos to max_dim to avoid OOM while preserving face resolution.
    """
    h, w = img.shape[:2]
    if h > max_dim or w > max_dim:
        scaling_factor = max_dim / float(max(h, w))
        return cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img


def preprocess_image(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Combined preprocessing: computes brightness and applies CLAHE only for genuinely
    dark scenes (< 45). Skips deblur — MiniFASNet doesn't benefit from sharpened edges
    and preserving raw camera texture reduces false spoofing rejections.
    Returns the processed image and the original measured brightness.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    logger.info(f"💡 Image Brightness: {brightness:.2f}")

    if 15 <= brightness < 45:
        logger.info("🔧 Applying CLAHE Auto-Enhancement for low light...")
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip_limit = 3.0 if brightness < 30 else 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        new_brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        logger.info(f"✨ Enhanced Brightness: {new_brightness:.2f} (clipLimit: {clip_limit})")

    return img, brightness
