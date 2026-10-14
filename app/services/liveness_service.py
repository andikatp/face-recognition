from typing import Tuple, Optional
import numpy as np
from deepface import DeepFace

from app.config import logger, DETECTOR_BACKEND, MIN_FACE_CONFIDENCE


def warmup_liveness_model() -> None:
    """
    Forces pre-compilation and weight loading of the liveness and face detector models into RAM.
    """
    logger.info("🤖 Booting System: Compiling Lightweight Liveness Engine...")
    try:
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.extract_faces(
            img_path=dummy_img,
            anti_spoofing=True,
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
        )
        logger.info("✅ Liveness Engine pre-compiled successfully.")
    except Exception as exc:
        logger.warning(f"⚠️ Initial cache warm-up notice: {exc}")


def _check_detection_quality(primary_face: dict, img_shape: tuple, brightness: float) -> Optional[str]:
    """
    Checks if a face was genuinely detected by the detector or fell back to full-frame dummy area.
    Returns an error message if invalid, or None if detection passed.
    """
    confidence = float(primary_face.get("confidence") or 0)
    facial_area = primary_face.get("facial_area") or {}
    face_w = int(facial_area.get("w") or 0)
    face_h = int(facial_area.get("h") or 0)
    img_h, img_w = img_shape[:2]

    is_full_frame_fallback = face_w >= int(img_w * 0.95) and face_h >= int(img_h * 0.95)

    if confidence < MIN_FACE_CONFIDENCE or is_full_frame_fallback:
        logger.warning(
            f"❌ Rejected: Face not detected clearly (confidence={confidence:.2f}, "
            f"box={face_w}x{face_h}, fallback={is_full_frame_fallback})"
        )
        if brightness < 40:
            return "Pencahayaan terlalu redup sehingga wajah tidak terdeteksi jelas. Pastikan wajah tersorot cahaya layar/lampu."
        return "Wajah tidak terdeteksi dengan jelas. Pastikan seluruh wajah terlihat, tidak terlalu jauh, dan kamera tidak buram."

    return None


def _calculate_fakeness_score(primary_face: dict) -> float:
    """
    Normalizes the DeepFace anti-spoofing output to a pure fakeness score between 0.0 and 1.0.
    0.0 = completely real, 1.0 = completely spoof/fake screen.
    """
    deepface_is_real = primary_face.get("is_real", False)
    deepface_score = float(primary_face.get("antispoof_score", 0.0))
    logger.info(f"🤖 DeepFace Raw Output -> is_real: {deepface_is_real}, confidence: {deepface_score:.4f}")

    if not deepface_is_real:
        fakeness_score = deepface_score
    else:
        fakeness_score = 1.0 - deepface_score

    fakeness_score = round(fakeness_score, 4)
    logger.info(f"🧮 Calculated Fakeness Score: {fakeness_score:.4f}")
    return fakeness_score


def evaluate_liveness(img: np.ndarray, brightness: float, threshold: float = 0.6) -> Tuple[bool, str, Optional[float]]:
    """
    Runs face extraction and anti-spoofing on the pre-processed image.
    Returns:
        (is_accepted: bool, message: str, antispoof_score: Optional[float])
    """
    face_objs = DeepFace.extract_faces(
        img_path=img,
        anti_spoofing=True,
        enforce_detection=False,
        detector_backend=DETECTOR_BACKEND,
        align=True,
        expand_percentage=20,
    )

    if not face_objs:
        if brightness < 40:
            return False, "Ruangan terlalu gelap atau wajah tidak terdeteksi. Silakan arahkan wajah ke cahaya layar atau tempat yang lebih terang.", None
        return False, "Tidak ada wajah yang terdeteksi.", None

    primary_face = face_objs[0]

    # Validate face detection confidence
    rejection_msg = _check_detection_quality(primary_face, img.shape, brightness)
    if rejection_msg:
        return False, rejection_msg, None

    # Calculate fakeness score
    fakeness_score = _calculate_fakeness_score(primary_face)
    is_real = fakeness_score < threshold

    logger.info(f"⚖️ Final Decision -> Accepted: {is_real} (Max Fakeness Allowed: {threshold})")

    if not is_real:
        return False, "Verifikasi gagal. Foto palsu atau layar digital terdeteksi.", fakeness_score

    return True, "Verifikasi berhasil. Wajah asli terdeteksi.", fakeness_score
