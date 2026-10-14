from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.config import logger, MAX_PAYLOAD_BYTES
from app.schemas import VerifyResponse, ErrorResponse
from app.services.image_service import (
    decode_image_bytes,
    downscale_image,
    apply_mild_deblur_if_soft,
    check_and_enhance_lighting,
)
from app.services.liveness_service import evaluate_liveness

router = APIRouter(prefix="/api/v1", tags=["Verification"])


@router.post(
    "/verify-face",
    response_model=VerifyResponse,
    responses={
        200: {"model": VerifyResponse, "description": "Verification outcome (success or rejected)"},
        400: {"model": ErrorResponse, "description": "Invalid input format or unreadable image bytes"},
        413: {"model": ErrorResponse, "description": "Payload size exceeds 5MB limit"},
        500: {"model": ErrorResponse, "description": "Internal server error during analysis"},
    },
)
async def verify_face(
    file: Annotated[UploadFile, File(..., description="Selfie / face photo to verify")],
    threshold: Annotated[float, Form(description="Anti-spoofing fakeness threshold (0.0 to 1.0)")] = 0.6,
):
    logger.info(f"--- 📸 New Verification Request | Target Threshold: {threshold} ---")

    # 1. Enforce payload constraints
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid data type. Must submit a valid image format.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_PAYLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large. Maximum size is 5MB.")

    # 2. Decode raw bytes to OpenCV image matrix
    img = decode_image_bytes(file_bytes)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image payload. Could not decode bytes.")

    # 3. Downscale massive camera captures & restore edges if soft
    img = downscale_image(img)
    img = apply_mild_deblur_if_soft(img)

    # 4. Check brightness & enhance lighting
    img, brightness = check_and_enhance_lighting(img)

    # Early rejection for pitch black frames (< 15)
    if brightness < 15:
        logger.warning(f"❌ Rejected: Image is pitch black ({brightness:.2f} < 15)")
        return JSONResponse(
            status_code=200,
            content={
                "status": "rejected",
                "message": "Ruangan terlalu gelap. Silakan cari tempat yang lebih terang.",
            },
        )

    # 5. Execute liveness detection & antispoofing
    try:
        is_success, message, antispoof_score = evaluate_liveness(img, brightness, threshold)
        status_str = "success" if is_success else "rejected"
        response_payload = {"status": status_str, "message": message}
        if antispoof_score is not None:
            response_payload["antispoof_score"] = antispoof_score

        return JSONResponse(status_code=200, content=response_payload)

    except ValueError as val_err:
        logger.warning(f"Validation notice during analysis: {val_err}")
        return JSONResponse(status_code=200, content={"status": "rejected", "message": str(val_err)})
    except Exception:
        logger.exception("Internal error occurred during liveness analysis")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Terjadi kesalahan internal pada server saat analisis."},
        )
