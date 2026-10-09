import gc
import os
import traceback
import cv2
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Suppress verbose TensorFlow logs and DeepFace deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPFACE_LOG_LEVEL"] = "40" # ERROR level only
from deepface import DeepFace

# --- CONFIGURE PROFESSIONAL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LivenessAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the lightweight liveness model into RAM.
    """
    logger.info("🤖 Booting System: Compiling Lightweight Liveness Engine...")
    try:
        # Force-load liveness model weights using a blank dummy target matrix
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.extract_faces(
            img_path=dummy_img,
            anti_spoofing=True,
            enforce_detection=False,
            detector_backend="opencv"
        )
        logger.info("✅ Liveness Engine pre-compiled successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Initial cache warm-up notice: {str(e)}")
    yield
    logger.info("🔌 Shutting down Liveness Engine...")


app = FastAPI(title="Liveness Detection Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log full trace server-side only — never expose internals to clients
    logger.error(f"GLOBAL EXCEPTION: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Terjadi kesalahan internal pada server."}
    )


@app.get("/")
async def root():
    """Root endpoint to check if the API is online."""
    return {"message": "Welcome to Liveness Detection Engine API", "status": "running"}


@app.post("/api/v1/verify-face")
async def verify_face(
    file: UploadFile = File(...),
    threshold: float = Form(0.6)
):
    logger.info(f"--- 📸 New Verification Request | Target Threshold: {threshold} ---")
    # 1. Enforce payload constraints
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid data type. Must submit a valid image format.")

    file_bytes = await file.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Payload too large. Maximum size is 5MB.")

    # 2. Convert to OpenCV format, then immediately free the raw bytes
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    del file_bytes, nparr  # Free raw bytes — img is all we need from here
    gc.collect()

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image payload. Could not decode bytes.")

    # 3. CRITICAL OOM PREVENTION: Downscale massive iPhone/Android photos
    max_dimension = 320
    h, w = img.shape[:2]
    if h > max_dimension or w > max_dimension:
        scaling_factor = max_dimension / float(max(h, w))
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # 3.5. BRIGHTNESS CHECK & AUTO-ENHANCEMENT
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    logger.info(f"💡 Image Brightness: {brightness:.2f}")

    # 1. Graceful Rejection: If the image is extremely dark, reject it early.
    if brightness < 40:
        logger.warning("❌ Rejected: Image is too dark (< 40)")
        return JSONResponse(
            status_code=200,
            content={
                "status": "rejected",
                "message": "Ruangan terlalu gelap. Silakan cari tempat yang lebih terang."
            }
        )

    # 2. Auto-Enhancement: If it's moderately dark, enhance the contrast/brightness
    if brightness < 90:
        logger.info("🔧 Applying CLAHE Auto-Enhancement for low light...")
        # Convert to LAB color space to modify just the Lightness channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Log the new brightness
        new_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        logger.info(f"✨ Enhanced Brightness: {new_brightness:.2f}")

    try:
        # 4. LIVENESS DETECTION ONLY
        face_objs = DeepFace.extract_faces(
            img_path=img,
            anti_spoofing=True,
            enforce_detection=False,
            detector_backend="opencv"
        )

        if not face_objs:
            return JSONResponse(status_code=200, content={"status": "rejected", "message": "Tidak ada wajah yang terdeteksi."})

        # Extract primary localized face structure
        primary_face = face_objs[0]

        # If confidence is very low, a face wasn't truly found
        confidence = primary_face.get("confidence", 0)
        if confidence < 0.5:
            return JSONResponse(status_code=200, content={"status": "rejected", "message": "Wajah tidak terdeteksi dengan jelas."})

        deepface_is_real = primary_face.get("is_real", False)
        deepface_score = float(primary_face.get("antispoof_score", 0.0))
        
        logger.info(f"🤖 DeepFace Raw Output -> is_real: {deepface_is_real}, confidence: {deepface_score:.4f}")

        # Calculate a pure "FAKENESS" score (0.0 to 1.0)
        # If deepface predicts spoof, its score is the fakeness confidence.
        # If deepface predicts real, its score is the realness confidence, so fakeness is 1 - score.
        if not deepface_is_real:
            fakeness_score = deepface_score
        else:
            fakeness_score = 1.0 - deepface_score

        # Round to 4 decimal places to prevent ugly scientific notation (like 4.17e-7)
        fakeness_score = round(fakeness_score, 4)
        logger.info(f"🧮 Calculated Fakeness Score: {fakeness_score:.4f}")

        # Your exact frontend logic: If fakeness is ABOVE the threshold, it is fake.
        is_real = fakeness_score < threshold
        
        logger.info(f"⚖️ Final Decision -> Accepted: {is_real} (Max Fakeness Allowed: {threshold})")
        
        # We override antispoof_score so the frontend always sees the pure "fakeness" percentage
        antispoof_score = fakeness_score

        if not is_real:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "rejected",
                    "message": "Verifikasi gagal. Foto palsu atau layar digital terdeteksi.",
                    "antispoof_score": antispoof_score
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Verifikasi berhasil. Wajah asli terdeteksi.",
                "antispoof_score": antispoof_score
            }
        )

    except ValueError as ve:
        logger.warning(f"Validation Error: {str(ve)}")
        return JSONResponse(status_code=200, content={"status": "rejected", "message": str(ve)})
    except Exception as e:
        logger.error(f"Internal Error during analysis: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Terjadi kesalahan internal pada server saat analisis."})


if __name__ == "__main__":
    import uvicorn
    # This allows you to just run `python main.py` directly to start the server!
    logger.info("🚀 Starting local development server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
