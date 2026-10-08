from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from deepface import DeepFace
import cv2
import numpy as np
import os
import pandas as pd

DB_PATH = "./employee_db"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Triggers automatically when the server boots.
    Ensures directory paths exist and initializes the DeepFace model cache to RAM.
    """
    print("🤖 Booting System: Compiling employee biometric database...")
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
        print(f"📁 Created empty registration folder at: {DB_PATH}")
    
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        try:
            # Force-load model cache weights using a blank dummy target matrix
            dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
            DeepFace.find(img_path=dummy_img, db_path=DB_PATH, model_name="ArcFace", enforce_detection=False)
            print("✅ In-Memory Biometric Registry pre-compiled successfully.")
        except Exception as e:
            print(f"⚠️ Initial cache warm-up notice: {str(e)}")
    else:
        print("⚠️ Database directory is empty. Add employee photos named by ID (e.g. EMP_101.jpg).")
    yield
    print("🔌 Shutting down Biometric System...")

app = FastAPI(title="Biometric Attendance Engine", lifespan=lifespan)

@app.post("/api/v1/verify-face")
async def verify_face(file: UploadFile = File(...)):
    # 1. Enforce payload constraints
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid data type. Must submit a valid image format.")

    try:
        # 2. Decode incoming file stream directly inside RAM
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Corrupted image packet data.")

        # 3. PHASE 1: PASSIVE LIVENESS CHECK (Anti-Spoofing Model Execution)
        face_objs = DeepFace.extract_faces(
            img_path=img, 
            anti_spoofing=True, 
            enforce_detection=False
        )

        if not face_objs:
            return JSONResponse(status_code=200, content={"status": "rejected", "reason": "No face detected in frame."})

        # Extract primary localized face structure
        primary_face = face_objs[0]
        is_real = primary_face.get("is_real", False)

        if not is_real:
            return JSONResponse(
                status_code=200, 
                content={
                    "status": "rejected", 
                    "reason": "Spoof attack blocked. Photo or digital screen detected."
                }
            )

        # 4. PHASE 2: FACE RECOGNITION (ArcFace Feature Vector Lookup)
        matched_dfs = DeepFace.find(
            img_path=img, 
            db_path=DB_PATH, 
            model_name="ArcFace",
            enforce_detection=False
        )

        if len(matched_dfs) > 0 and not matched_dfs[0].empty:
            # Grab top prioritized matching dataframe target file path string
            best_match_df = matched_dfs[0]
            file_path = best_match_df.iloc[0]['identity']
            
            # Isolate base filename to extract Employee ID target identifier string
            employee_id = os.path.splitext(os.path.basename(file_path))[0]
            
            return {
                "status": "success",
                "employee_id": employee_id,
                "message": "Liveness validated and identity verified successfully."
            }

        return JSONResponse(status_code=200, content={"status": "rejected", "reason": "Identity not found in system."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution pipeline error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
