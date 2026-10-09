import cv2
import numpy as np
from deepface import DeepFace

# Load image
img_path = "dummy/WhatsApp Image 2026-06-17 at 17.26.06.jpeg"
img = cv2.imread(img_path)

if img is None:
    print("Could not read image!")
    exit(1)

# Downscale
max_dimension = 320
h, w = img.shape[:2]
if h > max_dimension or w > max_dimension:
    scaling_factor = max_dimension / float(max(h, w))
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# Brightness check
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
brightness = np.mean(gray)
print(f"Original Brightness: {brightness}")

if brightness < 90:
    print("Applying CLAHE...")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    gray_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_brightness = np.mean(gray_new)
    print(f"New Brightness: {new_brightness}")

try:
    face_objs = DeepFace.extract_faces(
        img_path=img,
        anti_spoofing=True,
        enforce_detection=False,
        detector_backend="opencv"
    )

    if not face_objs:
        print("No face detected.")
        exit(0)

    primary_face = face_objs[0]
    confidence = primary_face.get("confidence", 0)
    print(f"Face Confidence: {confidence}")

    is_real = primary_face.get("is_real", False)
    antispoof_score = float(primary_face.get("antispoof_score", 0.0))

    print(f"Is Real (raw): {is_real}")
    print(f"Antispoof Score: {antispoof_score}")

except Exception as e:
    print(f"Error: {e}")

