import cv2
import numpy as np
from deepface import DeepFace

file_bytes = open("dummy/WhatsApp Image 2026-06-17 at 17.26.06.jpeg", "rb").read()
nparr = np.frombuffer(file_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

max_dimension = 320
h, w = img.shape[:2]
if h > max_dimension or w > max_dimension:
    scaling_factor = max_dimension / float(max(h, w))
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
brightness = np.mean(gray)
print(f"Brightness: {brightness}")

if brightness < 40:
    print("Rejected < 40")

if brightness < 90:
    print("CLAHE Applied")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

face_objs = DeepFace.extract_faces(
    img_path=img,
    anti_spoofing=True,
    enforce_detection=False,
    detector_backend="opencv"
)

primary_face = face_objs[0]
print("is_real:", primary_face.get("is_real"))
print("antispoof_score:", primary_face.get("antispoof_score"))
