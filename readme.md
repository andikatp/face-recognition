# Liveness Detection Engine API

This is a lightweight face verification and anti-spoofing API built with FastAPI, OpenCV, and DeepFace. It automatically downscales large images, applies auto-brightness enhancement (CLAHE) for low-light photos, and calculates a normalized 0-100% "realness" score.

## 🚀 1. How to Setup & Run Locally (Python)

1. **Activate your virtual environment & install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   venv/bin/python main.py
   ```
   The server will start at `http://localhost:8000` with hot-reloading enabled.

## 🐳 2. How to Run via Docker (For Production)

If you want to test exactly how it runs on Render or AWS:

1. **Build the image:**
   ```bash
   docker build -t liveness-engine .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 liveness-engine
   ```

---

## 🧪 3. How to Test via Postman

### Endpoint 1: Health Check
Use this to quickly check if your server is online.
- **Method:** `GET`
- **URL:** `http://localhost:8000/`
- **Response:** `{"message": "Welcome to Liveness Detection Engine API", "status": "running"}`

### Endpoint 2: Face Verification (Liveness Detection)
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/v1/verify-face`
- **Body:** Select **`form-data`**

**Required Form-Data Keys:**
1. **Key:** `file`
   - **Type:** `File` (Change the field type from 'Text' to 'File' in Postman)
   - **Value:** *Select an image from your computer*
2. **Key:** `threshold`
   - **Type:** `Text`
   - **Value:** `0.7` *(Accepts values from 0.0 to 1.0)*

### 📝 Understanding the Results
The `antispoof_score` returned by this API is a pure **Realness Score** between `0.0` and `1.0`. 
- `0.95` means the AI is 95% confident the face is a real person.
- `0.10` means the AI is 10% confident the face is real (i.e. it is 90% sure it's a spoof/screen).

If the `antispoof_score` is strictly greater than or equal to your requested `threshold`, the API will return `"status": "success"`. Otherwise, it will return `"status": "rejected"`.
