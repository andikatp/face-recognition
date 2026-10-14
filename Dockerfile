# ─────────────────────────────────────────────
# Stage 1 — Builder
#   Installs all Python dependencies and applies
#   the DeepFace patch in an isolated layer.
# ─────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.12-slim AS builder

WORKDIR /install

# Create a virtual environment and make it the default
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip, install everything with pinned versions, then patch DeepFace
RUN pip install --no-cache-dir --only-binary :all: "pip==25.0.1" && \
    pip install --no-cache-dir --only-binary :all: --default-timeout=1000 \
        "fastapi==0.143.0" \
        "python-multipart==0.0.32" \
        "uvicorn==0.54.0" \
        "deepface[tensorflow]==0.0.102" \
        "numpy==2.5.4" \
        "tensorflow-cpu==2.21.0" && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir --only-binary :all: "opencv-python-headless==4.10.0.84" && \
    LOCATION=$(pip show --path deepface 2>/dev/null | head -1 || \
               python -c "import deepface, os; print(os.path.dirname(deepface.__file__) + '/..')") && \
    PATCH_FILE=$(find /opt/venv -path "*/deepface/commons/package_utils.py" | head -1) && \
    sed -i '/def validate_for_keras3() -> None:/a\    return' "$PATCH_FILE" && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "*.pyo" -delete && \
    # Explicitly ensure haarcascades are exactly where DeepFace expects them
    mkdir -p /opt/venv/lib/python3.12/site-packages/cv2/data && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml', '/opt/venv/lib/python3.12/site-packages/cv2/data/haarcascade_frontalface_default.xml')" && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml', '/opt/venv/lib/python3.12/site-packages/cv2/data/haarcascade_eye.xml')"

# ─────────────────────────────────────────────
# Stage 2 — Runtime
#   Minimal image: only the installed packages
#   and application source code.
# ─────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.12-slim AS runtime

WORKDIR /app

# Install dependencies, setup non-root user, and download model weights in a merged layer (docker:S7018, S7031, S6471)
ENV DEEPFACE_HOME=/home/appuser
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser \
    && mkdir -p /home/appuser/.deepface/weights \
    && curl --proto "=https" --tlsv1.2 -fsSL -o /home/appuser/.deepface/weights/face_detection_yunet_2023mar.onnx \
        https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    && curl --proto "=https" --tlsv1.2 -fsSL -o /home/appuser/.deepface/weights/2.7_80x80_MiniFASNetV2.h5 \
        https://github.com/serengil/deepface_models/releases/download/v1.0/2.7_80x80_MiniFASNetV2.h5 \
    && curl --proto "=https" --tlsv1.2 -fsSL -o /home/appuser/.deepface/weights/4_0_0_80x80_MiniFASNetV1SE.h5 \
        https://github.com/serengil/deepface_models/releases/download/v1.0/4_0_0_80x80_MiniFASNetV1SE.h5 \
    && chown -R appuser:appuser /home/appuser/.deepface

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate virtual environment in runtime
ENV PATH="/opt/venv/bin:$PATH"

# Copy explicit application modules (avoids recursive COPY . . - fixes SonarCloud CRITICAL docker:S6470)
COPY app ./app
COPY main.py .

RUN chown -R appuser:appuser /app

# Runtime tuning for constrained environments (e.g. Render free tier 512 MB)
ENV MALLOC_ARENA_MAX=2 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_NUM_INTEROP_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Tell Docker to ping the health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/ || exit 1

# Run container as non-root user
USER appuser

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1} --timeout-keep-alive 5"]
