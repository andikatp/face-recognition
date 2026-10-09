# ─────────────────────────────────────────────
# Stage 1 — Builder
#   Installs all Python dependencies and applies
#   the DeepFace patch in an isolated layer.
# ─────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.10-slim AS builder

WORKDIR /install

# Copy only the dependency manifest first (better layer caching)
COPY requirements.txt .

# Create a virtual environment and make it the default
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip, install everything, then patch DeepFace
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt && \
    LOCATION=$(pip show --path deepface 2>/dev/null | head -1 || \
               python -c "import deepface, os; print(os.path.dirname(deepface.__file__) + '/..')") && \
    PATCH_FILE=$(find /opt/venv -path "*/deepface/commons/package_utils.py" | head -1) && \
    sed -i '/def validate_for_keras3() -> None:/a\    return' "$PATCH_FILE" && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "*.pyo" -delete && \
    # Explicitly ensure haarcascades are exactly where DeepFace expects them
    mkdir -p /opt/venv/lib/python3.10/site-packages/cv2/data && \
    curl -fsSL -o /opt/venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml \
         https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml && \
    curl -fsSL -o /opt/venv/lib/python3.10/site-packages/cv2/data/haarcascade_eye.xml \
         https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml

# ─────────────────────────────────────────────
# Stage 2 — Runtime
#   Minimal image: only the installed packages
#   and application source code.
# ─────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.10-slim AS runtime

WORKDIR /app

# Install system dependencies required by OpenCV (even headless needs some)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libxcb1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate virtual environment in runtime
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source (excludes whatever is in .dockerignore)
COPY . .

# Runtime tuning for constrained environments (e.g. Render free tier 512 MB)
ENV MALLOC_ARENA_MAX=2 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_NUM_INTEROP_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "5"]
