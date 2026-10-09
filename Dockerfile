# ─────────────────────────────────────────────
# Stage 1 — Builder
#   Installs all Python dependencies and applies
#   the DeepFace patch in an isolated layer.
# ─────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.10-slim AS builder

WORKDIR /install

# Copy only the dependency manifest first (better layer caching)
COPY requirements.txt .

# Upgrade pip, install everything into a dedicated prefix, then patch DeepFace
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 \
        --prefix=/install/deps \
        -r requirements.txt && \
    LOCATION=$(pip show --path deepface 2>/dev/null | head -1 || \
               python -c "import deepface, os; print(os.path.dirname(deepface.__file__) + '/..')") && \
    PATCH_FILE=$(find /install/deps -path "*/deepface/commons/package_utils.py" | head -1) && \
    sed -i '/def validate_for_keras3() -> None:/a\    return' "$PATCH_FILE" && \
    find /install/deps -name "__pycache__" -type d -exec rm -rf {} + && \
    find /install/deps -name "*.pyc" -delete && \
    find /install/deps -name "*.pyo" -delete

# ─────────────────────────────────────────────
# Stage 2 — Runtime
#   Minimal image: only the installed packages
#   and application source code.
# ─────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install/deps /usr/local

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
