# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# (No apt-get needed because we will use opencv-python-headless)

# Copy the requirements file into the container
COPY requirements.txt .

# Optimize memory allocation for 512MB RAM (Render Free Tier)
ENV MALLOC_ARENA_MAX=2
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1

# Upgrade pip, install packages, and patch DeepFace in a SINGLE layer to save space
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt && \
    LOCATION=$(pip show deepface | awk '/^Location:/ {print $2}') && \
    sed -i '/def validate_for_keras3() -> None:/a \ \ \ \ return' $LOCATION/deepface/commons/package_utils.py && \
    find /usr/local/lib/python3.10/site-packages/ -name "__pycache__" -type d -exec rm -rf {} +

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "5"]
