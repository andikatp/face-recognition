# Use an official Python runtime as a parent image
FROM python:3.14-slim

# Set the working directory in the container
WORKDIR /app

# (No apt-get needed because we will use opencv-python-headless)

# Copy the requirements file into the container
COPY requirements.txt .

# Optimize memory allocation for 512MB RAM (Render Free Tier)
ENV MALLOC_ARENA_MAX=2
ENV PYTHONUNBUFFERED=1

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# DeepFace forces `opencv-python` which breaks on Linux without GUI libraries.
# We uninstall it and force-reinstall the headless version to fix the corrupted cv2 folder.
RUN pip uninstall -y opencv-python opencv-python-headless
RUN pip install --force-reinstall --no-cache-dir "opencv-python-headless<5.0.0"

# Patch DeepFace so it doesn't crash asking for `tf-keras` on the newest TensorFlow
RUN sed -i '/def validate_for_keras3() -> None:/a \ \ \ \ return' $(python -c "import os, deepface; print(os.path.join(os.path.dirname(deepface.__file__), 'commons', 'package_utils.py'))")

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
