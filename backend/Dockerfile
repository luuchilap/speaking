# Use a lightweight Python 3.10 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install System Dependencies
# ffmpeg is CRITICAL for Whisper and Librosa to process audio files
# git is often needed for installing specific pip packages
# build-essential, cmake, and ninja-build are required for praat-parselmouth compilation
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY requirements.txt .
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# 3. Pre-download Whisper model during build to avoid runtime network issues
RUN python -c "import whisper; whisper.load_model('base')"

# 4. Copy the Application Code
# We copy this last so that if you change code, Docker doesn't re-install requirements
COPY audio_analysis_service.py .

# 5. Expose the port
EXPOSE 8000

# 6. Run the application
# host 0.0.0.0 is required for Docker containers to be accessible from outside
CMD ["uvicorn", "audio_analysis_service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]