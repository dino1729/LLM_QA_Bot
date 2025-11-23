FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies and cleanup in single layer to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

# Copy application code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "gradio_ui_full.py"]
