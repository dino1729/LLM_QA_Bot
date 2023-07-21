# Use an official Python runtime as a base image
FROM python:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN python -m pip install --no-cache-dir --upgrade pip

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements_lite.txt

# Expose the port that Gradio uses (default is 7860)
EXPOSE 7860

# Command to run your Gradio app when the container starts
CMD ["python", "azure_gradioui_lite.py"]
