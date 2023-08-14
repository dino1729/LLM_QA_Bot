# Use an official Python runtime as a base image
FROM python:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required system dependencies
RUN \
    set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    python3-venv \
    ffmpeg \
    git \
    ca-certificates \
    libasound2 \
    wget \
    ; \
    rm -rf /var/lib/apt/lists/*

# Download and install OpenSSL
RUN wget -O - https://www.openssl.org/source/openssl-1.1.1u.tar.gz | tar zxf - \
    && cd openssl-1.1.1u \
    && ./config --prefix=/usr/local \
    && make -j $(nproc) \
    && make install_sw install_ssldirs

# Update library cache
RUN ldconfig -v

# Set SSL_CERT_DIR environment variable
ENV SSL_CERT_DIR=/etc/ssl/certs

# Install required Python dependencies
RUN pip install -U pip && pip install -U wheel && pip install -U setuptools==59.5.0
COPY ./requirements_lite.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm -r /tmp/requirements.txt

# Expose the port that Gradio uses (default is 7860)
EXPOSE 7860

# Command to run your Gradio app when the container starts
CMD ["python", "azure_gradioui_lite.py"]
