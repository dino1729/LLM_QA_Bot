# Use a specific Python version as the base image
FROM python:3.11.4

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file initially to leverage Docker cache
COPY requirements_lite.txt /app/

# Install required system dependencies
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    ca-certificates \
    libasound2 \
    wget \
    ; \
    rm -rf /var/lib/apt/lists/*

# Install required Python dependencies
RUN pip install -U pip && pip install -U wheel && pip install -U setuptools==59.5.0 && pip install -r requirements_lite.txt

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

# Copy the rest of the application code
COPY . /app/

# Expose the port that Gradio uses (default is 7860)
EXPOSE 7860

# Command to run your Gradio app when the container starts
CMD ["python", "azure_gradioui_lite.py"]
