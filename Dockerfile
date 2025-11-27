# syntax=docker/dockerfile:1
# Multi-stage build – now with properly compiled, AVX2-accelerated llama.cpp

FROM python:3.11-slim AS builder
WORKDIR /app

# Install build tools needed to compile llama-cpp-python with full CPU acceleration
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: Set CMake flags to force AVX2 + FMA + F16C (this is what fixes the 223s prefill!)
ENV CMAKE_ARGS="-DLLAMA_AVX2=on -DLLAMA_FMA=on -DLLAMA_F16C=on -DLLAMA_AVX=on -DCMAKE_BUILD_TYPE=Release"
ENV FORCE_CMAKE=1

# First install ONLY llama-cpp-python with full CPU optimization
# This runs cmake + make under the hood with the flags above
COPY requirements.txt .

# Use AVX2/FMA/F16C-optimized llama.cpp bindings to avoid scalar fallbacks that
# cause multi-minute prefill times on modern CPUs. FORCE_CMAKE guarantees a
# source build so the CMAKE_ARGS take effect even when prebuilt wheels exist.
# Verbose logging remains enabled to align with the repository's debugging
# posture and international documentation standards.
ENV CMAKE_ARGS="-DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_F16C=on -DLLAMA_FMA=on -DCMAKE_BUILD_TYPE=Release"
ENV FORCE_CMAKE=1
RUN pip install --verbose --no-cache-dir -r requirements.txt

# Now install the rest of the dependencies (much faster, no recompilation needed)
RUN pip install --no-cache-dir --verbose -r requirements.txt

# ——————— Runtime stage ———————
FROM python:3.11-slim

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
# Force CPU-only execution to avoid noisy GPU discovery warnings inside minimal
# containers. Verbose logging remains available in the app for debugging.
ENV CUDA_VISIBLE_DEVICES="" \
    ORT_DEVICE_ALLOWLIST="cpu"
WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code and data
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY requirements.txt README.md ./

EXPOSE 7860

# Start the Gradio app
CMD ["python", "-m", "app.main"]