# syntax=docker/dockerfile:1
# Multi-stage build – GPU-friendly (ONNX + llama-cpp-python) for NVIDIA GPUs

FROM python:3.11-slim AS builder
WORKDIR /app

# Install build tools and system deps used by various Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    libgomp1 \
    libopenblas-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# IMPORTANT: GPU-aware Python dependencies
#
# You should ensure your requirements.txt uses:
#   - onnxruntime-gpu   (instead of onnxruntime)
#   - llama-cpp-python[cuda]  (instead of plain llama-cpp-python)
#
# Example lines in requirements.txt:
#   onnxruntime-gpu
#   "llama-cpp-python[cuda]"
#
# The prebuilt CUDA wheels will use your host GPU via nvidia-container-runtime,
# so we do NOT need to compile llama.cpp with CUDA here.
# -------------------------------------------------------------------

COPY requirements.txt .

# Install core Python dependencies (including GPU variants)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --verbose -r requirements.txt && \
    # Make absolutely sure GPU ONNX runtime is present (overrides CPU build if any)
    pip install --no-cache-dir --upgrade onnxruntime-gpu

# ——————— Runtime stage ———————
FROM python:3.11-slim

WORKDIR /app

# Runtime system dependencies (same as builder, but no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# DON’T disable CUDA here — let ONNX / llama-cpp see the GPU if Docker passes it through.
# If you ever need to force CPU, you can override at runtime:
#   -e CUDA_VISIBLE_DEVICES=""
#   -e ORT_DISABLE_GPU=1
#
# ENV CUDA_VISIBLE_DEVICES=""
# ENV ORT_DEVICE_ALLOWLIST="cpu"

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and data
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY requirements.txt README.md ./

EXPOSE 7860

# Start the Gradio app
CMD ["python", "-m", "app.main"]