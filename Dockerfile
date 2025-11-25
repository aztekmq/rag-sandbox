# syntax=docker/dockerfile:1
# Multi-stage build to keep runtime image minimal while providing
# all dependencies for the RAG sandbox application.

FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# Use AVX2/FMA/F16C-optimized llama.cpp bindings to avoid scalar fallbacks that
# cause multi-minute prefill times on modern CPUs. FORCE_CMAKE guarantees a
# source build so the CMAKE_ARGS take effect even when prebuilt wheels exist.
# Verbose logging remains enabled to align with the repository's debugging
# posture and international documentation standards.
ENV CMAKE_ARGS="-DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_F16C=on -DLLAMA_FMA=on -DCMAKE_BUILD_TYPE=Release"
ENV FORCE_CMAKE=1
RUN pip install --verbose --no-cache-dir -r requirements.txt

FROM python:3.11-slim

# Install lightweight system dependencies required by llama.cpp and PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY requirements.txt README.md ./

EXPOSE 7860

# Run as a module to ensure the package is discoverable when the container starts.
CMD ["python", "-m", "app.main"]
