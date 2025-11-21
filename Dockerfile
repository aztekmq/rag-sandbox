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
# Use verbose logging during dependency installation to aid debugging in CI/CD
# and container build workflows, following international documentation standards.
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
