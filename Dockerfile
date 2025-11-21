# syntax=docker/dockerfile:1
# Multi-stage build to keep runtime image minimal while providing
# all dependencies for the RAG sandbox application.

FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

# Install lightweight system dependencies required by llama.cpp and docling
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

CMD ["python", "app/main.py"]
