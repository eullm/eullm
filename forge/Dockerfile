# EULLM Forge — Python + PyTorch on NVIDIA CUDA
# Usage:
#   docker build -t eullm-forge .
#   docker build -t eullm-forge --build-arg CUDA_VERSION=12.4.1 .
#
# Run:
#   docker run --gpus all eullm-forge forge Qwen/Qwen3-14B --profile legal-it

ARG CUDA_VERSION=12.4.1

# ── Stage 1: Base with CUDA ─────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# ── Stage 2: Install dependencies ──────────────────────────────
WORKDIR /app

COPY pyproject.toml ./
COPY eullm_forge/ eullm_forge/

# Install PyTorch with CUDA support, then the package
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -e ".[distill]"

# ── Runtime ─────────────────────────────────────────────────────
VOLUME ["/models", "/output", "/data"]
ENV HF_HOME=/data/huggingface

ENTRYPOINT ["eullm-forge"]
CMD ["--help"]
