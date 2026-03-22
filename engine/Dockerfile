# EULLM Engine — multi-stage Rust build
# Usage:
#   docker build -t eullm-engine .
#   docker build -t eullm-engine --build-arg FEATURES=cuda .

# ── Stage 1: Build ──────────────────────────────────────────────
FROM rust:1.83-bookworm AS builder

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ARG FEATURES=""

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/

# Limit CUDA architectures to reduce binary size (~940MB → ~200MB)
# sm_80=A100, sm_86=RTX3000, sm_89=RTX4000, sm_90=H100
ENV CMAKE_CUDA_ARCHITECTURES="80;86;89;90"

# Build with optional GPU features
RUN if [ -z "$FEATURES" ]; then \
        cargo build --release; \
    else \
        cargo build --release --features "$FEATURES"; \
    fi

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false eullm
USER eullm

COPY --from=builder /build/target/release/eullm /usr/local/bin/eullm

# Default model storage
VOLUME ["/models", "/data/audit"]
ENV EULLM_MODELS_DIR=/models
ENV EULLM_AUDIT_DIR=/data/audit

EXPOSE 11434

ENTRYPOINT ["eullm"]
CMD ["serve"]
