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
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

ARG FEATURES=""

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/

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
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false eullm
USER eullm

COPY --from=builder /build/target/release/eullm /usr/local/bin/eullm

# Default model storage
VOLUME ["/models", "/data/audit"]
ENV EULLM_MODELS_DIR=/models
ENV EULLM_AUDIT_DIR=/data/audit

EXPOSE 11434

# Docker sends SIGTERM on `docker stop` — the engine handles it gracefully.
STOPSIGNAL SIGTERM

# Health check: verify the API is responding.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:11434/api/version || exit 1

ENTRYPOINT ["eullm"]
CMD ["serve"]
