# EULLM Hub — multi-stage Rust build
# Usage:
#   docker build -t eullm-hub .

# ── Stage 1: Build ──────────────────────────────────────────────
FROM rust:1.83-bookworm AS builder

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/

RUN cargo build --release

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false eullm
USER eullm

COPY --from=builder /build/target/release/eullm-hub /usr/local/bin/eullm-hub

# Model storage
VOLUME ["/models"]
ENV EULLM_HUB_STORAGE=/models

EXPOSE 3000

ENTRYPOINT ["eullm-hub"]
