# EuLLM Engine — CPU image, multi-stage Rust build.
#
# Built from the repository root, not from engine/: the engine compiles in the
# model catalog (catalog/v1/catalog.json), llama.cpp from engine/vendor and the
# workspace's Cargo.lock, so a context of engine/ alone could never build.
# llama.cpp is a git submodule and has to be checked out first:
#
#   git submodule update --init --recursive
#   docker build -f engine/Dockerfile -t eullm/engine .
#   docker compose build engine
#
# NVIDIA GPUs: engine/Dockerfile.cuda (`docker compose --profile gpu ...`).

# ── Stage 1: Build ──────────────────────────────────────────────
# rust:1, not a pinned minor: Cargo.toml declares edition 2024, which needs
# cargo 1.85 or newer, and 1.83 could not parse the manifest at all.
FROM rust:1-bookworm AS builder

# cmake and a C++ compiler build llama.cpp; libclang is bindgen's. No
# OpenSSL: the engine's HTTPS is rustls.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# The workspace's manifest and lockfile, every member (cargo reads the hub's
# manifest even to build the engine alone), and the catalog the engine embeds.
COPY Cargo.toml Cargo.lock ./
COPY catalog/ catalog/
COPY engine/ engine/
COPY hub/ hub/

# --locked: the versions CI tested, not whatever resolves today. The cache
# mounts keep the crates and llama.cpp's objects from one build to the next,
# so the binary is copied out of them in the same step.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --locked -p eullm-engine \
    && cp target/release/eullm /usr/local/bin/eullm

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM debian:bookworm-slim

# libgomp1: llama.cpp's CPU backend runs on OpenMP; beside the C and C++
# runtimes it is the only library the binary loads. curl: the health check.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# The volumes are created here, owned by the user the engine runs as: a
# volume Docker creates itself is root's, and an audit directory the engine
# cannot write to stops it at startup (EULLM_AUDIT_DIR, set below, is a
# control that must not be silently off). The user is 10001 in every EuLLM
# image: the compose file shares the `models` volume between them.
RUN groupadd -r -g 10001 eullm \
    && useradd -r -u 10001 -g eullm -s /usr/sbin/nologin eullm \
    && mkdir -p /models /data/audit \
    && chown eullm:eullm /models /data/audit
USER eullm

COPY --from=builder /usr/local/bin/eullm /usr/local/bin/eullm

VOLUME ["/models", "/data/audit"]
ENV EULLM_MODELS_DIR=/models
ENV EULLM_AUDIT_DIR=/data/audit

# The API, and the chat UI (`serve --no-ui` turns it off).
EXPOSE 11434 11435

# Docker sends SIGTERM on `docker stop` — the engine handles it gracefully.
STOPSIGNAL SIGTERM

# Health check: verify the API is responding.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:11434/api/version || exit 1

ENTRYPOINT ["eullm"]
# Without a command `eullm` prints its usage and exits: a container that
# stopped as soon as it started. `docker run eullm/engine run <model>` and
# any other subcommand still work.
CMD ["serve"]
