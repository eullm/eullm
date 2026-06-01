#!/usr/bin/env bash
# ─── Setup MinIO for EuLLM CI sccache backend ────────────────────────────
#
# Run as root on the VPS (Ubuntu 24.04). Idempotent.
#
# What it does:
#   - Installs Docker if missing
#   - Generates a self-signed TLS certificate for the configured hostname
#   - Brings up MinIO with HTTPS on port 9443 (configurable via SCCACHE_PORT)
#   - Creates the 'eullm-sccache' bucket
#   - Creates a dedicated CI service account
#   - Prints the credentials to configure GitHub Actions
#
# Why no Caddy / Let's Encrypt:
#   Many VPS already have Apache/nginx on ports 80 and 443. To avoid touching
#   the existing config, MinIO serves HTTPS directly on its own port using a
#   self-signed certificate. sccache will verify content via its credentials
#   (S3 access key + secret), so the TLS layer's purpose is just transport
#   encryption — self-signed is fine when paired with SCCACHE_S3_NO_SSL_VERIFY
#   on the client side, which we set in the workflow YAML.
#
# Reachable after setup:
#   https://<HOSTNAME>:<PORT>           → S3 API (used by sccache)
#   https://<HOSTNAME>:<CONSOLE_PORT>/  → MinIO web UI (admin only)

set -euo pipefail

DOMAIN="${SCCACHE_DOMAIN:-host19.appedevel.com}"
SCCACHE_PORT="${SCCACHE_PORT:-9443}"          # MinIO S3 API port (HTTPS)
SCCACHE_CONSOLE_PORT="${SCCACHE_CONSOLE_PORT:-9444}"  # MinIO admin console (HTTPS)
SCCACHE_DIR="/opt/sccache"
BUCKET_NAME="eullm-sccache"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: this script must be run as root (use sudo)." >&2
    exit 1
fi

# ─── 1. Install Docker if missing ────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "==> Installing Docker..."
    apt-get update -qq
    apt-get install -y -qq curl ca-certificates openssl
    curl -fsSL https://get.docker.com | sh
else
    echo "==> Docker already installed ($(docker --version))"
fi

# ─── 2. Generate or restore credentials ──────────────────────────────────
mkdir -p "$SCCACHE_DIR"
chmod 700 "$SCCACHE_DIR"
CREDS_FILE="$SCCACHE_DIR/credentials.env"

if [[ ! -f "$CREDS_FILE" ]]; then
    echo "==> Generating new credentials..."
    ROOT_USER="eullm-admin"
    ROOT_PASSWORD=$(openssl rand -base64 32 | tr -d '/+=' | head -c 32)
    CI_ACCESS_KEY=$(openssl rand -hex 12)        # 24 hex chars
    CI_SECRET_KEY=$(openssl rand -base64 32 | tr -d '/+=' | head -c 40)
    cat > "$CREDS_FILE" <<EOF
MINIO_ROOT_USER=$ROOT_USER
MINIO_ROOT_PASSWORD=$ROOT_PASSWORD
CI_ACCESS_KEY=$CI_ACCESS_KEY
CI_SECRET_KEY=$CI_SECRET_KEY
EOF
    chmod 600 "$CREDS_FILE"
else
    echo "==> Reusing existing credentials from $CREDS_FILE"
fi
# shellcheck source=/dev/null
source "$CREDS_FILE"

# ─── 3. Generate self-signed TLS certificate ─────────────────────────────
mkdir -p "$SCCACHE_DIR/certs"
chmod 700 "$SCCACHE_DIR/certs"

if [[ ! -f "$SCCACHE_DIR/certs/public.crt" ]]; then
    echo "==> Generating self-signed TLS certificate for $DOMAIN (10y validity)..."
    openssl req -x509 -nodes -days 3650 \
        -newkey rsa:2048 \
        -keyout "$SCCACHE_DIR/certs/private.key" \
        -out "$SCCACHE_DIR/certs/public.crt" \
        -subj "/CN=$DOMAIN" \
        -addext "subjectAltName=DNS:$DOMAIN" 2>/dev/null
    chmod 600 "$SCCACHE_DIR/certs/private.key"
    chmod 644 "$SCCACHE_DIR/certs/public.crt"
else
    echo "==> Reusing existing TLS certificate"
fi

# ─── 4. Write docker-compose.yml ─────────────────────────────────────────
# MinIO reads its TLS cert from /root/.minio/certs/{public.crt,private.key}
# when starting. We mount our self-signed cert into that location.

cat > "$SCCACHE_DIR/docker-compose.yml" <<COMPOSE
services:
  minio:
    image: minio/minio:latest
    container_name: eullm-minio
    restart: unless-stopped
    environment:
      MINIO_ROOT_USER: \${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: \${MINIO_ROOT_PASSWORD}
    ports:
      - "$SCCACHE_PORT:9000"
      - "$SCCACHE_CONSOLE_PORT:9001"
    volumes:
      - ./data:/data
      - ./certs:/root/.minio/certs:ro
    command: server /data --console-address ":9001"
COMPOSE

# ─── 5. Start MinIO ──────────────────────────────────────────────────────
cd "$SCCACHE_DIR"

# Clean up any partial state from a previous (failed) run before bringing up
# the new compose. Safe because /opt/sccache/data persists across recreations.
if docker ps -a --format '{{.Names}}' | grep -q '^eullm-minio$'; then
    echo "==> Removing previous eullm-minio container (data in ./data persists)..."
    docker compose down 2>/dev/null || docker rm -f eullm-minio 2>/dev/null || true
    # Also remove the orphaned Caddy container from the first (broken) run, if any
    docker rm -f eullm-caddy 2>/dev/null || true
fi

echo "==> Starting MinIO..."
docker compose --env-file credentials.env up -d
echo "==> Waiting 10s for MinIO to come up..."
sleep 10

# ─── 6. Create bucket + CI service account via mc ────────────────────────
echo "==> Configuring bucket '$BUCKET_NAME' and CI service account..."
docker exec eullm-minio sh -c "
    mc alias set local https://localhost:9000 '$MINIO_ROOT_USER' '$MINIO_ROOT_PASSWORD' --insecure
    mc mb -p --insecure local/$BUCKET_NAME 2>&1 || true
    mc admin user svcacct add --access-key '$CI_ACCESS_KEY' --secret-key '$CI_SECRET_KEY' --insecure local '$MINIO_ROOT_USER' 2>&1 || true
"

# ─── 7. Verify endpoint ──────────────────────────────────────────────────
echo "==> Verifying HTTPS endpoint..."
for i in $(seq 1 20); do
    if curl -fsS -k -o /dev/null "https://$DOMAIN:$SCCACHE_PORT/minio/health/live"; then
        echo "✓ MinIO is reachable at https://$DOMAIN:$SCCACHE_PORT"
        break
    fi
    if [[ $i -eq 20 ]]; then
        echo "⚠  Endpoint not reachable. Try:"
        echo "   curl -k https://$DOMAIN:$SCCACHE_PORT/minio/health/live"
        echo "   docker logs eullm-minio --tail 30"
        echo "   Check firewall: ufw allow $SCCACHE_PORT/tcp"
    fi
    sleep 1
done

# ─── 8. Print credentials to copy to GitHub repo secrets ─────────────────
cat <<INFO

╔══════════════════════════════════════════════════════════════════════════╗
║   EuLLM CI sccache backend — READY                                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   S3 endpoint    : https://$DOMAIN:$SCCACHE_PORT
║   Bucket         : $BUCKET_NAME
║   TLS            : self-signed (sccache configured to skip verification) ║
║                                                                          ║
║   Copy these to GitHub repo secrets:                                     ║
║   (Repo Settings → Secrets and variables → Actions → New repo secret)    ║
║                                                                          ║
║   SCCACHE_ENDPOINT      : https://$DOMAIN:$SCCACHE_PORT
║   SCCACHE_BUCKET        : $BUCKET_NAME
║                                                                          ║
║   AWS_ACCESS_KEY_ID     : $CI_ACCESS_KEY
║   AWS_SECRET_ACCESS_KEY : $CI_SECRET_KEY
║                                                                          ║
║   Admin console: https://$DOMAIN:$SCCACHE_CONSOLE_PORT/
║     username: $MINIO_ROOT_USER
║     password: $MINIO_ROOT_PASSWORD
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Credentials saved to: $CREDS_FILE  (root-only, mode 600)

Firewall reminder — if ufw is enabled, allow the ports:
    ufw allow $SCCACHE_PORT/tcp
    ufw allow $SCCACHE_CONSOLE_PORT/tcp   # optional, for admin console only

INFO

