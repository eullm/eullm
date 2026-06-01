#!/usr/bin/env bash
# ─── Setup MinIO + Caddy for EuLLM CI sccache backend ────────────────────
#
# Run as root on the VPS (host19.appedevel.com, Ubuntu 24.04).
# Idempotent: safe to re-run if something fails mid-way.
#
# What it does:
#   - Installs Docker + Compose if missing
#   - Generates random credentials for MinIO root and a dedicated CI service account
#   - Brings up MinIO (S3-compatible storage) + Caddy (reverse proxy with auto-HTTPS)
#   - Creates the 'eullm-sccache' bucket
#   - Prints the credentials to copy back to the CI maintainer
#
# Reachable after setup:
#   https://host19.appedevel.com   → S3 API (used by GitHub Actions sccache)
#   https://host19.appedevel.com/console/  → MinIO web UI (optional admin)

set -euo pipefail

DOMAIN="${SCCACHE_DOMAIN:-host19.appedevel.com}"
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
    apt-get install -y -qq curl ca-certificates
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
    # shellcheck source=/dev/null
    source "$CREDS_FILE"
fi
# shellcheck source=/dev/null
source "$CREDS_FILE"

# ─── 3. Write docker-compose.yml ─────────────────────────────────────────
cat > "$SCCACHE_DIR/docker-compose.yml" <<COMPOSE
services:
  minio:
    image: minio/minio:latest
    container_name: eullm-minio
    restart: unless-stopped
    environment:
      MINIO_ROOT_USER: \${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: \${MINIO_ROOT_PASSWORD}
      # Show in console where Caddy routes /console to so login works through reverse proxy
      MINIO_BROWSER_REDIRECT_URL: https://$DOMAIN/console/
    volumes:
      - ./data:/data
    command: server /data --console-address ":9001"
    networks:
      - sccache_net

  caddy:
    image: caddy:2-alpine
    container_name: eullm-caddy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    networks:
      - sccache_net
    depends_on:
      - minio

networks:
  sccache_net:
    driver: bridge

volumes:
  caddy_data:
  caddy_config:
COMPOSE

# ─── 4. Write Caddyfile (auto Let's Encrypt) ─────────────────────────────
cat > "$SCCACHE_DIR/Caddyfile" <<CADDY
$DOMAIN {
    # S3 API — sccache backend
    reverse_proxy minio:9000

    # MinIO admin console at /console (optional)
    handle_path /console/* {
        reverse_proxy minio:9001
    }
}
CADDY

# ─── 5. Start services ───────────────────────────────────────────────────
cd "$SCCACHE_DIR"
echo "==> Starting MinIO + Caddy..."
docker compose --env-file credentials.env up -d
echo "==> Waiting 15s for MinIO and Caddy to come up..."
sleep 15

# ─── 6. Create bucket + CI service account via mc ────────────────────────
echo "==> Configuring bucket '$BUCKET_NAME' and CI service account..."

docker run --rm --network sccache_sccache_net \
    -e MC_HOST_local="http://${MINIO_ROOT_USER}:${MINIO_ROOT_PASSWORD}@minio:9000" \
    minio/mc:latest sh -c "
        mc mb -p local/$BUCKET_NAME 2>&1 || true
        mc admin user svcacct add --access-key '$CI_ACCESS_KEY' --secret-key '$CI_SECRET_KEY' local '$MINIO_ROOT_USER' 2>&1 || true
        mc anonymous set none local/$BUCKET_NAME
    "

# ─── 7. Verify endpoint ──────────────────────────────────────────────────
echo "==> Verifying HTTPS endpoint (may take ~30s for Let's Encrypt to issue cert)..."
for i in $(seq 1 30); do
    if curl -fsS -o /dev/null "https://$DOMAIN/minio/health/live"; then
        echo "✓ MinIO is reachable at https://$DOMAIN"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "⚠  Endpoint not yet reachable. Check 'docker logs eullm-caddy' for Let's Encrypt errors."
        echo "   This is usually because: port 80 is blocked, or DNS not resolving to this VPS yet."
    fi
    sleep 2
done

# ─── 8. Print credentials to copy to GitHub repo secrets ─────────────────
cat <<INFO

╔══════════════════════════════════════════════════════════════════════════╗
║   EuLLM CI sccache backend — READY                                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   S3 endpoint    : https://$DOMAIN
║   Bucket         : $BUCKET_NAME
║   Region         : us-east-1 (MinIO default, accepted by sccache)
║                                                                          ║
║   Copy these 4 values to GitHub repo secrets:                            ║
║   (Repo Settings → Secrets and variables → Actions → New repo secret)    ║
║                                                                          ║
║   SCCACHE_ENDPOINT      : https://$DOMAIN
║   SCCACHE_BUCKET        : $BUCKET_NAME
║   SCCACHE_S3_KEY_PREFIX : sccache
║                                                                          ║
║   AWS_ACCESS_KEY_ID     : $CI_ACCESS_KEY
║   AWS_SECRET_ACCESS_KEY : $CI_SECRET_KEY
║                                                                          ║
║   Admin console (optional): https://$DOMAIN/console/                     ║
║     username: $MINIO_ROOT_USER
║     password: $MINIO_ROOT_PASSWORD
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Credentials saved to: $CREDS_FILE  (root-only, mode 600)

INFO
