#!/usr/bin/env bash
# Setup script for TurboQuant backend integration.
#
# This script prepares a local fork of llama-cpp-sys-2 that uses
# spiritbuun's TurboQuant CUDA fork of llama.cpp instead of upstream.
#
# After running this script, build eullm with:
#   cargo build --release --features "cuda turboquant"
#
# The fork is local to this workspace and does not affect other projects.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
VENDOR_DIR="${WORKSPACE_ROOT}/vendor"
SYS_CRATE_DIR="${VENDOR_DIR}/llama-cpp-sys-2"

echo "=== EULLM TurboQuant Backend Setup ==="
echo ""

# 1. Find the installed llama-cpp-sys-2 crate
CARGO_REGISTRY="${CARGO_HOME:-$HOME/.cargo}/registry/src"
INSTALLED_CRATE=$(find "$CARGO_REGISTRY" -maxdepth 2 -name "llama-cpp-sys-2-0.1.140" -type d | head -1)

if [ -z "$INSTALLED_CRATE" ]; then
    echo "ERROR: llama-cpp-sys-2 v0.1.140 not found in cargo registry."
    echo "Run 'cargo build --release' first to download it."
    exit 1
fi

echo "Found llama-cpp-sys-2 at: $INSTALLED_CRATE"

# 2. Create vendor directory and copy the crate
mkdir -p "$VENDOR_DIR"

if [ -d "$SYS_CRATE_DIR" ]; then
    echo "Vendor directory already exists: $SYS_CRATE_DIR"
    read -p "Replace it? (y/N) " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -rf "$SYS_CRATE_DIR"
fi

echo "Copying crate to vendor/..."
cp -r "$INSTALLED_CRATE" "$SYS_CRATE_DIR"

# 3. Replace the bundled llama.cpp with spiritbuun's CUDA fork
echo "Cloning spiritbuun's TurboQuant CUDA fork..."
rm -rf "${SYS_CRATE_DIR}/llama.cpp"
git clone --depth 1 --branch feature/turboquant-kv-cache \
    https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git \
    "${SYS_CRATE_DIR}/llama.cpp"

# 4. Remove .git from the cloned repo (we vendor it, not submodule it)
rm -rf "${SYS_CRATE_DIR}/llama.cpp/.git"

echo ""
echo "=== Setup complete ==="
echo ""
echo "The vendored llama-cpp-sys-2 at:"
echo "  ${SYS_CRATE_DIR}"
echo "now uses spiritbuun's TurboQuant CUDA fork of llama.cpp."
echo ""
echo "Build with:"
echo "  cd ${WORKSPACE_ROOT}/engine"
echo "  cargo build --release --features 'cuda turboquant'"
echo ""
echo "The [patch.crates-io] section in engine/Cargo.toml"
echo "redirects the dependency to this vendored copy."
