#!/usr/bin/env bash
# Setup script for TurboQuant backend integration.
#
# This script prepares a local fork of llama-cpp-sys-2 that uses
# spiritbuun's TurboQuant CUDA fork of llama.cpp instead of upstream.
#
# After running this script, build eullm with:
#   cargo build --release --features "cuda turboquant_native"
#
# The fork is local to this workspace and does not affect other projects.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(dirname "$SCRIPT_DIR")"
VENDOR_DIR="${ENGINE_DIR}/vendor"
SYS_CRATE_DIR="${VENDOR_DIR}/llama-cpp-sys-2"
# [patch.crates-io] must live in the workspace root Cargo.toml, not in the
# engine/ member manifest.  Detect workspace root by looking for [workspace].
WORKSPACE_ROOT="${ENGINE_DIR}/.."
if [ -f "${WORKSPACE_ROOT}/Cargo.toml" ] && grep -q '^\[workspace\]' "${WORKSPACE_ROOT}/Cargo.toml"; then
    CARGO_TOML="${WORKSPACE_ROOT}/Cargo.toml"
    # Path must be relative to the workspace root
    PATCH_PATH="engine/vendor/llama-cpp-sys-2"
else
    # Fallback: no workspace, patch the engine Cargo.toml directly
    CARGO_TOML="${ENGINE_DIR}/Cargo.toml"
    PATCH_PATH="vendor/llama-cpp-sys-2"
fi
CARGO_TOML="$(cd "$(dirname "$CARGO_TOML")" && pwd)/$(basename "$CARGO_TOML")"

echo "=== EULLM TurboQuant Backend Setup ==="
echo ""

# 1. Find the installed llama-cpp-sys-2 crate
CARGO_REGISTRY="${CARGO_HOME:-$HOME/.cargo}/registry/src"
INSTALLED_CRATE=$(find "$CARGO_REGISTRY" -maxdepth 2 -name "llama-cpp-sys-2-0.1.140" -type d 2>/dev/null | head -1)

if [ -z "$INSTALLED_CRATE" ]; then
    echo "ERROR: llama-cpp-sys-2 v0.1.140 not found in cargo registry."
    echo "Run 'cargo fetch' in the engine/ directory first."
    exit 1
fi

echo "Found llama-cpp-sys-2 at: $INSTALLED_CRATE"

# 2. Create vendor directory and copy the crate
mkdir -p "$VENDOR_DIR"

if [ -d "$SYS_CRATE_DIR" ]; then
    # In CI (non-interactive), always replace.
    if [ -t 0 ]; then
        read -p "Vendor directory exists. Replace it? (y/N) " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
    rm -rf "$SYS_CRATE_DIR"
fi

echo "Copying crate to vendor/..."
cp -r "$INSTALLED_CRATE" "$SYS_CRATE_DIR"
chmod -R u+w "$SYS_CRATE_DIR"

# 3. Replace the bundled llama.cpp with spiritbuun's CUDA fork
echo "Cloning spiritbuun's TurboQuant CUDA fork..."
rm -rf "${SYS_CRATE_DIR}/llama.cpp"
git clone --depth 1 --branch feature/turboquant-kv-cache \
    https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git \
    "${SYS_CRATE_DIR}/llama.cpp"

# 4. Remove .git from the cloned repo (we vendor it, not submodule it)
rm -rf "${SYS_CRATE_DIR}/llama.cpp/.git"

# 5. Compatibility patches
#    The spiritbuun fork may be based on an older llama.cpp.
#    Add missing struct fields so llama-cpp-sys-2 v0.1.140 compiles.

CHAT_H="${SYS_CRATE_DIR}/llama.cpp/common/chat.h"
echo "Checking fork compatibility with llama-cpp-sys-2 v0.1.140..."

if [ -f "$CHAT_H" ] && ! grep -q "thinking_forced_open" "$CHAT_H"; then
    echo "  Adding thinking_forced_open to structs in chat.h..."
    # Use awk with brace-depth tracking to insert before each struct's
    # closing '};', even when the struct contains nested { ... } initializers.
    awk '
    /struct common_chat_params \{/ || /struct common_chat_parser_params \{/ {
        in_s = 1; d = 0
    }
    in_s {
        for (i = 1; i <= length($0); i++) {
            c = substr($0, i, 1)
            if (c == "{") d++
            if (c == "}") d--
        }
        if (d == 0) {
            print "    bool thinking_forced_open = false; // TQ compat stub"
            in_s = 0
        }
    }
    { print }
    ' "$CHAT_H" > "${CHAT_H}.tmp" && mv "${CHAT_H}.tmp" "$CHAT_H"

    # Verify
    COUNT=$(grep -c "thinking_forced_open" "$CHAT_H" || true)
    echo "  Added thinking_forced_open to $COUNT locations"
    if [ "$COUNT" -lt 2 ]; then
        echo "ERROR: Expected at least 2 insertions (common_chat_params + common_chat_parser_params)"
        echo "--- chat.h around 'thinking_forced_open' ---"
        grep -n -B2 -A2 "thinking_forced_open" "$CHAT_H" || true
        exit 1
    fi
    echo "  -> OK"
else
    echo "  thinking_forced_open already present in fork headers"
fi

# 6. Verify TurboQuant types exist in the fork
if grep -q "GGML_TYPE_TURBO3_0" "${SYS_CRATE_DIR}/llama.cpp/ggml/include/ggml.h"; then
    echo "Verified: GGML_TYPE_TURBO3_0 found in fork"
else
    echo "ERROR: GGML_TYPE_TURBO3_0 not found in fork — wrong branch?"
    exit 1
fi

# 7. Activate [patch.crates-io] in Cargo.toml
#    Uncomment the patch section so cargo uses the vendored fork
#    instead of the standard crate from crates.io.
if grep -q '^# \[patch.crates-io\]' "$CARGO_TOML"; then
    echo "Activating [patch.crates-io] in Cargo.toml..."
    sed -i 's|^# \[patch.crates-io\]|[patch.crates-io]|' "$CARGO_TOML"
    sed -i 's|^# llama-cpp-sys-2 = |llama-cpp-sys-2 = |' "$CARGO_TOML"
elif grep -q '^\[patch.crates-io\]' "$CARGO_TOML"; then
    echo "[patch.crates-io] already active in Cargo.toml"
else
    echo "Adding [patch.crates-io] to Cargo.toml..."
    cat >> "$CARGO_TOML" <<PATCH

[patch.crates-io]
llama-cpp-sys-2 = { path = "${PATCH_PATH}" }
PATCH
fi

# 8. Verify the patch is active
if grep -q "llama-cpp-sys-2" "$CARGO_TOML" && \
   grep -q "^\[patch.crates-io\]" "$CARGO_TOML"; then
    echo "Verified: [patch.crates-io] is active"
else
    echo "WARNING: Could not verify [patch.crates-io] activation"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Vendored llama-cpp-sys-2: ${SYS_CRATE_DIR}"
echo "Patch in: ${CARGO_TOML}  (path = ${PATCH_PATH})"
echo "Fork verified: GGML_TYPE_TURBO3_0 = 41"
echo ""
echo "Build with:"
echo "  cd ${ENGINE_DIR}"
echo "  cargo build --release --features 'cuda turboquant_native'"
echo ""
echo "Verify with (from workspace root):"
echo "  cargo tree -i llama-cpp-sys-2  # should show (${PATCH_PATH})"
