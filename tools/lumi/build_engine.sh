#!/usr/bin/env bash
# Build the EULLM engine with the ROCm/HIP backend on a LUMI-G login node.
#
# Usage, from a login node (no Slurm allocation needed — compiling is a normal
# login-node activity and needs no GPU present):
#
#     bash tools/lumi/build_engine.sh
#
# Takes a while; run it under tmux so a dropped SSH session doesn't kill it.
#
# Why build here at all, when the release publishes eullm-linux-x64-rocm-gfx90a:
# that artifact is built against ROCm 6.3.4 because that is what LUMI had in
# September 2026. When the site moves and the sonames no longer match, this
# script picks up whatever ROCm the machine actually has, which is the same
# reason building from source remains the fallback on Leonardo (see
# docs/cineca/leonardo.md).
#
# Overridable:
#   ROCM_PATH             where ROCm lives (default: /opt/rocm)
#   EULLM_AMDGPU_TARGETS  GPU architecture (default: gfx90a = MI250X)
#   EULLM_REPO            repository root (default: inferred from this script)

set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
EULLM_AMDGPU_TARGETS="${EULLM_AMDGPU_TARGETS:-gfx90a}"
EULLM_REPO="${EULLM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

# ── Preflight ─────────────────────────────────────────────────────────────
# Every check here failed for someone, somewhere, in a way that wasted an hour
# of build before saying so.

[ -d "$ROCM_PATH/lib" ] || err "no ROCm at $ROCM_PATH — check 'module avail rocm' and set ROCM_PATH"
log "ROCm: $("$ROCM_PATH/bin/hipconfig" --version 2>/dev/null || echo unknown) at $ROCM_PATH"

command -v cmake >/dev/null || err "cmake not found — 'module load CMake' or equivalent"
CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
# ggml-hip calls enable_language(HIP), which CMake gained in 3.21.
printf '3.21\n%s\n' "$CMAKE_VER" | sort -V -C || err "cmake $CMAKE_VER is too old — the HIP language needs 3.21+"
log "cmake: $CMAKE_VER"

command -v cargo >/dev/null || err "cargo not found — install rustup into \$HOME from a login node (they have outbound network; compute nodes do not)"
log "cargo: $(cargo --version)"

[ -f "$EULLM_REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/CMakeLists.txt" ] \
    || err "the llama.cpp submodule is missing — run: git -C '$EULLM_REPO' submodule update --init --recursive"

# ── Build ─────────────────────────────────────────────────────────────────
# EULLM_AMDGPU_TARGETS is not optional here even though the build script treats
# it as such: without it the HIP architecture is resolved from the GPUs in this
# machine, and a login node has none.
log "building for $EULLM_AMDGPU_TARGETS (this takes a while — tmux recommended)"
cd "$EULLM_REPO"
ROCM_PATH="$ROCM_PATH" \
HIPCXX="${HIPCXX:-$ROCM_PATH/llvm/bin/clang++}" \
EULLM_AMDGPU_TARGETS="$EULLM_AMDGPU_TARGETS" \
    cargo build --release --features rocm -p eullm-engine

BIN="$EULLM_REPO/target/release/eullm"
[ -x "$BIN" ] || err "the build reported success but $BIN is not there"

# ── Verify what the binary actually contains ──────────────────────────────
# Not what the build was asked to contain. A HIP binary built for the wrong
# architecture links, starts, announces a GPU backend and runs on CPU — the AMD
# spelling of the trap that cost real A100 time on Leonardo.
#
# Both checks match against a variable rather than piping into `grep -q`:
# `grep -q` exits at the first match, the producer upstream dies of SIGPIPE, and
# `set -o pipefail` reports that 141 as the pipeline's status — failing the test
# on a binary that does contain what it is looking for. The CI job was rejecting
# a perfectly good gfx90a binary exactly that way. A `<<<` here-string is not a
# pipeline, so grep may exit as early as it likes.
LINKED=$(ldd "$BIN")
grep -q libamdhip64 <<<"$LINKED" \
    || err "libamdhip64 is not in the link table — the HIP backend was not compiled in"

ARCHS=$(strings "$BIN" | grep -oE "amdhsa--gfx[0-9a-z]+" | sort -u)
if ! grep -q "amdhsa--${EULLM_AMDGPU_TARGETS}" <<<"$ARCHS"; then
    echo "architectures actually present: ${ARCHS:-(none)}" >&2
    err "no ${EULLM_AMDGPU_TARGETS} device code in the binary"
fi

ok "device code present for: $(tr '\n' ' ' <<<"$ARCHS")"
ok "binary: $BIN"
echo
echo "Next: pull a model from a login node (compute nodes have no outbound"
echo "network), then run the smoke job:"
echo
echo "    $BIN pull qwen3-8b"
echo "    sbatch tools/lumi/sbatch_smoke.slurm"
