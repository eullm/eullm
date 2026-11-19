#!/bin/bash
# Build llama-server with CUDA, once, so the exam can ask the GGUF that ships.
#
#   sbatch forge/scripts/leonardo/sbatch_build_llama_server.slurm
#
# The llama.cpp checkout in $LCPP_DIR is built for the CPU (build/): enough to
# convert and quantize, far too slow to answer an exam of 900 questions. This
# builds llama-server alone, for the A100 (sm_80), into $LCPP_DIR/build-cuda,
# from the same checkout and offline -- compute nodes have no network, and
# none is needed. It ends by listing the GPU the binary sees, so a build that
# compiles but cannot run fails here and not in the first exam.
#
# The modules it compiled with are written to build-cuda/modules.txt;
# sbatch_norm_exam.slurm loads them before starting the server.
#
# On a GPU node, not the serial partition: the check at the end needs the
# GPU, and the build is a few minutes of one.
#
# LCPP_RPC=1 builds, into build-cuda-rpc instead, llama-server and rpc-server
# with the RPC backend (-DGGML_RPC=ON): a model larger than one node served
# over several (sbatch_rpc_rewrites.slurm). A directory of its own, so the
# jobs running from build-cuda are not touched while it builds.
#
# Settable:
#   LCPP_RPC      1 = the multi-node build (default: the one-node build)
#   LCPP_MODULES  modules to load, in order (default "gcc cuda": each name
#                 picks the site's default version; `module avail cuda` lists
#                 the others)
#
#SBATCH --job-name=eullm-build-llama-server
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

EULLM_REPO="${EULLM_REPO:-$WORK/eullm}"
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"

SRC="${LCPP_DIR:-$WORK/llama.cpp}"
if [ "${LCPP_RPC:-0}" = 1 ]; then
    BUILD="$SRC/build-cuda-rpc"; EXTRA="-DGGML_RPC=ON"; TARGETS="llama-server rpc-server"
else
    BUILD="$SRC/build-cuda"; EXTRA=""; TARGETS="llama-server"
fi
[ -f "$SRC/CMakeLists.txt" ] || { echo "[build] no llama.cpp checkout at $SRC" >&2; exit 1; }
echo "[build] llama.cpp $(git -C "$SRC" log -1 --format='%h %cd' --date=short 2>/dev/null)"

MODULES="${LCPP_MODULES:-gcc cuda}"
for m in $MODULES; do
    module load "$m" || { echo "[build] cannot load module '$m' -- see 'module avail ${m%%/*}'" >&2; exit 1; }
done
command -v nvcc >/dev/null || { echo "[build] no nvcc after loading: $MODULES" >&2; exit 1; }
echo "[build] $(g++ --version | head -1); $(nvcc --version | tail -1)"

# A cache configured with another compiler keeps using it (see
# quantize_to_gguf.sh): start clean.
rm -rf "$BUILD"
# shellcheck disable=SC2086
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF $EXTRA >/dev/null ||
    { echo "[build] cmake configure failed" >&2; exit 1; }
# shellcheck disable=SC2086
cmake --build "$BUILD" --config Release --target $TARGETS -j "${SLURM_CPUS_PER_TASK:-8}" \
    >"$BUILD/build.log" 2>&1 || { tail -30 "$BUILD/build.log" >&2; echo "[build] failed" >&2; exit 1; }

# Fully resolved names (gcc/12.2.0, not gcc), so the exam loads what this used.
module -t list 2>&1 | grep -E "^(${MODULES// /|})(/|$)" > "$BUILD/modules.txt" ||
    echo "$MODULES" > "$BUILD/modules.txt"
echo "[build] modules: $(tr '\n' ' ' < "$BUILD/modules.txt")"

"$BUILD/bin/llama-server" --list-devices 2>&1 | tail -5
if "$BUILD/bin/llama-server" --list-devices 2>&1 | grep -q "CUDA0"; then
    echo "[build] ok: $BUILD/bin/llama-server sees the GPU"
else
    echo "[build] the binary does not see a CUDA device" >&2
    exit 1
fi
