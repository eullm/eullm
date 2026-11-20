#!/bin/bash
# EULLM Engine with CUDA for the A100, compiled offline on a GPU node.
# Submitted by build_engine.sh, which fetches the crates first on the login
# node (this node has no network):
#
#   bash forge/scripts/leonardo/build_engine.sh [LABEL]
#
# The recipe is docs/cineca/leonardo.md's ("Build recipe that works"), every
# step of it there for a blocker met on 2026-10-14: Spack's gcc/12.2.0, not
# RHEL's 8.5 (std::filesystem); libclang from the llvm module with gcc 12's
# own headers (bindgen); the CUDA stubs for -lcuda; -no-pie on the eullm
# binary alone (nvcc's host code is not position-independent, proc-macros
# must stay so). Two things added: device code for sm_80 named rather than
# guessed, and gcc 12's libstdc++ found through the binary's rpath, so a job
# runs it without loading the module first.
#
# On a GPU node, not the serial partition: eight cores and the memory nvcc
# wants (the serial partition caps both), the binary started where it will
# run, and the build is under an hour of one GPU.
#
# Settable:
#   EB_LABEL     the result's name, $WORK/eullm-engine/EB_LABEL/eullm
#   EB_MODULES   modules to load (default "gcc/12.2.0 cuda/12.2")
#   EB_LLVM      the llvm module for libclang (default llvm/14.0.6--gcc--12.2.0-cuda-12.2)
#   EULLM_REPO   checkout to build (default $WORK/eullm)
#
#SBATCH --job-name=eullm-build-engine
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

EULLM_REPO="${EULLM_REPO:-$WORK/eullm}"
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
export RUSTUP_HOME="${RUSTUP_HOME:-$WORK/rustup}"
export CARGO_HOME="${CARGO_HOME:-$WORK/cargo}"
export PATH="$CARGO_HOME/bin:$PATH"

LABEL="${EB_LABEL:-$(git -C "$EULLM_REPO" rev-parse --short HEAD)}"
OUT="$WORK/eullm-engine/$LABEL"
export CARGO_TARGET_DIR="$WORK/eullm-target/$LABEL"
die() { echo "[engine] $*" >&2; exit 1; }

command -v cargo >/dev/null || die "no cargo in $CARGO_HOME: run build_engine.sh on the login node"

for m in ${EB_MODULES:-gcc/12.2.0 cuda/12.2}; do
    module load "$m" || die "cannot load module '$m' -- see 'module avail ${m%%/*}'"
done
command -v nvcc >/dev/null || die "no nvcc after loading the modules"
CUDA_ROOT="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export CUDA_PATH="$CUDA_ROOT" CUDACXX="$CUDA_ROOT/bin/nvcc"
export CC=gcc CXX=g++
GCC_PREFIX="$(dirname "$(dirname "$(command -v gcc)")")"
GCC_VER="$(gcc -dumpfullversion)"
GCC_LIB="$GCC_PREFIX/lib64"
[ -f "$GCC_LIB/libstdc++.so.6" ] || die "no libstdc++ in $GCC_LIB"
echo "[engine] gcc $GCC_VER, $(nvcc --version | tail -1)"

# libclang for bindgen: the llvm module only puts it on LD_LIBRARY_PATH
if [ -z "${LIBCLANG_PATH:-}" ]; then
    module load "${EB_LLVM:-llvm/14.0.6--gcc--12.2.0-cuda-12.2}" 2>/dev/null || module load llvm 2>/dev/null || true
    IFS=: read -r -a dirs <<<"${LD_LIBRARY_PATH:-}"
    for d in "${dirs[@]}"; do
        if compgen -G "$d/libclang.so*" >/dev/null; then LIBCLANG_PATH="$d"; break; fi
    done
fi
[ -n "${LIBCLANG_PATH:-}" ] || die "no libclang: 'module avail llvm' and set EB_LLVM"
export LIBCLANG_PATH
export BINDGEN_EXTRA_CLANG_ARGS="-I$GCC_PREFIX/lib/gcc/x86_64-pc-linux-gnu/$GCC_VER/include"
export RUSTFLAGS="-L $CUDA_ROOT/targets/x86_64-linux/lib -L $CUDA_ROOT/targets/x86_64-linux/lib/stubs"
command -v cmake >/dev/null || module load cmake 2>/dev/null || die "no cmake"
echo "[engine] libclang $LIBCLANG_PATH, $(cmake --version | head -1)"
echo "[engine] eullm $(git -C "$EULLM_REPO" log -1 --format='%h %cd' --date=short), into $CARGO_TARGET_DIR"

cd "$EULLM_REPO" || die "no checkout at $EULLM_REPO"
mkdir -p "$CARGO_TARGET_DIR"
SECONDS=0
CMAKE_CUDA_ARCHITECTURES=80 CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-8}" \
    cargo rustc --offline --locked --release --features "cuda,multimodal" \
    -j "${SLURM_CPUS_PER_TASK:-8}" -p eullm-engine --bin eullm \
    -- -C link-arg=-no-pie -C "link-arg=-Wl,-rpath,$GCC_LIB" \
    >"$CARGO_TARGET_DIR/build.log" 2>&1 ||
    { tail -40 "$CARGO_TARGET_DIR/build.log" >&2; die "build failed after $SECONDS s (log $CARGO_TARGET_DIR/build.log)"; }
echo "[engine] built in $((SECONDS / 60)) min"

BIN="$CARGO_TARGET_DIR/release/eullm"
[ -x "$BIN" ] || die "the build reported success but $BIN is not there"

# What the binary contains. Matched against variables, not `| grep -q`: under
# pipefail grep's early exit fails a good binary with SIGPIPE.
LINKED=$(ldd "$BIN" 2>&1 || true)
grep -q 'libcuda\.so' <<<"$LINKED" || die "$BIN does not link libcuda: not a CUDA build"
grep -q 'not found' <<<"$LINKED" && { grep 'not found' <<<"$LINKED" >&2; die "$BIN misses libraries"; }
ELF=$(cuobjdump --list-elf "$BIN" 2>/dev/null || true)
grep -q 'sm_80' <<<"$ELF" || die "$BIN has no device code for sm_80"
echo "[engine] ok: links the CUDA driver, device code for sm_80"

# The rpath, not the module: what a job that loads nothing finds
CLEAN=$(env -i PATH=/usr/bin:/bin ldd "$BIN" 2>&1 || true)
grep -q 'not found' <<<"$CLEAN" && { grep 'not found' <<<"$CLEAN" >&2; die "$BIN needs a module loaded to start"; }
grep -q "$GCC_LIB/libstdc++" <<<"$CLEAN" || die "$BIN takes the system libstdc++, not gcc $GCC_VER's"

mkdir -p "$OUT"
cp "$BIN" "$OUT/eullm.new" && mv "$OUT/eullm.new" "$OUT/eullm"
git -C "$EULLM_REPO" log -1 --format='%H %cd' --date=iso > "$OUT/commit.txt"
"$OUT/eullm" --version
"$OUT/eullm" serve --help 2>&1 | grep -q -- '--no-mmproj' && echo "[engine] serve has --no-mmproj"

# On the GPU: the device it sees, from its own startup log
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
env -i PATH=/usr/bin:/bin HOME="$HOME" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$OUT/eullm" serve --port "$PORT" --no-ui >"$OUT/startup.log" 2>&1 &
pid=$!
for _ in $(seq 60); do
    curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null && break
    sleep 2
done
kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
# llama.cpp names the device when its backend starts, which may be only at
# the first model load: no line here is not a failure, the first job that
# loads a model says it (its server log)
if grep -qiE 'A100|compute capability 8\.0' "$OUT/startup.log"; then
    grep -iE 'A100|compute capability' "$OUT/startup.log" | head -3
else
    echo "[engine] no device line before a model is loaded: see the first job's server log"
fi
echo "[engine] ok: $OUT/eullm"
