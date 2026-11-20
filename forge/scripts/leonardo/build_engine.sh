#!/bin/bash
# EULLM Engine from this checkout, with CUDA for the A100 (sm_80): the login
# half, then the build as a job.
#
#   bash forge/scripts/leonardo/build_engine.sh [LABEL]
#
# On the login node, which reaches the internet: Rust from rustup (once, into
# $WORK), and `cargo fetch`, every crate the engine needs. Then it submits
# sbatch_build_engine.slurm, which compiles offline on a GPU node (compute
# nodes have no network) and checks that the binary sees the GPU. It prints
# the job's id, so the jobs that use the engine can wait on it:
#
#   b=$(bash forge/scripts/leonardo/build_engine.sh | tail -1)
#   sbatch -d afterok:$b ...
#
# LABEL names the result, $WORK/eullm-engine/LABEL/eullm (default the
# checkout's short commit): a running job keeps the binary it started with,
# so a new engine gets a directory of its own.
#
# Why build here and not take a release: the published Linux CUDA binaries
# carry the consumer cards' device code, and a binary without code for the
# GPU it runs on starts, announces CUDA and runs on the CPU
# (docs/cineca/leonardo.md); a release also lags main, and the point of
# building is to measure main.

set -euo pipefail

EULLM_REPO="${EULLM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export EULLM_REPO
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"

export RUSTUP_HOME="${RUSTUP_HOME:-$WORK/rustup}"
export CARGO_HOME="${CARGO_HOME:-$WORK/cargo}"
export PATH="$CARGO_HOME/bin:$PATH"

LABEL="${1:-$(git -C "$EULLM_REPO" rev-parse --short HEAD)}"

LCPP="engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
[ -f "$EULLM_REPO/$LCPP/CMakeLists.txt" ] ||
    { echo "[engine] the llama.cpp submodule is missing: git -C $EULLM_REPO submodule update --init --recursive" >&2; exit 1; }
case "$(git -C "$EULLM_REPO" submodule status "$LCPP")" in
    +*|-*) echo "[engine] llama.cpp is not at the commit this checkout wants: git -C $EULLM_REPO submodule update --init --recursive" >&2
           exit 1 ;;
esac

if ! command -v cargo >/dev/null; then
    echo "[engine] installing Rust into $CARGO_HOME (once)"
    mkdir -p "$RUSTUP_HOME" "$CARGO_HOME"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
        sh -s -- -y --no-modify-path --profile minimal
fi
# edition 2024 needs a recent toolchain: stable, brought up to date
rustup update stable --no-self-update >/dev/null
echo "[engine] $(cargo --version), llama.cpp $(git -C "$EULLM_REPO/$LCPP" log -1 --format='%h %cd' --date=short)"

cd "$EULLM_REPO"
cargo fetch --locked
echo "[engine] crates fetched into $CARGO_HOME"

# the job's environment checks, here and now: modules, compiler, CUDA,
# libclang, cmake. A failure is said before anything is queued.
( EB_PREFLIGHT=1 EB_LABEL="$LABEL" bash "$EULLM_REPO/forge/scripts/leonardo/sbatch_build_engine.slurm" ) >&2 ||
    { echo "[engine] preflight failed: nothing submitted" >&2; exit 1; }

mkdir -p "$EULLM_REPO/logs"
sbatch --parsable -J "eullm-build-engine-$LABEL" \
    --export=ALL,EB_LABEL="$LABEL",RUSTUP_HOME="$RUSTUP_HOME",CARGO_HOME="$CARGO_HOME" \
    "$EULLM_REPO/forge/scripts/leonardo/sbatch_build_engine.slurm"
