#!/usr/bin/env bash
# Build the engine against ROCm 7.2.4, as LUMI-G will have it after the
# January 2027 maintenance, inside CSC's ccpe/26.03 container
# (cpe-26.03.01-SP7-LUMI-26.01-ROCm-7.2.4.sif): ROCm 7.2.4 at /opt/rocm,
# GCC 14.3, CMake 3.28 and git in the container; cargo and rustup from the
# host, whose directories the container binds. build_engine.sh does the rest,
# as on the host.
#
#   EULLM_REPO=$S/eullm-v01010 CARGO_TARGET_DIR=$S/eullm-target/v01010-rocm72 \
#       bash tools/lumi/build_engine_ccpe.sh
#
# The container's module is installed once, in the project's EasyBuild tree
# (28-11-2026; the second path of the sources is read-only, the first is
# where EasyBuild writes):
#
#   export EBU_USER_PREFIX=/projappl/project_465003366/EasyBuild
#   module load LUMI/25.09 partition/container EasyBuild-user
#   EASYBUILD_SOURCEPATH=$EBU_USER_PREFIX/sources:/appl/local/containers/easybuild-sif-images \
#       eb ccpe-26.03-rocm-7.2.4-SP7-LUMI.eb
#
# The binary links the container's ROCm 7.2.4: it runs inside the container
# (ccpe-exec), not on the host's ROCm 6.3.4, and before the maintenance on a
# driver that supports ROCm user space up to 7.0, which is what is tested.
#
#   EBU_USER_PREFIX   the EasyBuild tree with the module (default: the project's)
#   CCPE_MODULE       the container's module (default: ccpe/26.03-rocm-7.2.4-SP7-LUMI)

# No `set -u`: Lmod's own functions read unset variables.
set -eo pipefail

export EBU_USER_PREFIX="${EBU_USER_PREFIX:-/projappl/${SBATCH_ACCOUNT:-project_465003366}/EasyBuild}"
CCPE_MODULE="${CCPE_MODULE:-ccpe/26.03-rocm-7.2.4-SP7-LUMI}"
[ -n "${EULLM_REPO:-}" ] || { echo "[err] EULLM_REPO: the checkout or worktree to build" >&2; exit 1; }
[ -n "${CARGO_TARGET_DIR:-}" ] || { echo "[err] CARGO_TARGET_DIR: where the build goes, on /scratch" >&2; exit 1; }
CARGO_BIN="$(command -v cargo)" \
    || { echo "[err] no cargo on PATH: install rustup on a login node" >&2; exit 1; }

# What the container needs of the host's Rust, passed by name: ccpe-exec
# clears the modules, not these.
export EULLM_REPO CARGO_TARGET_DIR
export EULLM_CCPE_CARGO_DIR="${CARGO_BIN%/*}"
export EULLM_CCPE_RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export EULLM_CCPE_CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
echo "[ccpe] cargo from $EULLM_CCPE_CARGO_DIR (RUSTUP_HOME $EULLM_CCPE_RUSTUP_HOME)"

module load CrayEnv "$CCPE_MODULE"
command -v ccpe-exec >/dev/null \
    || { echo "[err] $CCPE_MODULE gives no ccpe-exec: is it installed under $EBU_USER_PREFIX?" >&2; exit 1; }

# shellcheck disable=SC2016  # expanded inside the container
ccpe-exec bash -c '
    eval "$INITCCPE" >/dev/null 2>&1 || true
    module load PrgEnv-gnu >/dev/null 2>&1
    export PATH="$EULLM_CCPE_CARGO_DIR:$PATH" RUSTUP_HOME="$EULLM_CCPE_RUSTUP_HOME" \
        CARGO_HOME="$EULLM_CCPE_CARGO_HOME"
    echo "[ccpe] inside: ROCm $(cat /opt/rocm/.info/version 2>/dev/null), $(gcc --version | head -1), $(cargo --version)"
    exec bash "$EULLM_REPO/tools/lumi/build_engine.sh"
'
