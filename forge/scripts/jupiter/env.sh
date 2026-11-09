# Forge's environment on JUPITER Booster (JSC): source it, don't run it, from
# a login shell or a job, as forge/scripts/leonardo/env.sh is on Leonardo.
#
#   source forge/scripts/jupiter/env.sh
#
# What the GPU probe found (job 2326998, 14-10-2026, probe_env.sh): `module
# load PyTorch` alone gives Stages/2026, Python 3.13.5 and torch 2.9.1+cu128,
# built for sm_80/90/100/120 and seeing the node's four GH200 (95 GiB each);
# transformers, peft, accelerate, datasets and safetensors are not there.
# So the site's PyTorch, and a venv on top of it with --system-site-packages
# for the rest (setup_venv.sh), never a second torch from pip.
#
# The module's packages reach Python through PYTHONPATH, which comes before a
# venv's own site-packages: an older package of the module would win over the
# version the venv installed. The venv's site-packages is put first on
# PYTHONPATH here, as JSC's venv template does.
#
# Everything of Forge's goes under $PROJECT/$USER/forge: the venv, the Hugging
# Face cache (compute nodes have no internet: models are fetched on a login
# node), the datasets and the runs. Not $SCRATCH: files idle for 90 days are
# deleted there.

JUPITER_PROJECT="${JUPITER_PROJECT:-e-aif-2026pg01-1434}"
if [ -z "${PROJECT:-}" ] || [[ "$PROJECT" != *"$JUPITER_PROJECT"* ]]; then
    jutil env activate -p "$JUPITER_PROJECT" >/dev/null 2>&1 || true
fi
: "${PROJECT:?no \$PROJECT: run 'jutil env activate -p $JUPITER_PROJECT' first}"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-$JUPITER_PROJECT}"

if [ -z "${EULLM_REPO:-}" ]; then
    EULLM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
export EULLM_REPO
export FORGE_HOME="${FORGE_HOME:-$PROJECT/$USER/forge}"
export FORGE_VENV="${FORGE_VENV:-$FORGE_HOME/venv}"
export FORGE_DATA_DIR="${FORGE_DATA_DIR:-$FORGE_HOME/datasets}"
export FORGE_RUN_DIR="${FORGE_RUN_DIR:-$FORGE_HOME/runs}"
export HF_HOME="${HF_HOME:-$FORGE_HOME/hf_home}"

# In a job the Hugging Face stack is offline: a cache miss fails at once
# instead of waiting on a network the node does not have.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# The site's PyTorch, quietly and without -u: Lmod's output does not survive
# `set -u` (tools/jupiter/env.sh).
if ! command -v module >/dev/null 2>&1 && [ -n "${LMOD_CMD:-}" ]; then
    module() { eval "$("$LMOD_CMD" bash "$@")"; }
fi
if command -v module >/dev/null 2>&1; then
    _forge_nounset=""
    [[ $- == *u* ]] && _forge_nounset=1
    set +u
    # On failure the default stage is loaded first and the load tried again
    # (the stages were reset by the software update of 02-11-2026, after
    # which every job of the night printed this failure), and Lmod's own
    # words are shown, not swallowed.
    # In this shell, not a $(...) subshell: a load there would not last.
    _forge_err=$(mktemp 2>/dev/null || echo "/tmp/forge-module-$$")
    if ! module load "${FORGE_MODULES:-PyTorch}" >/dev/null 2>"$_forge_err"; then
        module load Stages/2026 >/dev/null 2>&1
        module load "${FORGE_MODULES:-PyTorch}" >/dev/null 2>&1 ||
            { echo "[!!] module load ${FORGE_MODULES:-PyTorch} failed: see 'module spider PyTorch'" >&2
              grep -v '^\s*$' "$_forge_err" | head -4 | sed 's/^/     /' >&2; }
    fi
    rm -f "$_forge_err"
    unset _forge_err
    [ -n "$_forge_nounset" ] && set -u
    unset _forge_nounset
fi

if [ -f "$FORGE_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$FORGE_VENV/bin/activate"
    # First, always: removed wherever it is and put back in front. Checking
    # only that it was there was not enough: job 2327297 (14-10-2026),
    # submitted from a shell with the venv active, imported the module's
    # regex 2025.7.31 instead of the venv's (transformers 5.19 wants
    # 2025.10.22 or later), where the same venv imported fine on the login
    # node. Whatever reordered the path in the job, the venv now leads.
    for _forge_sp in "$FORGE_VENV"/lib/python*/site-packages; do
        _forge_pp=":${PYTHONPATH:-}:"
        _forge_pp="${_forge_pp//:$_forge_sp:/:}"
        _forge_pp="${_forge_pp#:}"
        _forge_pp="${_forge_pp%:}"
        export PYTHONPATH="$_forge_sp${_forge_pp:+:$_forge_pp}"
    done
    unset _forge_sp _forge_pp
fi

# FORGE_EXTRA_LIBS: a --target directory that goes before even the venv, for
# the jobs that need a library the venv must not carry. Qwen3.5 runs set it
# to $FORGE_HOME/libs-qwen35 (setup_qwen35_libs.sh): a newer Triton than the
# module's, which the Qwen3 jobs keep.
if [ -n "${FORGE_EXTRA_LIBS:-}" ] && [ -d "$FORGE_EXTRA_LIBS" ]; then
    _forge_pp=":${PYTHONPATH:-}:"
    _forge_pp="${_forge_pp//:$FORGE_EXTRA_LIBS:/:}"
    _forge_pp="${_forge_pp#:}"
    _forge_pp="${_forge_pp%:}"
    export PYTHONPATH="$FORGE_EXTRA_LIBS${_forge_pp:+:$_forge_pp}"
    unset _forge_pp
fi

mkdir -p "$FORGE_HOME" "$FORGE_DATA_DIR" "$FORGE_RUN_DIR/logs" "$HF_HOME"

# This job's time limit in seconds (empty outside a job, or when Slurm does
# not answer): the jobs size their training from it, so a link can be asked
# for with any --time and still stop in time to save.
job_time_limit_s() {
    local l d=0 h m s
    l=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %l 2>/dev/null) || return 0
    [ -n "$l" ] || return 0
    case "$l" in *-*) d=${l%%-*}; l=${l#*-} ;; esac
    IFS=: read -r h m s <<<"$l"
    [ -z "$s" ] && { s=$m; m=$h; h=0; }
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#${s:-0} ))
}
