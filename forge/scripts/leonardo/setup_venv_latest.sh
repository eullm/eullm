#!/usr/bin/env bash
# Builds the Forge venv on Leonardo from the latest stable releases
# (requirements-latest.txt), on a login node (compute nodes have no internet).
#
#   module load python/<newest>     # `module av python`: 3.12 or 3.13 if there, else 3.11.7
#   TORCH_CUDA=cu126 bash forge/scripts/leonardo/setup_venv_latest.sh
#
# TORCH_CUDA follows the GPU driver, which a login node cannot see: read it
# on a compute node first (nvidia-smi, "CUDA Version" on its first line).
#   driver r535-r575 (CUDA 12.x)  ->  cu126: torch 2.14.1, the newest with a CUDA 12 build
#   driver r580 or newer (CUDA 13) ->  cu132: torch 2.15.0, the newest of all
# (download.pytorch.org on 03-11-2026: torch 2.15.0 exists only for cu132.)
# The Playground's venv ran torch 2.11.0+cu126 on driver r535 (CUDA 12.2):
# a CUDA 12 build runs on any r525+ driver, a CUDA 13 build refuses to start.
#
# The venv goes to $EULLM_VENV_NEW (default $WORK/eullm_venv-latest), never
# over an existing one. It becomes $EULLM_VENV only after
# sbatch_venv_smoke.slurm has passed on a GPU; its pip freeze is then the new
# known-good list. Idempotent: run again, it only adds what is missing.
set -euo pipefail

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

[ -z "${SLURM_JOB_ID:-}" ] || err "run this on a login node: compute nodes have no internet"
: "${WORK:?WORK is not set}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
VENV="${EULLM_VENV_NEW:-$WORK/eullm_venv-latest}"
TORCH_CUDA="${TORCH_CUDA:?set TORCH_CUDA=cu126 (driver CUDA 12.x) or cu132 (driver r580+): see the header}"

case "$TORCH_CUDA" in
    cu126) TORCH_V=2.14.1 ;;
    cu132) TORCH_V=2.15.0 ;;
    *) err "TORCH_CUDA=$TORCH_CUDA: only cu126 and cu132 are planned (see the header)" ;;
esac

PY="${PYTHON:-$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3)}"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' ||
    err "need Python >= 3.11 (torch $TORCH_V, transformers 5.22): module load python/3.11.7"

if [ ! -f "$VENV/bin/activate" ]; then
    log "creating $VENV with $($PY --version)"
    "$PY" -m venv "$VENV"
else
    ok "venv already at $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
[ "$(command -v python)" = "$VENV/bin/python" ] || err "the venv is not active"

log "torch==$TORCH_V+$TORCH_CUDA from download.pytorch.org"
python -m pip install --quiet "torch==$TORCH_V+$TORCH_CUDA" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
    --extra-index-url https://pypi.org/simple

# Hold torch and the triton it brought: nothing installed after may move them.
CONSTRAINTS="$VENV/constraints-torch.txt"
python - "$CONSTRAINTS" <<'PY'
import importlib.metadata as md, sys
lines = [f"torch=={md.version('torch')}"]
try:
    lines.append(f"triton=={md.version('triton')}")
except md.PackageNotFoundError:
    pass
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print("  held:", ", ".join(lines))
PY

log "requirements-latest.txt (torch held)"
python -m pip install --quiet -c "$CONSTRAINTS" -r "$HERE/requirements-latest.txt"

log "eullm_forge from $REPO/forge (editable, no dependencies)"
python -m pip install --quiet --no-deps -e "$REPO/forge"

log "import check (CPU only here: the GPU check is sbatch_venv_smoke.slurm)"
python - "$VENV" <<'PY'
import importlib, importlib.metadata as md, sys
venv = sys.argv[1]
bad = []
for name, dist in (("torch", "torch"), ("numpy", "numpy"), ("transformers", "transformers"),
                   ("peft", "peft"), ("accelerate", "accelerate"), ("datasets", "datasets"),
                   ("safetensors", "safetensors"), ("huggingface_hub", "huggingface-hub"),
                   ("tokenizers", "tokenizers"), ("bitsandbytes", "bitsandbytes"),
                   ("yaml", "pyyaml"), ("click", "click"), ("rich", "rich"),
                   ("requests", "requests"), ("bs4", "beautifulsoup4"),
                   ("sentencepiece", "sentencepiece"), ("einops", "einops"),
                   ("fla", "fla-core"), ("eullm_forge", "eullm-forge")):
    try:
        importlib.import_module(name)
    except Exception as exc:
        # bitsandbytes and fla look for a GPU at import; a login node has none.
        if name in {"bitsandbytes", "fla"}:
            print(f"  {name:<16}{md.version(dist):<14}installed; imports on a GPU node only ({type(exc).__name__})")
            continue
        bad.append(name)
        print(f"  {name:<16}FAILED ({type(exc).__name__}: {exc})")
        continue
    print(f"  {name:<16}{md.version(dist)}")
import torch
print(f"  torch built for CUDA {torch.version.cuda}")
if bad:
    sys.exit(f"[err] not importable: {' '.join(bad)}")
PY

FREEZE="$WORK/eullm_venv-latest-candidate-$(date +%Y-%m-%d).txt"
python -m pip freeze > "$FREEZE"
ok "venv ready at $VENV; its freeze: $FREEZE"
echo
echo "Next, on a GPU (about 15 minutes of one A100):"
echo "  EULLM_VENV=$VENV sbatch -A \$EULLM_ACCOUNT $HERE/sbatch_venv_smoke.slurm"
echo "When its log ends with SMOKE PASSED: make it the venv of every job,"
echo "  export EULLM_VENV=$VENV          # in ~/.bashrc"
echo "and keep the freeze as the known-good list:"
echo "  cp $FREEZE $WORK/eullm_venv-known-good-$(date +%Y-%m-%d).txt"
