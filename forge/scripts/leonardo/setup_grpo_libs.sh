#!/usr/bin/env bash
# Install TRL for the GRPO jobs WITHOUT touching the shared venv.
#
#   bash forge/scripts/leonardo/setup_grpo_libs.sh      # on a login node: needs internet
#
# The shared venv ($EULLM_VENV) is what every training, exam and package job
# imports. On 2026-09-30 one `pip install --upgrade` inside it replaced torch
# and every job died at `import torch` until the next morning. So TRL goes to
# a directory of its own ($GRPO_LIBS), installed with --no-deps so pip cannot
# pull a new torch, transformers or accelerate along with it, and only
# sbatch_grpo.slurm puts that directory on PYTHONPATH.
#
# TRL needs datasets >= 4.7; if the venv's is older, a newer one goes into the
# same directory, again with --no-deps. The last step imports GRPOTrainer
# against the venv's torch and transformers: if that fails, nothing else
# changed.

set -euo pipefail

TRL_VERSION="${TRL_VERSION:-1.17.1}"
DATASETS_VERSION="${DATASETS_VERSION:-5.0.1}"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LIBS="${GRPO_LIBS:-$WORK/pylibs/grpo}"
mkdir -p "$LIBS"

echo "[grpo-libs] installing trl==$TRL_VERSION into $LIBS (no dependencies)"
python -m pip install --quiet --no-deps --upgrade --target "$LIBS" "trl==$TRL_VERSION"

have="$(python -c 'import datasets; print(datasets.__version__)' 2>/dev/null || echo 0)"
if ! python - "$have" <<'EOF'
import sys
from packaging.version import Version
sys.exit(0 if Version(sys.argv[1]) >= Version("4.7.0") else 1)
EOF
then
    echo "[grpo-libs] venv datasets $have < 4.7: adding datasets==$DATASETS_VERSION to $LIBS"
    python -m pip install --quiet --no-deps --upgrade --target "$LIBS" "datasets==$DATASETS_VERSION"
fi

PYTHONPATH="$LIBS:${PYTHONPATH:-}" python - <<'EOF'
import datasets, torch, transformers, trl
from trl import GRPOConfig, GRPOTrainer  # noqa: F401
print(f"[grpo-libs] ok: trl {trl.__version__}, datasets {datasets.__version__}, "
      f"torch {torch.__version__}, transformers {transformers.__version__}")
EOF
