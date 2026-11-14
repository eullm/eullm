#!/bin/bash
# Two engine builds on the same GCD, one after the other: bench/speed_check.py
# three times on each (a short story written, a document of PROMPT_TOKENS
# read), so that a difference the smoke job shows can be told from the node
# it ran on. The smoke measures one short request once, model load and first
# graphs included in the minutes around it; on 14-11-2026 it gave 41.2 tok/s
# for llama.cpp 1.0 against 87.8 for 0.8.9 on another node.
#
#     AB_BINS="old=/scratch/$SBATCH_ACCOUNT/$USER/eullm-target/b12283/release/eullm \
#              new=/scratch/$SBATCH_ACCOUNT/$USER/eullm-target/llama-v1/release/eullm" \
#     EULLM_MODELS_DIR=/scratch/$SBATCH_ACCOUNT/$USER/eullm-models \
#         sbatch tools/lumi/sbatch_engine_ab.slurm
#
# MODEL (default qwen3-8b) must be in the store already: compute nodes have no
# network. One GCD on small-g, as the smoke job (see its header for why not
# dev-g); about ten minutes.

#SBATCH --job-name=eullm-engine-ab
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=eullm-engine-ab-%j.out

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
read -r -a BINS <<<"${AB_BINS:?AB_BINS: \"old=<eullm> new=<eullm>\"}"
MODEL="${MODEL:-qwen3-8b}"
RUNS="${AB_RUNS:-3}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8000}"
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-$HOME/.eullm/models}"
PORT=$((12400 + ${SLURM_JOB_ID:-0} % 500))
OUT="${SLURM_SUBMIT_DIR:-$PWD}/eullm-engine-ab-${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

echo "=== $(hostname), ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}, $(date -Is) ==="
rocm-smi --showproductname 2>&1 | grep -m1 "Card Series" | sed 's/^/  /'
pid=
trap '[[ -n $pid ]] && kill "$pid" 2>/dev/null' EXIT

for entry in "${BINS[@]}"; do
    label=${entry%%=*}
    bin=${entry#*=}
    echo
    echo "=== $label: $("$bin" --version 2>&1 | head -1) ==="
    "$bin" serve --port "$PORT" --default-model "$MODEL" --ctx-size 16384 --no-ui \
        >"$OUT/serve-$label.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
    done
    if ! curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null; then
        echo "  the server did not come up; the end of its log:"
        tail -n 20 "$OUT/serve-$label.log" | sed 's/^/  /'
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; pid=
        continue
    fi
    for run in $(seq 1 "$RUNS"); do
        python3 "$EULLM_REPO/bench/speed_check.py" --url "http://127.0.0.1:$PORT/v1" \
            --model "$MODEL" --prompt-tokens "$PROMPT_TOKENS" 2>&1 |
            grep -E "writes answers|reads a prompt|answer text" | sed "s/^/  run $run: /"
    done
    grep -iE "graph|error|warn" "$OUT/serve-$label.log" | head -5 | sed 's/^/  log: /'
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    pid=
done
echo
echo "=== done $(date -Is); servers' logs in $OUT ==="
