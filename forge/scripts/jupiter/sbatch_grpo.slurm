#!/bin/bash
# GRPO on a merged open-book model, Leonardo's recipe (grpo_train.py, the
# r2l rounds, forge/scripts/leonardo/sbatch_grpo.slurm) on one GH200 node:
# the policy on GPUs 0-2, one process each (three x --per-device-batch 8 =
# 24 answers a step, three prompts of eight), and the judge that grades the
# practical questions on GPU 3, Qwen3-30B-A3B-Instruct-2507 Q8_0 served by
# EULLM Engine's /v1/chat/completions (on Leonardo, llama-server with the
# same GGUF). The verifiable rewards grade the rest (eullm_forge.rl.rewards).
#
#     eullm && GRPO_MODEL=<merged> GRPO_PROMPTS=<prompts.jsonl> GRPO_OUT=<dir> \
#         FORGE_EXTRA_LIBS=$PROJECT/$USER/forge/libs-qwen35 \
#         sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/forge/runs/logs/grpo-%j.out" \
#         forge/scripts/jupiter/sbatch_grpo.slurm
#
# submit_grpo_round.sh submits it as a chain of links, then the exams. Each
# link resumes from the last checkpoint and stops before its time is up; the
# link that finds the adapter written merges it into GRPO_OUT/merged, and
# one that finds the merged model exits at once.
#
# Needs TRL in $FORGE_HOME/libs-grpo (setup_grpo_libs.sh), Qwen3.5's Triton
# in libs-qwen35 (FORGE_EXTRA_LIBS), and the judge in the engine's store
# (`$EULLM_BIN pull hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q8_0`, as
# for sbatch_gen_instruct.slurm).
#
# Settable:
#   GRPO_MODEL     merged model to start from (required)
#   GRPO_PROMPTS   prompts (filter_grpo_prompts.py output; required)
#   GRPO_OUT       checkpoints, adapter, merged model (required)
#   GRPO_ARGS      more grpo_train.py options, e.g. "--max-steps 300 --lr 1e-5"
#   GRPO_NO_MERGE  1: leave the adapter unmerged (the smoke test)
#   GRPO_LIBS      where TRL is (default $FORGE_HOME/libs-grpo)

#SBATCH --job-name=forge-grpo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
export EULLM_MODELS_DIR="${EULLM_MODELS_DIR:-$PROJECT/$USER/eullm-models}"
EULLM_BIN="${EULLM_BIN:-$PROJECT/$USER/eullm-target/next10/release/eullm}"
LIBS="${GRPO_LIBS:-$FORGE_HOME/libs-grpo}"
MODEL="${GRPO_MODEL:-}" PROMPTS="${GRPO_PROMPTS:-}" OUT="${GRPO_OUT:-}"
JOB="${SLURM_JOB_ID:-local}"
T0=$(date +%s)

echo "=== Forge GRPO: job $JOB on $(hostname), $(date -Is) ==="
echo "  model   $MODEL"
echo "  prompts $PROMPTS ($(grep -c . "$PROMPTS" 2>/dev/null || echo 0))"
echo "  out     $OUT"
echo "  args    ${GRPO_ARGS:-}"
fail=0
[ -f "$MODEL/config.json" ] || { echo "[!!] no merged model at '$MODEL'"; fail=1; }
# Prompts a job wrote after the round was submitted (submit_grpo_round.sh
# with GRPO_AFTER): filtered here, once, by the command it left.
if [ ! -s "$PROMPTS" ] && [ -s "$(dirname "$PROMPTS")/filter.sh" ]; then
    echo "  filtering the prompts: $(dirname "$PROMPTS")/filter.sh"
    bash "$(dirname "$PROMPTS")/filter.sh"
fi
[ -s "$PROMPTS" ] || { echo "[!!] no prompts at '$PROMPTS'"; fail=1; }
[ -n "$OUT" ] || { echo "[!!] GRPO_OUT is required"; fail=1; }
[ "$fail" = 0 ] || exit 1
if [ -f "$OUT/merged/config.json" ]; then
    echo "[ok] $OUT/merged is written: nothing left to do"
    exit 0
fi
if [ -s "$OUT/STOPPED" ] && [ ! -f "$OUT/adapter/adapter_config.json" ]; then
    echo "[!!] the run guard stopped this run: $(cat "$OUT/STOPPED")"
    exit 1
fi
mkdir -p "$OUT" "$FORGE_RUN_DIR/power"
export PYTHONPATH="$LIBS:$EULLM_REPO/forge${PYTHONPATH:+:$PYTHONPATH}"

POWER="$FORGE_RUN_DIR/power/grpo-$JOB.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 >"$POWER" 2>/dev/null &
SAMPLER=$!
JUDGE_PID=""
finish() {
    [ -n "$JUDGE_PID" ] && { kill "$JUDGE_PID" 2>/dev/null; wait "$JUDGE_PID" 2>/dev/null; }
    kill "$SAMPLER" 2>/dev/null
    wait "$SAMPLER" 2>/dev/null
    python "$EULLM_REPO/forge/scripts/jupiter/energy.py" "$POWER" --interval 5
    echo "=== done $(date -Is) (exit $1) ==="
    exit "$1"
}

if [ ! -f "$OUT/adapter/adapter_config.json" ]; then
    python -c "import trl; print('  trl', trl.__version__, 'from', trl.__file__)" ||
        { echo "[!!] TRL does not import from $LIBS: setup_grpo_libs.sh on a login node"; finish 1; }
    JMODEL=$("$EULLM_BIN" list 2>/dev/null | awk 'tolower($1) ~ /30b-a3b-instruct-2507/ && !/file missing/ { print $1; exit }')
    [ -n "$JMODEL" ] || { echo "[!!] Qwen3-30B-A3B-Instruct-2507 is not in $EULLM_MODELS_DIR: pull it on a login node"; finish 1; }

    # The judge: 16 slots of 8,192 tokens (an article, the answer, the
    # rubric), on GPU 3 and its Grace. Loaded once and kept (keep_alive -1):
    # the reward's requests name no model, and the engine answers them with
    # the one loaded.
    PORT=$((18700 + ${SLURM_JOB_ID:-0} % 200))
    export EULLM_AUDIT_DIR="$OUT/judge-audit"
    CUDA_VISIBLE_DEVICES=3 taskset -c 216-287 "$EULLM_BIN" serve --port "$PORT" \
        --batch-size 16 --ctx-size 131072 --mmap >"$OUT/judge-server-$JOB.log" 2>&1 &
    JUDGE_PID=$!
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1 && break
        kill -0 "$JUDGE_PID" 2>/dev/null || break
        sleep 1
    done
    curl -s -m 900 "http://127.0.0.1:$PORT/api/chat" -H 'Content-Type: application/json' \
        -d "{\"model\": \"$JMODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Ciao.\"}], \"stream\": false, \"keep_alive\": -1, \"options\": {\"num_predict\": 4}}" \
        >"$OUT/judge-warmup-$JOB.json" 2>&1
    grep -q '"done"' "$OUT/judge-warmup-$JOB.json" ||
        { echo "[!!] the judge did not answer:"; tail -5 "$OUT/judge-server-$JOB.log"; finish 1; }
    echo "  judge   $JMODEL on GPU 3, port $PORT, loaded in $(( $(date +%s) - T0 )) s"

    # Stop in time to save and merge: the link after this one resumes from
    # the last checkpoint (every --save-steps).
    LIMIT_S=$(job_time_limit_s)
    TRAIN_S=$(( ${LIMIT_S:-21600} - ($(date +%s) - T0) - 1200 ))
    [ "$TRAIN_S" -gt 600 ] || { echo "[!!] no time left to train ($TRAIN_S s)"; finish 1; }
    echo "  policy  GPUs 0-2, at most $TRAIN_S s"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=0,1,2 timeout --signal=TERM "$TRAIN_S" \
        accelerate launch --num_processes 3 --num_machines 1 --mixed_precision bf16 \
        --dynamo_backend no "$EULLM_REPO/forge/scripts/grpo_train.py" \
        --model "$MODEL" --prompts "$PROMPTS" --out "$OUT" \
        --judge-url "http://127.0.0.1:$PORT" --save-steps 10 ${GRPO_ARGS:-}
    RC=$?
    kill "$JUDGE_PID" 2>/dev/null
    wait "$JUDGE_PID" 2>/dev/null
    JUDGE_PID=""
    if [ ! -f "$OUT/adapter/adapter_config.json" ]; then
        if [ "$RC" = 124 ]; then
            echo "[ok] time is up: the next link resumes from $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
            finish 0
        fi
        echo "[!!] grpo_train.py: exit $RC, no adapter"
        finish "$RC"
    fi
fi
[ "${GRPO_NO_MERGE:-0}" = 1 ] && { echo "[ok] adapter $OUT/adapter (not merged)"; finish 0; }
python - "$MODEL" "$OUT/adapter" "$OUT/merged" <<'PY' || { echo "[!!] merge failed"; finish 1; }
import sys
from eullm_forge.identity import merge_identity_adapter
merge_identity_adapter(sys.argv[1], sys.argv[2], sys.argv[3])
PY
echo "[ok] merged into $OUT/merged"
finish 0
