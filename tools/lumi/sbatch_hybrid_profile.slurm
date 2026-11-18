#!/bin/bash
# Where a hybrid model's time goes on one GCD, in llama.cpp alone (no EuLLM):
# Qwen3.6-27B (three layers in four gated delta net, a recurrent state per
# sequence) against Qwen3-32B (attention only), the same llama.cpp commit as
# the engine, built by install_runtimes.sh.
#
#     export SBATCH_ACCOUNT=project_465003366
#     SKIP_OLLAMA=1 SKIP_MODELS=1 bash tools/lumi/install_runtimes.sh   # once, on a login node
#     sbatch tools/lumi/sbatch_hybrid_profile.slurm [MODEL_ID ...]
#
# Default: qwen3.6-27b qwen3-32b. Output: eullm-hybrid-profile-<job>.out
# beside where it was submitted, the kernel traces in eullm-hybrid-profile-<job>/.
#
# Reading it, in order:
#  1. llama-batched-bench, 1 to 32 sequences of 128 prompt and 128 answer
#     tokens each: S_TG is the answers' tokens/s over all sequences. In c01
#     (r-next6) EuLLM gained 5.2x from 1 to 32 on the 27B and 9.9x on the
#     32B; the same ratio here puts the limit in llama.cpp, a better one in
#     EuLLM's scheduler.
#  2. llama-bench: reading a prompt of 512 and 4096 tokens (pp) and writing
#     128 tokens (tg), one sequence.
#  3. The GPU kernels, by total time, while 1 and 32 sequences answer
#     (rocprof): what grows with the sequences is what the 27B pays per
#     sequence that the 32B does not.
#  4. Two conversations of five turns over a 6,000-token document each
#     (multiturn_probe.py), on EuLLM (as it starts by default, and with
#     --ctx-checkpoints 8) and on llama-server: `prompt_ms` per turn. On a
#     hybrid model a server that cannot restore an earlier state reads the
#     whole conversation every turn; on the 32B the slot's cache is reused.
#     Needs EULLM_BIN and LLAMA_SERVER_BIN; left out without them. With
#     EULLM_BIN_BEFORE, an older engine has the same conversation first
#     (label eullm-before), for a before and after on one node.
#
# PARTS picks the parts (default "1 2 3 4"): PARTS=4 has the conversations
# alone, a few minutes.

#SBATCH --job-name=eullm-hybrid-profile
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=eullm-hybrid-profile-%j.out

set -uo pipefail
# shellcheck source=campaign_env.sh
source "${EULLM_REPO:-$PWD}/tools/lumi/campaign_env.sh"
RT="$CAMPAIGN_DIR/runtimes"
BATCHED="$RT/llama-batched-bench"
BENCH="$RT/llama-bench"
for bin in "$BATCHED" "$BENCH"; do
    [ -x "$bin" ] || { echo "[err] $bin is not there: SKIP_OLLAMA=1 SKIP_MODELS=1 bash tools/lumi/install_runtimes.sh"; exit 1; }
done

if [ $# -gt 0 ]; then MODELS=("$@"); else MODELS=(qwen3.6-27b qwen3-32b); fi
OUT="eullm-hybrid-profile-${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

# rocprofv3 from ROCm 6.2 on, rocprof before it.
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
PROF=""
for p in rocprofv3 rocprof; do
    if [ -x "$ROCM_PATH/bin/$p" ]; then PROF="$ROCM_PATH/bin/$p"; break; fi
    if command -v "$p" >/dev/null 2>&1; then PROF="$(command -v "$p")"; break; fi
done

gguf_of() {
    "$PY" -c "import sys; sys.path.insert(0, '$EULLM_REPO/bench/campaign')
from point import store_gguf; print(store_gguf('$1') or '')"
}

echo "=== hybrid profile: ${MODELS[*]} on $(hostname), $(date -Is) ==="
echo "  llama.cpp    $("$BENCH" --version 2>&1 | grep -m1 version)"
echo "  profiler     ${PROF:-none found under $ROCM_PATH/bin or on PATH}"
echo

# Every load reads the model in (-lm none), as the engine does from Lustre.
COMMON=(-ngl 99 -lm none)
PORT=$((18700 + ${SLURM_JOB_ID:-0} % 300))
PARTS=" ${PARTS:-1 2 3 4} "
part() { [[ "$PARTS" == *" $1 "* ]]; }
TURNS="$PY $EULLM_REPO/tools/lumi/multiturn_probe.py"

wait_ready() {  # PATH PID: up to five minutes for the server to answer, while it lives
    for _ in $(seq 1 300); do
        curl -sf "localhost:$PORT$1" >/dev/null 2>&1 && return 0
        kill -0 "$2" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

stop() {
    kill "$1" 2>/dev/null
    wait "$1" 2>/dev/null
    sleep 2
}

conversation() {  # MODEL GGUF LABEL: five turns on one server, EuLLM or llama-server
    local m=$1 gguf=$2 label=$3 api=eullm ready=/api/version server bin=$EULLM_BIN
    shift 3
    case "$label" in
        llama-server)
            api=llama ready=/health
            "$LLAMA_SERVER_BIN" --host 127.0.0.1 --port "$PORT" -m "$gguf" --alias "$m" \
                -c 16384 -np 1 -ngl 999 >"$OUT/$m.$label.log" 2>&1 &
            ;;
        *)
            [ "$label" = eullm-before ] && bin=$EULLM_BIN_BEFORE
            # The scheduler's debug lines say which states it kept and restored.
            RUST_LOG="eullm=info,eullm::inference::scheduler=debug" \
                "$bin" serve --port "$PORT" --batch-size 1 --ctx-size 16384 "$@" \
                >"$OUT/$m.$label.log" 2>&1 &
            ;;
    esac
    server=$!
    if wait_ready "$ready" "$server"; then
        $TURNS --url "http://127.0.0.1:$PORT" --api "$api" --model "$m" --label "$label" \
            --turns 5 --conversations 2
    else
        echo "[!!] $label did not start: tail $OUT/$m.$label.log"
    fi
    stop "$server"
}

for m in "${MODELS[@]}"; do
    gguf=$(gguf_of "$m")
    if [ -z "$gguf" ]; then echo "[!!] $m is not in $EULLM_MODELS_DIR"; continue; fi
    echo "=== $m: $gguf ==="

    if part 1; then
        echo "--- 1. batched-bench: 1 to 32 sequences ---"
        "$BATCHED" -m "$gguf" "${COMMON[@]}" -fa on -c 16384 -b 2048 -ub 512 \
            -npp 128 -ntg 128 -npl 1,4,16,32 2>"$OUT/$m.batched.log" | grep -E '^\|'
    fi

    if part 2; then
        echo "--- 2. llama-bench: prompts of 512 and 4096 tokens, 128 written ---"
        "$BENCH" -m "$gguf" "${COMMON[@]}" -fa 1 -ub 512 -p 512,4096 -n 128 -r 3 -o md \
            2>"$OUT/$m.bench.log"
    fi

    if part 3 && [ -n "$PROF" ]; then
        for n in 1 32; do
            echo "--- 3. kernels while $n sequences answer (32 prompt, 128 answer tokens each) ---"
            dir="$OUT/$m.npl$n"
            mkdir -p "$dir"
            cmd=("$BATCHED" -m "$gguf" "${COMMON[@]}" -fa on -c 8192 -b 2048 -ub 512
                 -npp 32 -ntg 128 -npl "$n")
            if [[ "$PROF" == *rocprofv3 ]]; then
                "$PROF" --kernel-trace --stats -d "$dir" -o run -- "${cmd[@]}" >"$dir/run.log" 2>&1
                stats=$(find "$dir" -name '*kernel_stats.csv' | head -1)
            else
                "$PROF" --stats -o "$dir/run.csv" "${cmd[@]}" >"$dir/run.log" 2>&1
                stats="$dir/run.stats.csv"
            fi
            if [ -n "$stats" ] && [ -s "$stats" ]; then
                "$PY" - "$stats" <<'EOF'
import csv, sys
with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
    rows = list(csv.DictReader(f))
total = sum(float(r["TotalDurationNs"]) for r in rows) or 1.0
rows.sort(key=lambda r: -float(r["TotalDurationNs"]))
print(f"  {'ms':>9} {'share':>6} {'calls':>7}  kernel  (all kernels: {total / 1e6:.1f} ms)")
for r in rows[:20]:
    name = r["Name"].split("(")[0].replace("void ", "")[:100]
    ns = float(r["TotalDurationNs"])
    print(f"  {ns / 1e6:9.1f} {100 * ns / total:5.1f}% {int(float(r['Calls'])):7d}  {name}")
EOF
            else
                echo "  [!!] no kernel statistics: tail $dir/run.log"
            fi
        done
    fi

    if ! part 4; then
        :
    elif [ -x "${EULLM_BIN:-}" ] && [ -x "${LLAMA_SERVER_BIN:-}" ]; then
        echo "--- 4. five turns over a 6,000-token document: prompt_ms per turn ---"
        if [ -x "${EULLM_BIN_BEFORE:-}" ]; then
            conversation "$m" "$gguf" eullm-before
        fi
        conversation "$m" "$gguf" eullm
        conversation "$m" "$gguf" llama-server
        # Which states the engine kept and restored, by the log.
        grep -hE "kept its recurrent|restored the recurrent|reused prefill failed" \
            "$OUT/$m.eullm.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-160 | tail -8
    else
        echo "--- 4. skipped: set EULLM_BIN (and LLAMA_SERVER_BIN, campaign_env.sh finds it) ---"
    fi
    echo
done
echo "=== done $(date -Is) ==="
