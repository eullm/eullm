#!/usr/bin/env bash
# EuLLM's writing for docs/strata-study.md §7:
# - how fast a server that has read no long prompt yet writes;
# - how much of that time the GPU has work;
# - whether phase 6b's copies from VRAM change it.
#
#   bench/fresh_check.sh EULLM_BINARY MODEL.gguf [MORE SERVE FLAGS...]
#
# For each setting of SETTINGS ("4 4:bus 4:bus 4" by default, so that running
# second shows apart from the setting) the script starts a new server:
#   `eullm serve --moe-prefetch N --ctx-size CTX --moe-cache MOE_CACHE`
# plus the flags after MODEL. A setting N:bus runs the server with
# LLAMA_MOE_PREFETCH_FROM_CACHE=0, as bench/prefetch_check.sh does: every
# expert of a long prompt over the bus. On each server:
# 1. One answer of ANSWER_TOKENS (512) to bench/speed_check.py's story,
#    greedy. Meanwhile nvidia-smi samples the GPU every 200 ms; "busy" is the
#    mean of its utilization.gpu, the share of the time a kernel was running.
# 2. bench/speed_check.py, which writes before it reads, so its writing is
#    that of a server before any long prompt.
#
# With STATS_EVERY=N the servers also print the expert cache's statistics
# every N decode steps (LLAMA_MOE_CACHE_STATS, patch 0002), and each line
# gives their median:
#   ms a step = to the routers + in the cache + after, then the share of
#   experts found in VRAM.
# Those statistics wait for the copies of one step in eight, which slows the
# writing a little: leave them off for the speeds.
#
# One line per server; each server's log, answer and samples stay in $OUT.
set -u
export LC_ALL=C

BIN=${1:?usage: bench/fresh_check.sh EULLM_BINARY MODEL.gguf [serve flags]}
MODEL=${2:?usage: bench/fresh_check.sh EULLM_BINARY MODEL.gguf [serve flags]}
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-11535}
SPEED_CHECK=${SPEED_CHECK:-$HERE/speed_check.py}
OUT=${OUT:-$HOME/work/fresh-check}
CTX=${CTX:-40960}
MOE_CACHE=${MOE_CACHE:-auto}
PROMPT_TOKENS=${PROMPT_TOKENS:-33200}
SETTINGS=${SETTINGS:-"4 4:bus 4:bus 4"}
STATS_EVERY=${STATS_EVERY:-}
ANSWER_TOKENS=${ANSWER_TOKENS:-512}

if ! python3 "$SPEED_CHECK" --help 2>/dev/null | grep -q -- --temperature; then
    echo "$SPEED_CHECK is missing or too old (no --temperature): use bench/speed_check.py" >&2
    exit 1
fi
if ! "$BIN" serve --help 2>/dev/null | grep -q -- --moe-prefetch; then
    echo "$BIN has no --moe-prefetch: build the engine from this checkout" >&2
    exit 1
fi
pid=
sampler=
trap '[[ -n $sampler ]] && kill "$sampler" 2>/dev/null; [[ -n $pid ]] && kill "$pid" 2>/dev/null' EXIT
trap 'exit 130' INT TERM

mkdir -p "$OUT"
PYTHONPATH="$HERE" python3 - "$MODEL" "$ANSWER_TOKENS" >"$OUT/story.json" <<'EOF' || exit 1
import json
import sys

from speed_check import STORY

print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": STORY}],
    "max_tokens": int(sys.argv[2]),
    "temperature": 0,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "cache_prompt": False,
    "stream": False,
    "think": False,
}))
EOF

# The median of the cache's decode lines:
# "llama_moe_cache: 64 steps of 1.0 tokens: 17.5 ms/step = 13.8 ms to the routers of 48 layers
#  + 0.3 ms in the cache + 3.4 ms after; 60.1 MiB/step copied, 90.2% of the experts in VRAM; ..."
step_median() {
    python3 -I - "$1" <<'EOF'
import re
import statistics
import sys

pat = re.compile(r"steps of 1\.0 tokens: ([\d.]+) ms/step = ([\d.]+) ms to the routers of \d+ layers"
                 r" \+ ([\d.]+) ms in the cache \+ (-?[\d.]+) ms after; [\d.]+ MiB/step copied, ([\d.]+)% of the experts")
rows = [tuple(map(float, m.groups())) for m in map(pat.search, open(sys.argv[1], encoding="utf-8", errors="replace")) if m]
if not rows:
    print("-")
else:
    s, w, c, a, v = (statistics.median(col) for col in zip(*rows))
    print(f"{s:.1f}={w:.1f}+{c:.1f}+{a:.1f}ms,{v:.0f}%")
EOF
}

printf '%-10s %6s %12s %10s %10s %10s %8s  %s\n' setting busy answer_tok/s writes first_run reads cache step
i=0
for setting in $SETTINGS; do
    i=$((i + 1))
    slots=${setting%%:*}
    p=$i-${setting/:/-} # the server in file names
    vars=()
    [[ $setting == *:bus ]] && vars+=(LLAMA_MOE_PREFETCH_FROM_CACHE=0)
    [[ -n $STATS_EVERY ]] && vars+=(LLAMA_MOE_CACHE_STATS="$STATS_EVERY")
    log="$OUT/serve-$p.log"
    env "${vars[@]}" "$BIN" serve --port "$PORT" --default-model "$MODEL" --moe-prefetch "$slots" \
        --ctx-size "$CTX" --moe-cache "$MOE_CACHE" "$@" >"$log" 2>&1 &
    pid=$!
    up=
    for _ in $(seq 1 600); do
        curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null && { up=1; break; }
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if [[ -z $up ]]; then
        printf '%-10s did not start: %s\n' "$setting" "$log"
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        pid=
        continue
    fi

    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 200 \
            >"$OUT/busy-$p.csv" 2>/dev/null &
        sampler=$!
    fi
    t0=$(date +%s.%N)
    curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d @"$OUT/story.json" >"$OUT/answer-$p.json"
    t1=$(date +%s.%N)
    if [[ -n $sampler ]]; then
        kill "$sampler" 2>/dev/null
        wait "$sampler" 2>/dev/null
        sampler=
    fi
    # The first and last second aside: the short prompt, and the reply's way back.
    busy=$(awk 'NF {v[++n] = $1} END {for (k = 6; k <= n - 5; k++) {s += v[k]; m++}
        if (m) printf "%.0f%%", s / m; else print "-"}' "$OUT/busy-$p.csv" 2>/dev/null)
    answer=$(python3 -I -c 'import json, sys
try:
    n = json.load(open(sys.argv[1], encoding="utf-8"))["usage"]["completion_tokens"]
    print(f"{n / (float(sys.argv[3]) - float(sys.argv[2])):.1f}")
except Exception:
    print("?")' "$OUT/answer-$p.json" "$t0" "$t1")

    python3 "$SPEED_CHECK" --url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
        --prompt-tokens "$PROMPT_TOKENS" >"$OUT/speed-$p.txt" 2>&1
    kill "$pid"
    wait "$pid" 2>/dev/null
    pid=
    read -r writes first reads <<<"$(awk '
        /writes answers/ {w = $3; if (match($0, /warming up: [0-9.]+/)) f = substr($0, RSTART + 12, RLENGTH - 12)}
        /reads a prompt/ {r = $4}
        END {print (w ? w : "?"), (f ? f : "?"), (r ? r : "?")}' "$OUT/speed-$p.txt")"
    # the --fit line: "expert tensors in CPU RAM, 5.75 GiB of VRAM caching ..."
    cache=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -o 'CPU RAM, [0-9.]* GiB of VRAM caching' |
        head -n 1 | awk '{print $3 "G"}')
    step=-
    [[ -n $STATS_EVERY ]] && step=$(step_median "$log")
    printf '%-10s %6s %12s %10s %10s %10s %8s  %s\n' "$setting" "${busy:--}" "$answer" "$writes" "$first" \
        "$reads" "${cache:--}" "$step"
done
