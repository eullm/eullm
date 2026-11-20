#!/usr/bin/env bash
# Several generation models resident at once (`--max-loaded-models`), checked
# on real hardware: what the engine's CPU tests cannot see — how the models
# are sized on a GPU, whether a third model waits for busy ones instead of
# cutting their answers off, whether loading one holds up requests to
# another. Each check prints PASS or FAIL; the logs stay in $OUT.
#
#   tools/residency_check.sh <eullm built from main> <eullm built from the branch>
#
# 1. One model at a time, as main: the same models loaded in the same order
#    must get the same GPU layers and context from both binaries.
# 2. Two models side by side: both answer, both stay loaded, both whole on
#    the GPU; with nvidia-smi, /api/ps's VRAM against the GPU's (2b).
# 3. A third model while the two are streaming: it waits for one of them to
#    finish (or gets a 503 after the wait), and neither stream is cut off.
# 4. A load does not stall a resident model: time to first token of model A
#    while model B loads, against A alone (the plan's gate: within 10%).
# 5. Optional, BIG=<model>: a model that cannot fit beside the others
#    unloads what it must, and is never split while another is resident.
# 6. Optional, VISION=<model>: image requests to a vision model beside a
#    text model, without a failed context allocation.
# 7. An embedding model loaded by a request beside the two (V5): it unloads
#    the least recently used of them when it does not fit, and only it — it
#    used to unload both — and the churn counters say what each round costs.
#    The other half of V5, a third model while two answers stream, is check 3.
#
# Models are catalog names, pulled when missing (SKIP_PULL=1 to skip):
# SMALL=qwen3-1.7b A=qwen3-4b B=qwen3-8b; BIG and VISION only when set.
# EMBED=qwen3-embedding-0.6b-gguf-q8_0, pulled from EMBED_PULL
# (hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0); EMBED= skips check 7.
# EMBED_CTX="20480 16384 12288 8192": the contexts check 7 tries for A and B,
# largest first; it keeps the first at which the two sit side by side, which
# leaves the embedder the least room. 20480 alone was too much for a 16 GB
# card: loading B unloaded A, and the check had nothing to measure.
# PORT=11500, OUT=/tmp/residency-check. Needs curl and python3.
set -u
# Numbers with a decimal point whatever the locale: awk and printf read and
# write them, and an Italian locale writes "6,0".
export LC_ALL=C

if [ $# -ne 2 ]; then
    sed -n '2,35p' "$0"
    exit 2
fi
MAIN=$(realpath "$1")
BRANCH=$(realpath "$2")
PORT=${PORT:-11500}
OUT=$(realpath -m "${OUT:-/tmp/residency-check}")
SMALL=${SMALL:-qwen3-1.7b}
A=${A:-qwen3-4b}
B=${B:-qwen3-8b}
BIG=${BIG:-}
VISION=${VISION:-}
EMBED=${EMBED-qwen3-embedding-0.6b-gguf-q8_0}
EMBED_PULL=${EMBED_PULL:-hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0}
EMBED_CTX=${EMBED_CTX:-20480 16384 12288 8192}
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"
: > "$OUT/summary.txt"
PASS=0
FAIL=0
PID=

result() {
    if [ "$1" = PASS ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
    printf '%-4s  %s\n' "$1" "$2" | tee -a "$OUT/summary.txt"
}

# Start a server (binary, log file, flags...) and wait until it answers. It
# runs in $OUT, where no .env sets API keys or an IP allowlist, and audits
# there too, not into the real audit trail.
up() {
    local bin=$1 log=$2
    shift 2
    # A server already on the port would answer for ours, and every check
    # would measure it instead.
    if curl -sf "$URL/api/version" > /dev/null; then
        echo "something already answers on port $PORT: stop it, or set PORT"
        return 1
    fi
    (cd "$OUT" && EULLM_AUDIT_DIR="$OUT/audit" exec "$bin" serve --port "$PORT" "$@") > "$log" 2>&1 &
    PID=$!
    for _ in $(seq 1 600); do
        curl -sf "$URL/api/version" > /dev/null && return 0
        kill -0 "$PID" 2> /dev/null || { echo "the server exited, see $log"; return 1; }
        sleep 0.5
    done
    echo "the server did not start, see $log"
    return 1
}

down() {
    if [ -n "$PID" ]; then
        kill "$PID" 2> /dev/null
        wait "$PID" 2> /dev/null
        PID=
    fi
}
trap down EXIT

# POST a JSON body to an endpoint: post <path> <body> [curl options...].
post() {
    local path=$1 body=$2
    shift 2
    curl -s "$@" -H 'Content-Type: application/json' "$URL$path" -d "$body"
}

# One short answer from `model`: prints "<model> <done_reason>" or "ERROR <text>".
ask() {
    post /api/chat "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one sentence.\"}],\"stream\":false,\"options\":{\"num_predict\":16}}" |
        python3 -c 'import json,sys
t=sys.stdin.read()
try:
    d=json.loads(t)
except ValueError:
    print("ERROR", t[:200].replace("\n"," ")); sys.exit()
print("ERROR", d["error"]) if "error" in d else print(d.get("model"), d.get("done_reason"))'
}

# The generation models /api/ps lists, sorted, one line.
resident() {
    curl -s "$URL/api/ps" | python3 -c 'import json,sys
d=json.load(sys.stdin)
print(" ".join(sorted(m["name"] for m in d.get("models",[]) if (m.get("eullm") or {}).get("slot","generation")=="generation")))'
}

# Time to the first token of a one-token answer from `model`, in ms, timed
# inside python3 to the first streamed line (curl's own timer stops at the
# response headers, sent before any token); "failed" when there is no token.
ttft() {
    python3 - "$PORT" "$1" << 'PY'
import http.client, json, sys, time
c = http.client.HTTPConnection("127.0.0.1", int(sys.argv[1]), timeout=600)
body = json.dumps({"model": sys.argv[2], "prompt": "Hello", "stream": True, "options": {"num_predict": 1}})
t = time.perf_counter()
c.request("POST", "/api/generate", body, {"Content-Type": "application/json"})
r = c.getresponse()
line = r.readline() if r.status == 200 else b""
ms = (time.perf_counter() - t) * 1000
r.read()
try:
    ok = "response" in json.loads(line) and "error" not in json.loads(line)
except ValueError:
    ok = False
print(f"{ms:.1f}" if ok else "failed")
PY
}

# How many loads a log shows split between GPU and RAM.
partials() {
    sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -cE "offloading [0-9]+/[0-9]+ layers|--fit: MoE model"
}

# VRAM in use on the first GPU, in MiB; empty without nvidia-smi.
vram() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1
}

# The sizing decisions a log shows: each loaded model's GPU layers and context.
decisions() {
    sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -E '^ *(Model|GPU layers|Context|Mode):' | sed -E 's/ +/ /g'
}

if [ "${SKIP_PULL:-0}" != 1 ]; then
    for m in "$SMALL" "$A" "$B" ${BIG:+"$BIG"} ${VISION:+"$VISION"}; do
        "$BRANCH" list 2> /dev/null | grep -q "^$m[[:space:]]" || "$BRANCH" pull "$m" || exit 1
    done
    if [ -n "$EMBED" ]; then
        "$BRANCH" list 2> /dev/null | grep -q "^$EMBED[[:space:]]" || "$BRANCH" pull "$EMBED_PULL" || exit 1
    fi
fi
echo "main:   $("$MAIN" -V)"
echo "branch: $("$BRANCH" -V)"
echo "models: SMALL=$SMALL A=$A B=$B${BIG:+ BIG=$BIG}${VISION:+ VISION=$VISION}${EMBED:+ EMBED=$EMBED}"
others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2> /dev/null)
if [ -n "$others" ]; then
    echo "Other programs hold VRAM, so the sizes will not compare; stop them first:"
    echo "$others"
fi
echo

# 1. One model at a time, as main.
for which in main branch; do
    bin=$MAIN
    [ "$which" = branch ] && bin=$BRANCH
    up "$bin" "$OUT/1-$which.log" || exit 1
    for m in "$A" "$B" "$A" ${BIG:+"$BIG"}; do ask "$m" >> "$OUT/1-$which.answers"; done
    down
    decisions "$OUT/1-$which.log" > "$OUT/1-$which.decisions"
done
if [ -s "$OUT/1-main.decisions" ] && diff -q "$OUT/1-main.decisions" "$OUT/1-branch.decisions" > /dev/null &&
    ! grep -q ERROR "$OUT/1-branch.answers"; then
    result PASS "1 one model at a time: the same sizing as main ($(grep -c 'Model:' "$OUT/1-branch.decisions") loads)"
else
    result FAIL "1 one model at a time: sizing differs from main or an answer failed (diff $OUT/1-main.decisions $OUT/1-branch.decisions)"
fi

# 2. Two models side by side, both whole on the GPU.
up "$BRANCH" "$OUT/2.log" --max-loaded-models 2 || exit 1
before=$(vram)
a=$(ask "$A")
b=$(ask "$B")
after=$(vram)
loaded=$(resident)
split=$(partials "$OUT/2.log")
if [[ $a != ERROR* && $b != ERROR* ]] && [ "$loaded" = "$(printf '%s\n' "$A" "$B" | sort | tr '\n' ' ' | sed 's/ $//')" ] &&
    [ "$split" = 0 ] && ! grep -q "Unloaded" "$OUT/2.log"; then
    result PASS "2 two models side by side: $loaded, neither split between GPU and RAM${before:+ (VRAM ${before} -> ${after} MiB)}"
else
    result FAIL "2 two models side by side: answers '$a' / '$b', resident '$loaded', $split split loads (see $OUT/2.log)"
fi
# What /api/ps says the two hold, against what the GPU shows (the plan: within 5%).
if [ -n "$before" ]; then
    ps_mib=$(curl -s "$URL/api/ps" | python3 -c 'import json,sys
d=json.load(sys.stdin)
print(round(sum(m.get("size_vram",0) for m in d.get("models",[]) if (m.get("eullm") or {}).get("slot")=="generation")/2**20))')
    delta=$((after - before))
    off=$(awk "BEGIN{printf \"%.1f\", ($delta ? 100*($ps_mib-$delta)/$delta : 100)}")
    if awk "BEGIN{exit !($off <= 5 && $off >= -5)}"; then
        result PASS "2b /api/ps size_vram against nvidia-smi: $ps_mib MiB against $delta MiB ($off%)"
    else
        result FAIL "2b /api/ps size_vram against nvidia-smi: $ps_mib MiB against $delta MiB ($off%, the plan allows 5%)"
    fi
fi

# 3. A third model while both stream.
stream() {
    post /api/generate "{\"model\":\"$1\",\"prompt\":\"Write a long story about a lighthouse keeper.\",\"stream\":true,\"options\":{\"num_predict\":${STREAM_TOKENS:-1500}}}" -N > "$OUT/3-$1.ndjson"
}
stream "$A" &
s1=$!
stream "$B" &
s2=$!
sleep 2
t0=$(date +%s.%N)
c=$(ask "$SMALL")
t1=$(date +%s.%N)
wait "$s1" "$s2"
ok=1
for m in "$A" "$B"; do
    tail -1 "$OUT/3-$m.ndjson" | grep -q '"done":true' || ok=0
    grep -q '"error"' "$OUT/3-$m.ndjson" && ok=0
done
waited=$(grep -c "waiting up to" "$OUT/2.log")
if [ $ok = 1 ] && { [[ $c != ERROR* ]] || [[ $c == *"Retry"* ]]; }; then
    result PASS "3 a third model while two stream: both streams finished whole; $SMALL answered after $(awk "BEGIN{printf \"%.1f\", $t1-$t0}") s ($waited wait)"
else
    result FAIL "3 a third model while two stream: streams whole=$ok, third answer '$c' (see $OUT/3-*.ndjson, $OUT/2.log)"
fi
down

# 4. A load does not stall requests to a resident model: the plan's gate is
#    A's time to first token within 10% while B loads (5 ms of slack for the
#    timer). An empty prompt loads B without generating, so only the load
#    overlaps the samples.
up "$BRANCH" "$OUT/4.log" --max-loaded-models 2 || exit 1
ask "$A" > /dev/null
for _ in $(seq 1 20); do ttft "$A"; done > "$OUT/4-alone.ms"
post /api/generate "{\"model\":\"$B\"}" > "$OUT/4-load.json" &
loader=$!
while kill -0 "$loader" 2> /dev/null; do ttft "$A"; done > "$OUT/4-during.ms"
wait "$loader"
down
median() { grep -v failed "$1" | sort -n | awk '{v[NR]=$1} END {if (NR) printf "%.1f", (NR%2 ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2); else print "none"}'; }
idle=$(median "$OUT/4-alone.ms")
dur=$(median "$OUT/4-during.ms")
n=$(grep -cv failed "$OUT/4-during.ms")
failed=$(cat "$OUT/4-alone.ms" "$OUT/4-during.ms" | grep -c failed)
worst=$(grep -v failed "$OUT/4-during.ms" | sort -n | tail -1)
loaded=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("done_reason"))' "$OUT/4-load.json" 2> /dev/null)
if [ "$failed" -gt 0 ] || [ "$idle" = none ] || [ "$loaded" != load ]; then
    result FAIL "4 a load does not stall a resident model: $failed failed requests, load of $B '$loaded' (see $OUT/4.log)"
elif [ "$n" -eq 0 ]; then
    result PASS "4 a load does not stall a resident model: $B loaded before a sample could be taken (TTFT alone ${idle} ms)"
elif awk "BEGIN{exit !($dur <= 1.10*$idle + 5)}"; then
    result PASS "4 a load does not stall a resident model: TTFT median ${idle} ms alone, ${dur} ms during the load (worst ${worst} ms, $n samples)"
else
    result FAIL "4 a load does not stall a resident model: TTFT median ${idle} ms alone, ${dur} ms during the load (worst ${worst} ms, $n samples; the gate is +10%)"
fi

# 5. A model too large to sit beside the others: it may load whole beside
#    one of them, or alone, but never split while another model is resident.
if [ -n "$BIG" ]; then
    up "$BRANCH" "$OUT/5.log" --max-loaded-models 2 || exit 1
    ask "$A" > /dev/null
    ask "$B" > /dev/null
    big=$(ask "$BIG")
    loaded=$(resident)
    down
    split=$(partials "$OUT/5.log")
    unloaded=$(grep -c Unloaded "$OUT/5.log")
    if [[ $big != ERROR* ]] && { [ "$loaded" = "$BIG" ] || [ "$split" = 0 ]; }; then
        how="whole beside $(echo "$loaded" | tr ' ' '\n' | grep -vx "$BIG" | tr '\n' ' ' | sed 's/ $//')"
        [ "$loaded" = "$BIG" ] && how="alone"
        [ "$split" -gt 0 ] && how="$how, split between GPU and RAM"
        result PASS "5 a model too large to share: $BIG loaded $how ($unloaded unloaded)"
    else
        result FAIL "5 a model too large to share: answer '$big', resident '$loaded', $split split loads (see $OUT/5.log)"
    fi
fi

# 6. Images beside text.
if [ -n "$VISION" ]; then
    up "$BRANCH" "$OUT/6.log" --max-loaded-models 2 || exit 1
    # The body goes through a file: the image alone is near the size limit
    # of one command-line argument.
    printf '{"model":"%s","messages":[{"role":"user","content":"What does this chart show?","images":["%s"]}],"stream":false,"options":{"num_predict":32}}' \
        "$VISION" "$(base64 -w0 "$(dirname "$0")/../bench/results/chart_quality_comparison.png")" > "$OUT/6-body.json"
    errors=0
    for _ in $(seq 1 10); do
        r=$(post /api/chat "@$OUT/6-body.json")
        [[ $r == *'"error"'* ]] && errors=$((errors + 1))
        [[ $(ask "$A") == ERROR* ]] && errors=$((errors + 1))
    done
    alloc=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/6.log" | grep -ciE "failed to create|could not allocate|out of memory")
    down
    if [ $errors = 0 ] && [ "$alloc" = 0 ]; then
        result PASS "6 images beside text: 20 requests, no errors"
    else
        result FAIL "6 images beside text: $errors errors, $alloc allocation failures (see $OUT/6.log)"
    fi
fi

# 7. An embedder loaded by a request beside two generation models (V5).
#    Each round asks A, then B — B is then the more recently used — and embeds
#    once. When the embedder does not fit beside both, A must go and B stay:
#    one model unloaded, never both, never the more recent one. The next
#    round's request to A loads it back, which unloads the embedder (it is
#    not reserved): the counters below are that churn, round after round.
embed() {
    post /api/embed "{\"model\":\"$1\",\"input\":[\"A lighthouse keeper's log, first entry.\",\"Second entry: the lamp is lit.\"]}" |
        python3 -c 'import json,sys
t=sys.stdin.read()
try:
    d=json.loads(t)
except ValueError:
    print("ERROR", t[:200].replace("\n"," ")); sys.exit()
print("ERROR", d["error"]) if "error" in d else print("ok", len(d.get("embeddings",[])))'
}
# "<model_swaps> <generation_evictions>" from /api/version.
counters() {
    curl -s "$URL/api/version" | python3 -c 'import json,sys
d=json.load(sys.stdin)
print(d.get("model_swaps"), d.get("generation_evictions"))'
}
if [ -n "$EMBED" ]; then
    pair=$(printf '%s\n' "$A" "$B" | sort | tr '\n' ' ' | sed 's/ $//')
    # The largest context at which A and B sit side by side.
    for ctx in $EMBED_CTX; do
        up "$BRANCH" "$OUT/7.log" --max-loaded-models 2 --ctx-size "$ctx" || exit 1
        ask "$A" > /dev/null
        ask "$B" > /dev/null
        [ "$(resident)" = "$pair" ] && break
        down
    done
    [ -n "$PID" ] || up "$BRANCH" "$OUT/7.log" --max-loaded-models 2 --ctx-size "$ctx" || exit 1
    rounds=""
    ok=1
    evicting=0
    for round in 1 2 3; do
        a=$(ask "$A")
        b=$(ask "$B")
        before_round=$(resident)
        read -r swaps0 _ <<< "$(counters)"
        e=$(embed "$EMBED")
        read -r swaps1 gen1 <<< "$(counters)"
        after_round=$(resident)
        went=$((swaps1 - swaps0))
        rounds="$rounds [$round: $went unloaded, left '$after_round']"
        if [[ $a == ERROR* || $b == ERROR* || $e == ERROR* ]]; then
            ok=0
            rounds="$rounds (answers '$a' / '$b', embed '$e')"
        elif [ "$before_round" != "$pair" ]; then
            ok=0
            rounds="$rounds ($A and $B were not both loaded before embedding at --ctx-size $ctx: '$before_round')"
        elif [ "$went" = 0 ]; then
            [ "$after_round" = "$pair" ] || ok=0
        else
            evicting=1
            { [ "$went" = 1 ] && [ "$after_round" = "$B" ]; } || ok=0
        fi
    done
    down
    if [ $ok = 1 ] && [ $evicting = 1 ]; then
        result PASS "7 an embedder beside two models at --ctx-size $ctx: the least recently used goes, and only it —$rounds; model_swaps $swaps1, generation_evictions $gen1"
    elif [ $ok = 1 ]; then
        result PASS "7 an embedder beside two models at --ctx-size $ctx: it fit beside both, nothing unloaded —$rounds (no context left the embedder too little room: see EMBED_CTX)"
    else
        result FAIL "7 an embedder beside two models:$rounds; model_swaps $swaps1 (see $OUT/7.log)"
    fi
fi

echo
echo "$PASS passed, $FAIL failed. Logs and the summary are in $OUT."
[ "$FAIL" = 0 ]
