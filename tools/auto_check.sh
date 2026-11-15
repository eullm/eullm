#!/usr/bin/env bash
# `"model": "auto"` checked on real hardware: what the engine's CPU tests
# cannot see — whether the router is fast enough on a GPU to be worth its
# place, whether it holds up under concurrency and over an hour of mixed
# traffic, and what it costs on a machine without a GPU. Each check prints
# PASS or FAIL; the logs, reports and the summary stay in $OUT.
#
#   tools/auto_check.sh <eullm built from the branch> <decision GGUF> [<second decision GGUF>]
#
# The decision models are GGUF paths (--decision-model): the Jev-Style 0.8B,
# and optionally the 2B, which the routing checks are then repeated with.
#
# 1. Routed answers (V7): a server routing between the pair, its candidates
#    warmed up at start; "model": "auto" on /api/generate, /api/chat and
#    /v1/chat/completions, streamed and not, answers with a candidate and
#    says so in the X-EuLLM-* headers and the last line or chunk.
# 2. Router latency (V7): /api/route at concurrency 1, 4 and 16. The plan's
#    gate, at concurrency 1: p50 <= 60 ms and p95 <= 150 ms of decision_ms.
#    At 4 and 16 the decisions queue on one worker: reported, with the share
#    that timed out to the fallback.
# 3. AutoBench (V7): bench/reflexbench/autobench.py on each pair with each
#    decision model — calls avoided, accuracy against always-large with its
#    95% CI, router p50/p95, and end to end at concurrency 1, 4 and 16 the
#    share of answers identical to the chosen model's alone (99% expected)
#    and the time to the first token — the tables in $OUT/3-*.md. Stage 1's
#    answers are kept per pair in $OUT, so the second decision model reuses
#    them.
# 4. Soak (V8): DURATION seconds (one hour) of /api/chat, /v1/chat/completions,
#    /api/embed, /v1/systemone and "auto" at concurrency 8, VRAM sampled every
#    5 s: no failed request, VRAM flat after the first minute (within 5%), no
#    out-of-memory or GGML_ASSERT in the log, every audit line parses and
#    there is one per request of each kind.
# 5. Optional, CPU_DECISION=<Qwen3-0.6B GGUF> (V9): the same pair's smaller
#    model and CPU_SMALL beside it, routed with the GPU hidden
#    (CUDA_VISIBLE_DEVICES=), N=2: router latency at concurrency 1 on a CPU,
#    reported as measured — with OMP_WAIT_POLICY=PASSIVE, as a CPU-only
#    server should run (see docs/engine.md).
#
# Models are catalog names, pulled when missing (SKIP_PULL=1 to skip):
# PAIRS="qwen3-4b,qwen3-8b qwen3-1.7b,qwen3-14b" (pairs A and B of the plan,
# small first; check 1, 2 and 4 use the first), CTX=8192 for pair A — pair B
# runs at 4096 with --decision-ctx 4096, as the plan sizes it.
# EMBED=qwen3-embedding-0.6b-gguf-q8_0, pulled from EMBED_PULL
# (hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0): reserved at launch for
# check 4, and for the kNN baseline of check 3 loaded on demand and unloaded
# before its end-to-end stage — reserved there too, its 2 GiB would keep the
# pair from sitting side by side on a 16 GB card. LIMIT=100 items per set, SETS=gsm8k,arc-easy,
# arc-challenge,mmlu. DURATION=3600. CPU_SMALL=qwen3-1.7b.
# CHECKS="1 2 3 4 5": the checks run (1 and 2 share a server, so one runs the
# other); CHECKS=4 is the soak alone. SOAK_CTX="8192 6144 4096 2048": the
# contexts the soak tries, the first at which the pair sits side by side.
# PORT=11500, OUT=/tmp/auto-check. Needs curl and python3.
set -u
# Numbers with a decimal point whatever the locale.
export LC_ALL=C

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    sed -n '2,52p' "$0"
    exit 2
fi
BIN=$(realpath "$1")
DECISIONS=("$(realpath "$2")")
[ $# = 3 ] && DECISIONS+=("$(realpath "$3")")
for d in "${DECISIONS[@]}"; do
    [ -f "$d" ] || { echo "no decision model at $d"; exit 2; }
done
PAIRS=${PAIRS:-qwen3-4b,qwen3-8b qwen3-1.7b,qwen3-14b}
CTX=${CTX:-8192}
EMBED=${EMBED-qwen3-embedding-0.6b-gguf-q8_0}
EMBED_PULL=${EMBED_PULL:-hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0}
LIMIT=${LIMIT:-100}
SETS=${SETS:-gsm8k,arc-easy,arc-challenge,mmlu}
DURATION=${DURATION:-3600}
CPU_DECISION=${CPU_DECISION:-}
CPU_SMALL=${CPU_SMALL:-qwen3-1.7b}
PORT=${PORT:-11500}
OUT=$(realpath -m "${OUT:-/tmp/auto-check}")
URL=http://127.0.0.1:$PORT
BENCH=$(dirname "$(realpath "$0")")/../bench/reflexbench/autobench.py
read -r SMALL LARGE <<< "$(echo "${PAIRS%% *}" | tr ',' ' ')"
CHECKS=${CHECKS:-1 2 3 4 5}
want() { [[ " $CHECKS " == *" $1 "* ]]; }
mkdir -p "$OUT"
: > "$OUT/summary.txt"
PASS=0
FAIL=0
PID=

result() {
    if [ "$1" = PASS ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
    printf '%-4s  %s\n' "$1" "$2" | tee -a "$OUT/summary.txt"
}

# Start a server (log file, flags...) and wait until it answers. It runs in
# $OUT, where no .env sets API keys or an IP allowlist, and audits into
# $OUT/audit-<log name>, not into the real audit trail.
up() {
    local log=$1
    shift
    local audit
    audit="$OUT/audit-$(basename "$log" .log)"
    rm -rf "$audit"
    # A server already on the port would answer for ours, and every check
    # would measure it instead.
    if curl -sf "$URL/api/version" > /dev/null; then
        echo "something already answers on port $PORT: stop it, or set PORT"
        return 1
    fi
    (cd "$OUT" && EULLM_AUDIT_DIR="$audit" exec "$BIN" serve --port "$PORT" "$@") > "$log" 2>&1 &
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

# Wait until the log says the warm-up is over, up to 10 minutes.
warmed() {
    for _ in $(seq 1 1200); do
        grep -q "Auto routing warm-up:" "$1" && return 0
        sleep 0.5
    done
    return 1
}

# A server routing between `small` and `large` with `decision`, as the plan
# sizes it: routed <log> <small> <large> <decision> [flags...].
routed() {
    local log=$1 small=$2 large=$3 decision=$4
    shift 4
    up "$log" --max-loaded-models 2 \
        --auto-model "$small=Short everyday requests, simple facts and quick rewrites" \
        --auto-model "$large=Multi-step reasoning, maths, code, analysis and long answers" \
        --decision-model "$decision" "$@"
}

# Whether the warm-up left every candidate resident: "yes", or the line that
# says why not. A pair that cannot sit side by side swaps on every change of
# route, and every number after that measures the swaps.
side_by_side() {
    local clean
    clean=$(sed 's/\x1b\[[0-9;]*m//g' "$1")
    if echo "$clean" | grep -q "Auto routing warm-up: \([0-9]*\) of \1 candidates"; then
        echo yes
    else
        echo "$clean" | grep -o "Auto routing warm-up stops: .*" | head -1
    fi
}

# /api/route at each concurrency, `n` requests each: one line per level,
# "<concurrency> <p50 ms> <p95 ms> <timeouts> <decided> <wall p50 ms>".
router_latency() {
    python3 - "$URL" "$1" "${@:2}" << 'PY'
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
url, n, levels = sys.argv[1], int(sys.argv[2]), [int(c) for c in sys.argv[3:]]
asks = [
    "What is the capital of France?",
    "Prove that the square root of 2 is irrational, step by step.",
    "Write a Python function that merges two sorted lists, with tests.",
    "Translate 'good morning' into Italian.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the journey, and what if it is 25 minutes late?",
    "Summarise the causes of the First World War in three paragraphs.",
    "Is 391 a prime number?",
    "Say hello.",
]
def one(i):
    body = {"model": "auto", "messages": [{"role": "user", "content": f"{asks[i % len(asks)]} (#{i})"}]}
    req = urllib.request.Request(url + "/api/route", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        route = json.load(r)
    return route, (time.perf_counter() - t) * 1000
def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")
for c in levels:
    with ThreadPoolExecutor(c) as pool:
        got = list(pool.map(one, range(n)))
    ms = [r["decision_ms"] for r, _ in got]
    wall = [w for _, w in got]
    timeouts = sum(r["reason"] == "timeout" for r, _ in got)
    decided = sum(r["reason"] == "decided" for r, _ in got)
    print(c, f"{pct(ms, 0.5):.1f}", f"{pct(ms, 0.95):.1f}", timeouts, decided, f"{pct(wall, 0.5):.1f}")
PY
}

if [ "${SKIP_PULL:-0}" != 1 ]; then
    for m in $(echo "$PAIRS" | tr ', ' '\n\n' | sort -u) ${CPU_DECISION:+"$CPU_SMALL"}; do
        "$BIN" list 2> /dev/null | grep -q "^$m[[:space:]]" || "$BIN" pull "$m" || exit 1
    done
    if [ -n "$EMBED" ]; then
        "$BIN" list 2> /dev/null | grep -q "^$EMBED[[:space:]]" || "$BIN" pull "$EMBED_PULL" || exit 1
    fi
fi
echo "eullm:     $("$BIN" -V)"
echo "pairs:     $PAIRS (checks 1, 2 and 4: $SMALL and $LARGE)"
echo "decisions: ${DECISIONS[*]}"
others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2> /dev/null)
if [ -n "$others" ]; then
    echo "Other programs hold VRAM, so the numbers will not compare; stop them first:"
    echo "$others"
fi
echo

# 1 and 2, with each decision model.
if want 1 || want 2; then
n=0
for decision in "${DECISIONS[@]}"; do
    n=$((n + 1))
    dname=$(basename "$decision" .gguf)
    routed "$OUT/1-$n.log" "$SMALL" "$LARGE" "$decision" --ctx-size "$CTX" || exit 1
    warmed "$OUT/1-$n.log"
    warm=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/1-$n.log" | grep -o "Auto routing warm-up: .*" | head -1)
    answers=$(python3 - "$URL" << 'PY'
import json, sys, urllib.request
url = sys.argv[1]
bad = []
for path in ("/api/generate", "/api/chat", "/v1/chat/completions"):
    for stream in (False, True):
        body = {"model": "auto", "stream": stream, "think": False, "max_tokens": 16,
                "options": {"num_predict": 16}}
        if path == "/api/generate":
            body["prompt"] = "Name a colour."
        else:
            body["messages"] = [{"role": "user", "content": "Name a colour."}]
        req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                model, reason = r.headers.get("X-EuLLM-Model"), r.headers.get("X-EuLLM-Route")
                lines = [l.decode().strip() for l in r]
        except Exception as e:
            bad.append(f"{path} stream={stream}: {e}")
            continue
        lines = [l[5:].strip() if l.startswith("data:") else l for l in lines]
        objects = [json.loads(l) for l in lines if l and l != "[DONE]"]
        route = (objects[-1].get("eullm") or {}).get("route") if objects else None
        if not model or not route or route.get("model") != model or any(o.get("model") != model for o in objects):
            bad.append(f"{path} stream={stream}: header {model}/{reason}, route {route}")
        else:
            print(f"{path}{' (stream)' if stream else ''}: {model} ({reason})", file=sys.stderr)
print("; ".join(bad) if bad else "ok")
PY
    )
    both=$(side_by_side "$OUT/1-$n.log")
    if [ "$answers" = ok ] && [ "$both" = yes ]; then
        result PASS "1 routed answers with $dname on every endpoint, streamed and not; $warm"
    elif [ "$answers" = ok ]; then
        result FAIL "1 routed answers with $dname, but the pair does not stay loaded side by side: $both (see $OUT/1-$n.log)"
    else
        result FAIL "1 routed answers with $dname: $answers (see $OUT/1-$n.log)"
    fi

    router_latency 48 1 4 16 > "$OUT/2-$n.txt"
    read -r _ p50 p95 timeouts decided _ < <(head -1 "$OUT/2-$n.txt")
    levels=$(awk '{printf " c=%s p50 %s p95 %s ms, %s/48 timed out, wall p50 %s ms;", $1, $2, $3, $4, $6}' "$OUT/2-$n.txt")
    if [ "$decided" -gt 0 ] && awk "BEGIN{exit !($p50 <= 60 && $p95 <= 150)}"; then
        result PASS "2 router latency with $dname:$levels the gate (c=1 p50 <= 60, p95 <= 150 ms) holds"
    else
        result FAIL "2 router latency with $dname:$levels the gate is c=1 p50 <= 60, p95 <= 150 ms ($decided/48 decided)"
    fi
    down
done
fi

# 3. AutoBench on each pair with each decision model.
if want 3; then
for pair in $PAIRS; do
    read -r small large <<< "$(echo "$pair" | tr ',' ' ')"
    flags=(--ctx-size "$CTX")
    [ "$pair" != "${PAIRS%% *}" ] && flags=(--ctx-size 4096 --decision-ctx 4096)
    n=0
    for decision in "${DECISIONS[@]}"; do
        n=$((n + 1))
        tag="$small-$large-$n"
        routed "$OUT/3-$tag.log" "$small" "$large" "$decision" "${flags[@]}" || exit 1
        warmed "$OUT/3-$tag.log"
        python3 "$BENCH" \
            --url "$URL" --small "$small" --large "$large" ${EMBED:+--embed-model "$EMBED"} \
            --sets "$SETS" --limit "$LIMIT" --answers "$OUT/3-$small-$large.answers.jsonl" \
            --out "$OUT/3-$tag.json" --details "$OUT/3-$tag.jsonl" > "$OUT/3-$tag.md" 2> "$OUT/3-$tag.err"
        status=$?
        down
        reflex=$(grep -c "| reflex" "$OUT/3-$tag.md")
        # The lowest share of answers identical to the chosen model's alone,
        # over the sets and concurrencies of the end-to-end stage.
        same=$(python3 -c 'import json,sys
rows=json.load(open(sys.argv[1], encoding="utf-8")).get("end_to_end",[])
v=[r["metrics"]["same_answer"] for r in rows if r["metrics"]["same_answer"] is not None]
print(f"{min(v):.3f}" if v else "none")' "$OUT/3-$tag.json" 2> /dev/null)
        what="3 autobench $small/$large with $(basename "$decision" .gguf)"
        if [ $status != 0 ] || [ "$reflex" = 0 ]; then
            result FAIL "$what: exit $status (see $OUT/3-$tag.err)"
        elif [ "$same" = none ] || awk "BEGIN{exit !(${same:-0} < 0.99)}"; then
            result FAIL "$what: end to end, as low as $same of answers the same as the model's alone (99% expected): $OUT/3-$tag.md"
        else
            result PASS "$what: end to end, at least $same of answers the same as the model's alone; $OUT/3-$tag.md"
        fi
    done
done
fi

# 4. Soak: mixed traffic at concurrency 8, VRAM sampled every 5 s.
# The pair has to sit side by side beside the reserved embedder, or the hour
# measures the swaps: on 4 October, at --ctx-size 8192 on 16 GB, the two
# chat models took turns for the whole of it. The largest context of
# SOAK_CTX at which the warm-up keeps both.
if want 4; then
soak_ctx=
for ctx in ${SOAK_CTX:-$CTX 6144 4096 2048}; do
    routed "$OUT/4.log" "$SMALL" "$LARGE" "${DECISIONS[0]}" --ctx-size "$ctx" \
        ${EMBED:+--embedding-model "$EMBED"} || exit 1
    warmed "$OUT/4.log"
    if [ "$(side_by_side "$OUT/4.log")" = yes ]; then
        soak_ctx=$ctx
        break
    fi
    down
done
if [ -z "$soak_ctx" ]; then
    result FAIL "4 soak: $SMALL and $LARGE do not sit side by side beside ${EMBED:-nothing} at any context of ${SOAK_CTX:-$CTX 6144 4096 2048}: $(side_by_side "$OUT/4.log")"
else
# The server's own count of models unloaded, before and after: VRAM that moves
# while models come and go is churn, VRAM that grows with none is a leak.
churn() {
    curl -s "$URL/api/version" | python3 -c 'import json,sys
d=json.load(sys.stdin)
print(d.get("model_swaps", 0), d.get("generation_evictions", 0))'
}
read -r swaps0 evictions0 <<< "$(churn)"
python3 - "$URL" "$DURATION" "$SMALL" "$LARGE" "$EMBED" "$OUT/4-vram.csv" > "$OUT/4.json" << 'PY'
import json, subprocess, sys, threading, time, urllib.request
url, duration, small, large, embed = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
series = sys.argv[6]
chat = [{"role": "user", "content": "Give one tip for a tidy desk."}]
kinds = [
    ("chat", "/api/chat", {"model": small, "messages": chat, "stream": True, "options": {"num_predict": 48}}),
    ("chat.completions", "/v1/chat/completions", {"model": large, "messages": chat, "max_tokens": 48}),
    ("auto", "/api/chat", {"model": "auto", "messages": chat, "stream": False, "options": {"num_predict": 48}}),
    ("systemone", "/v1/systemone", {"state": "A customer asks when the shop opens on Sunday.",
        "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?",
            "criteria": {"hours": "Opening hours", "refund": "A refund", "other": "Something else"}}}}),
]
if embed:
    kinds.append(("embed", "/api/embed", {"model": embed, "input": ["a lighthouse", "a harbour"]}))
done = {k[0]: 0 for k in kinds}
failed = []
by_kind = {}
vram = []
started = time.time()
stop = started + duration
lock = threading.Lock()
def kind_of(name, e):
    text = str(e)
    for k in ("Connection refused", "queue full", "timed out"):
        if k in text:
            return f"{name}: {k}"
    try:
        text = json.loads(text)["error"]
    except (ValueError, KeyError, TypeError):
        pass
    return f"{name}: {text.split(':')[0][:60]}"
def worker(w):
    i = w
    while time.time() < stop:
        name, path, body = kinds[i % len(kinds)]
        i += 1
        req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                text = r.read().decode()
            if '"error"' in text:
                raise RuntimeError(text[:200])
            with lock:
                done[name] += 1
        except Exception as e:
            with lock:
                failed.append(f"{name}: {e}"[:300])
                k, at = kind_of(name, e), round(time.time() - started)
                n, first, _ = by_kind.get(k, (0, at, at))
                by_kind[k] = (n + 1, first, at)
            # A server that fails at once would otherwise be asked thousands
            # of times a second: on 5 October a stopped one made 6 million.
            time.sleep(1)
def sample():
    while time.time() < stop:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.split()
            if out:
                vram.append((round(time.time() - started), int(out[0])))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        time.sleep(5)
threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)] + [threading.Thread(target=sample)]
for t in threads:
    t.start()
for t in threads:
    t.join()
with open(series, "w") as f:
    f.write("seconds,mib\n" + "".join(f"{s},{m}\n" for s, m in vram))
settled = vram[12:] or vram
low = min(settled, key=lambda v: v[1]) if settled else None
high = max(settled, key=lambda v: v[1]) if settled else None
def minute(samples):
    return round(sum(m for _, m in samples) / len(samples)) if samples else None
print(json.dumps({"done": done, "failed": failed[:20], "failures": len(failed), "by_kind": by_kind,
                  "vram_min": low[1] if low else None, "vram_min_at": low[0] if low else None,
                  "vram_max": high[1] if high else None, "vram_max_at": high[0] if high else None,
                  "vram_second_minute": minute(settled[:12]), "vram_last_minute": minute(vram[-12:]),
                  "samples": len(vram)}))
PY
read -r swaps1 evictions1 <<< "$(churn)"
down
soak=$(python3 - "$OUT/4.json" "$OUT/audit-4/audit.jsonl" "$OUT/4.log" \
    "$((swaps1 - swaps0))" "$((evictions1 - evictions0))" << 'PY'
import json, re, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
swaps, evictions = int(sys.argv[4]), int(sys.argv[5])
done = report["done"]
problems = []
if report["failures"]:
    kinds = sorted(report.get("by_kind", {}).items(), key=lambda kv: -kv[1][0])
    problems.append(f"{report['failures']} failed requests: " + "; ".join(
        f"'{k}' {n} times, from {a} s to {b} s" for k, (n, a, b) in kinds[:6]))
lo, hi = report["vram_min"], report["vram_max"]
if lo is not None and hi > lo * 1.05:
    problems.append(
        f"VRAM moved between {lo} MiB ({report['vram_min_at']} s) and {hi} MiB ({report['vram_max_at']} s) "
        f"after the first minute, {report['vram_second_minute']} MiB in the second minute and "
        f"{report['vram_last_minute']} in the last, with {swaps} models unloaded meanwhile")
elif swaps:
    problems.append(f"{swaps} models unloaded while every model fit, {evictions} of them to make room")
log = re.sub(r"\x1b\[[0-9;]*m", "", open(sys.argv[3], encoding="utf-8", errors="replace").read())
if re.search(r"GGML_ASSERT|out of memory|failed to allocate", log, re.I):
    problems.append("an out-of-memory or GGML_ASSERT in the log")
counts, bad = {}, 0
try:
    for line in open(sys.argv[2], encoding="utf-8", errors="replace"):
        try:
            entry = json.loads(line)
        except ValueError:
            bad += 1
            continue
        counts[entry["request_type"]] = counts.get(entry["request_type"], 0) + 1
except OSError:
    problems.append("no audit trail")
if bad:
    problems.append(f"{bad} audit lines do not parse")
# Each routed request writes a route line and a chat line; a request
# answered while the soak ended may not be counted on the client side.
want = {"chat": done["chat"] + done["auto"], "chat.completions": done["chat.completions"],
        "route": done["auto"], "systemone": done["systemone"]}
for kind, n in want.items():
    if not n <= counts.get(kind, 0) <= n + 8:
        problems.append(f"{counts.get(kind, 0)} '{kind}' audit lines for {n} requests")
vram = "" if lo is None else f", VRAM {lo}-{hi} MiB over {report['samples']} samples (4-vram.csv)"
summary = ", ".join(f"{n} {k}" for k, n in done.items()) + vram
print(("FAIL " + "; ".join(problems) + f" ({summary})") if problems else f"PASS {summary}")
PY
)
result "${soak%% *}" "4 soak, $DURATION s at concurrency 8, --ctx-size $soak_ctx: ${soak#* }"
fi
fi

# 5. Router latency on a CPU.
if want 5 && [ -n "$CPU_DECISION" ]; then
    # The last check: the GPU stays hidden from here on.
    export CUDA_VISIBLE_DEVICES= OMP_WAIT_POLICY=PASSIVE
    routed "$OUT/5.log" "$CPU_SMALL" "$SMALL" "$(realpath "$CPU_DECISION")" --ctx-size 4096 || exit 1
    warmed "$OUT/5.log"
    router_latency 16 1 > "$OUT/5.txt"
    down
    read -r _ p50 p95 timeouts decided wall < "$OUT/5.txt"
    if [ "$((decided + timeouts))" -gt 0 ]; then
        result PASS "5 router on a CPU ($CPU_SMALL/$SMALL, $(basename "$CPU_DECISION" .gguf)): decision p50 $p50 ms, p95 $p95 ms, $timeouts/16 past the 1000 ms timeout, wall p50 $wall ms"
    else
        result FAIL "5 router on a CPU: no route answered (see $OUT/5.log)"
    fi
fi

echo
echo "$PASS passed, $FAIL failed. Logs and the summary are in $OUT."
[ "$FAIL" = 0 ]
