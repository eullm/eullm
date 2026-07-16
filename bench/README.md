# bench/ — Stress Test & Parallelism Verification

Real stress test that **proves** whether an inference server processes requests in parallel or just queues them sequentially.

## `reuse_validation.py` — roadmap 0.7-A real-hardware checklist

Validates the KV-cache prefix reuse scheduler against the checklist in
[`docs/roadmap-engine-0.7-1.0.md`](../docs/roadmap-engine-0.7-1.0.md) § 0.7-A:
a 20-turn growing-history conversation, 8 concurrent conversations, abrupt
mid-stream disconnects, a slow-consuming client (the v0.6.20 `Full`-vs-`Closed`
channel regression test), and byte-identical output at a fixed seed.

Start the server headless first — note this is `eullm run <model>`, not
`eullm serve` (which starts the API with no model loaded and expects a
`model` field per-request instead). `--no-ui` skips the browser/chat-UI
auto-open, and `< /dev/null` keeps stdin a non-tty so it doesn't drop into
the interactive REPL when backgrounded. Bump `--ctx-size` for a 20-turn
growing conversation plus `--batch-size` concurrent slots — the default
4096 is split across all slots (`ctx_size / batch_size`) and fills up fast:

```bash
bin/eullm run <model-id-or-path> --no-ui --batch-size 8 --ctx-size 16384 \
    < /dev/null > server.log 2>&1 &
```

Then:

```bash
pip install aiohttp

python bench/reuse_validation.py \
    --url http://localhost:11434 \
    --model <same-model-id-or-path> \
    --server-log server.log
```

Run a subset with `--tests multiturn,slow-consumer` (see `--help` for every
flag: turn/concurrency counts, token budgets, timeouts). Exit code is nonzero
if any test fails. Pass `--baseline-url` to diff the determinism test's output
against a second server (e.g. an old binary on another port) for a true A/B
instead of a self-comparison.

This exercises the same scheduler code path the `--cli` REPL uses (both
resend the full growing history and share the scheduler), so it doubles as
an automated stand-in for the 20-turn CLI conversation check — driving the
REPL by hand and grepping its log for `reused N from cache` remains a useful
manual cross-check but isn't required to run this suite.

## What it measures

| Metric | What it proves |
|--------|---------------|
| **TTFT** (Time To First Token) | Do all requests start generating immediately, or do later ones wait? |
| **Timeline overlap** | Are generation periods overlapping in time? |
| **Token interleaving** | Do tokens from different requests arrive interleaved (true batching) or in sequence? |
| **Throughput** | Total tok/s across all concurrent requests |
| **Latency distribution** | P50, P95, P99 for TTFT and total latency |

## Quick start

```bash
# Install dependency
pip install aiohttp

# Test EULLM
python bench/stress_test.py \
    --url http://localhost:11434 \
    --model Qwen3.5-9B-Q4_K_M \
    --concurrency 1,2,4,8 \
    --tokens 100 \
    --warmup

# Test Ollama
python bench/stress_test.py \
    --url http://localhost:11435 \
    --model qwen3.5:9b \
    --concurrency 1,2,4,8 \
    --tokens 100 \
    --warmup

# Compare both (requires both servers running)
./bench/compare.sh Qwen3.5-9B-Q4_K_M qwen3.5:9b
```

## How to read the output

### Timeline

```
  req 1 |....############################################  | TTFT   120ms  94.2 tok/s  100 tokens
  req 2 |....############################################  | TTFT   125ms  93.8 tok/s  100 tokens
  req 3 |....#############################################| TTFT   130ms  93.1 tok/s  100 tokens
  req 4 |.....############################################| TTFT   135ms  92.5 tok/s  100 tokens
```

- `.` = prefill/waiting (submit → first token)
- `#` = generating tokens (first token → last token)
- If `#` bars overlap vertically → **real parallel processing**
- If `#` bars are sequential (no vertical overlap) → **queued processing**

### Parallelism verdict

```
  PARALLELISM ANALYSIS (EULLM):
    Overlap:      YES — 6/6 pairs overlap (100%)
    Interleaving: YES — 285 context switches (72% transition rate)
    VERDICT:      REAL PARALLEL PROCESSING
```

vs

```
  PARALLELISM ANALYSIS (Ollama):
    Overlap:      NO  — requests appear sequential
    Interleaving: NO  — tokens arrive in sequence, not interleaved
    VERDICT:      SEQUENTIAL PROCESSING (no real parallelism)
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | (required) | Server base URL |
| `--model` | (required) | Model name |
| `--label` | auto | Label for output |
| `--concurrency` | `1,2,4,8` | Comma-separated concurrency levels |
| `--tokens` | `100` | Tokens to generate per request |
| `--rounds` | `1` | Rounds per concurrency level (for averaging) |
| `--warmup` | off | Send a warmup request first |
| `--json` | none | Write JSON results to file |

## JSON output

Use `--json results.json` to get machine-readable results for further analysis:

```json
{
  "label": "EULLM",
  "model": "Qwen3.5-9B-Q4_K_M",
  "tokens_per_request": 100,
  "results": [
    {
      "concurrency": 4,
      "wall_ms": 3300,
      "throughput": 121.2,
      "overlap": {"is_parallel": true, "overlap_ratio": 1.0},
      "interleaving": {"is_interleaved": true, "transition_rate": 0.72},
      "requests": [...]
    }
  ]
}
```

## Why `bench.sh` was not enough

The old `bench.sh` fired N curl requests with `"stream": false` and measured only wall time. This doesn't prove parallelism — a fast sequential processor could achieve similar wall times. The new stress test uses **streaming** to track individual token arrival timestamps, enabling definitive proof of parallel vs sequential processing.
