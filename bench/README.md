# bench/ — Stress Test & Parallelism Verification

Real stress test that **proves** whether an inference server processes requests in parallel or just queues them sequentially.

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
