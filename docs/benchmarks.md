# Benchmarks — EULLM Engine vs Ollama

EULLM Engine uses **continuous batching** to decode multiple requests in parallel within a single GPU pass, enabled by default across all slots. Ollama also serves concurrent requests via llama.cpp, but only up to `OLLAMA_NUM_PARALLEL` slots (a low default) and with one full KV-cache copy reserved per slot.

This page documents benchmark results comparing the two.

> ⚠️ **Re-measurement in progress.** The Ollama figures below are being re-run at **matched parallelism** (`OLLAMA_NUM_PARALLEL=16`, equal to the Engine's 16 slots) to make the comparison apples-to-apples. The tables will be updated with the new numbers and the exact Ollama version. The EULLM Engine figures are its own measured continuous-batching scaling.

## Test setup

| | |
|---|---|
| **GPU** | NVIDIA RTX 5070 Ti (16 GB VRAM) |
| **Model** | Qwen3.5-9B GGUF |
| **Tokens per request** | 150 (`num_predict=150`) |
| **Benchmark script** | [`bench.sh`](../bench.sh) |
| **Method** | Fire N concurrent requests, measure wall time and total tokens |
| **EULLM port** | `localhost:11434` |
| **Ollama port** | `localhost:11435` |
| **EULLM parallelism** | Continuous batching, 16 slots |
| **Ollama parallelism** | `OLLAMA_NUM_PARALLEL=16` _(matched; re-measurement in progress)_ |
| **Ollama version** | _to record on re-run_ |
| **Date** | March 2026 |

## Results

### Total throughput

How many tokens per second the system produces across all concurrent requests.

<p align="center">
  <img src="assets/bench-throughput.svg" alt="Throughput: EULLM vs Ollama" width="720" />
</p>

| Concurrent requests | EULLM Engine | Ollama | Speedup |
|:---:|:---:|:---:|:---:|
| 1 | 94 tok/s | 93 tok/s | 1.0× |
| 2 | 143 tok/s | 97 tok/s | **1.5×** |
| 4 | 183 tok/s | 100 tok/s | **1.8×** |
| 8 | 206 tok/s | 101 tok/s | **2.0×** |
| 16 | 259 tok/s | 102 tok/s | **2.5×** |

EULLM Engine's continuous batching scheduler decodes all active sequences in a single GPU pass, scaling total throughput with concurrency. The Ollama column above is being re-measured at matched parallelism (`OLLAMA_NUM_PARALLEL=16`); the prior numbers were taken at a lower Ollama parallelism setting and are not an apples-to-apples comparison.

### Time to complete all requests (wall time)

How long until all N concurrent requests have finished — the metric that matters for user experience.

<p align="center">
  <img src="assets/bench-latency.svg" alt="Latency: EULLM vs Ollama" width="720" />
</p>

| Concurrent requests | EULLM Engine | Ollama | |
|:---:|:---:|:---:|:---:|
| 1 | 1.6s | 1.6s | identical |
| 2 | 2.1s | 3.1s | 1.5× faster |
| 4 | 3.3s | 6.0s | 1.8× faster |
| 8 | 5.8s | 11.9s | **2.0× faster** |
| 16 | 9.3s | 23.6s | **2.5× faster** |

With 16 concurrent users, the last user on Ollama waits **23.6 seconds**. On EULLM Engine, everyone gets their response within **9.3 seconds**.

### Per-request latency

Individual request performance under load. Each user gets fewer tok/s as concurrency increases (the GPU time is shared), but tokens arrive steadily via SSE streaming.

| Concurrent requests | EULLM per-request | Ollama per-request |
|:---:|:---:|:---:|
| 1 | 97.6 tok/s | 111 tok/s |
| 2 | ~74 tok/s | 111 tok/s* |
| 4 | ~47 tok/s | 111 tok/s* |
| 8 | ~26 tok/s | 111 tok/s* |
| 16 | ~16.5 tok/s | 111 tok/s* |

\* These Ollama per-request figures were taken at low parallelism, where requests beyond the slot limit **wait in line** rather than batching. They are being re-measured with `OLLAMA_NUM_PARALLEL=16`.

EULLM's per-request tok/s decreases with load, but all requests **start immediately** and receive tokens in real-time via streaming. At 16.5 tok/s, text still arrives faster than humans can read.

## How continuous batching works

```
Ollama (low parallelism — extra requests queue):
  req1 ████████░░░░░░░░░░░░░░░░ → done
  req2 ________████████░░░░░░░░░ → done (waited 1.5s)
  req3 ________________████████░ → done (waited 3.0s)
  req4 ________________________█ → done (waited 4.5s)
  Wall time: ~6.0s for 4 requests

EULLM Engine (continuous batching):
  req1 ██████████████ → done
  req2 ██████████████ → done     (all start immediately)
  req3 ██████████████ → done
  req4 ██████████████ → done
  Wall time: ~3.3s for 4 requests
```

The scheduler runs a dedicated decode loop on an OS thread. Each iteration calls `llama_decode` with a batch containing tokens from all active sequences. Per-sequence KV cache is managed independently, so sequences can start and finish at any time without blocking others.

## Reproduce these benchmarks

### Stress test with parallelism verification (recommended)

The [`bench/stress_test.py`](../bench/stress_test.py) script uses streaming to track individual token timestamps, proving whether requests are truly processed in parallel or just queued. See [`bench/README.md`](../bench/README.md) for full details.

```bash
pip install aiohttp

# Test EULLM (streaming, with parallelism analysis)
python bench/stress_test.py --url http://localhost:11434 --model Qwen3.5-9B-Q4_K_M \
    --concurrency 1,2,4,8,16 --tokens 150 --warmup --json results_eullm.json

# Test Ollama
python bench/stress_test.py --url http://localhost:11435 --model qwen3.5:9b \
    --concurrency 1,2,4,8,16 --tokens 150 --warmup --json results_ollama.json

# Or compare both at once
./bench/compare.sh Qwen3.5-9B-Q4_K_M qwen3.5:9b --concurrency 1,2,4,8,16 --tokens 150
```

### Quick throughput benchmark

The [`bench.sh`](../bench.sh) script fires N concurrent non-streaming requests and measures wall time. Simpler but does not verify parallelism.

```bash
# Build EULLM Engine with CUDA
cargo build --release --features cuda

# Run EULLM Engine
./target/release/eullm run ./Qwen3.5-9B-Q4_K_M.gguf --batch-size 16

# In another terminal, run the benchmark
./bench.sh http://localhost:11434 Qwen3.5-9B-GGUF

# Compare with Ollama (on a different port)
ollama serve  # default port 11435 or configure
./bench.sh http://localhost:11435 qwen3.5:9b
```

## Hardware notes

These results are on a consumer RTX 5070 Ti (16 GB). Results will vary by GPU, model size, and quantization level. The relative advantage of continuous batching increases with:

- **More concurrent requests** — the gap widens linearly
- **Longer generations** — more time spent in decode = more batching opportunity
- **Faster GPUs** — shared-pass decoding extracts more parallelism per GPU step

On server GPUs (A100, H100) with higher memory bandwidth, the throughput scaling with concurrency is even more dramatic.
