# Benchmarks — EULLM Engine continuous batching

EULLM Engine uses **continuous batching** to decode multiple requests in parallel within a single GPU pass, enabled by default across all slots. The scheduler runs a dedicated decode loop on an OS thread; each iteration calls `llama_decode` with a batch containing tokens from every active sequence, so requests start receiving tokens immediately instead of queueing for a slot.

This page documents how that scaling behaves on a consumer GPU.

## Test setup

| | |
|---|---|
| **GPU** | NVIDIA RTX 5070 Ti (16 GB VRAM) |
| **Model** | Qwen3.5-9B GGUF |
| **Tokens per request** | 150 (`num_predict=150`) |
| **Benchmark script** | [`bench.sh`](../bench.sh) |
| **Method** | Fire N concurrent requests, measure wall time and total tokens |
| **Engine port** | `localhost:11434` |
| **Engine parallelism** | Continuous batching, 16 slots, shared KV cache pool |
| **Date** | March 2026 |

## Results — Engine scaling

### Total throughput

How many tokens per second the system produces across all concurrent requests.

<p align="center">
  <img src="assets/bench-throughput.svg" alt="Engine throughput scaling 1→16 concurrent" width="720" />
</p>

| Concurrent requests | Engine throughput | Speedup vs N=1 |
|:---:|:---:|:---:|
| 1 | 94 tok/s | 1.0× |
| 2 | 143 tok/s | **1.5×** |
| 4 | 183 tok/s | **1.9×** |
| 8 | 206 tok/s | **2.2×** |
| 16 | **259 tok/s** | **2.75×** |

Continuous batching extracts more throughput per GPU step as concurrency rises because the same forward pass amortises across multiple sequences.

### Time to complete all requests (wall time)

How long until every concurrent request has finished — the metric that matters for user experience.

<p align="center">
  <img src="assets/bench-latency.svg" alt="Engine wall time vs concurrency" width="720" />
</p>

| Concurrent requests | Engine wall time (16×150 tok) |
|:---:|:---:|
| 1 | 1.6 s |
| 2 | 2.1 s |
| 4 | 3.3 s |
| 8 | 5.8 s |
| 16 | **9.3 s** |

With 16 concurrent users, the slowest of the 16 responses still finishes within ~9 seconds, and every user starts receiving tokens via SSE streaming immediately — no slot queueing.

### Per-request latency

Individual request performance under load. Each user gets fewer tok/s as concurrency increases (the GPU time is shared between active sequences), but tokens arrive steadily via SSE streaming.

| Concurrent requests | Engine per-request |
|:---:|:---:|
| 1 | 97.6 tok/s |
| 2 | ~71 tok/s |
| 4 | ~46 tok/s |
| 8 | ~26 tok/s |
| 16 | ~16.5 tok/s |

At ~16 tok/s text still arrives faster than humans can read; the trade-off chosen by continuous batching is "everyone gets tokens in real-time" rather than "one user at full speed, the rest queued."

## Reproduce

### Stress test with parallelism verification (recommended)

The [`bench/stress_test.py`](../bench/stress_test.py) script uses streaming to track individual token timestamps, proving whether requests are truly processed in parallel or just queued. See [`bench/README.md`](../bench/README.md) for full details.

```bash
pip install aiohttp

python bench/stress_test.py --url http://localhost:11434 --model Qwen3.5-9B-Q4_K_M \
    --concurrency 1,2,4,8,16 --tokens 150 --warmup --json results_eullm.json
```

### Quick throughput benchmark

The [`bench.sh`](../bench.sh) script fires N concurrent non-streaming requests and measures wall time. Simpler but does not verify parallelism.

```bash
# Build Engine with CUDA
cargo build --release --features cuda

# Run Engine
./target/release/eullm run ./Qwen3.5-9B-Q4_K_M.gguf --batch-size 16

# In another terminal, run the benchmark
./bench.sh http://localhost:11434 Qwen3.5-9B-GGUF
```

## Hardware notes

These results are on a consumer RTX 5070 Ti (16 GB). Results will vary by GPU, model size, and quantization level. The relative gain of continuous batching grows with:

- **More concurrent requests** — the speedup vs N=1 widens
- **Longer generations** — more time spent in decode = more batching opportunity
- **Faster GPUs** — shared-pass decoding extracts more parallelism per GPU step

On server GPUs (A100, H100) with higher memory bandwidth, the throughput scaling with concurrency is more dramatic still.
