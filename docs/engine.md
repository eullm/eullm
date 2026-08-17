# EULLM Engine

The EULLM Engine is a CLI + API server for running GGUF models locally, with real llama.cpp inference, built-in EU model catalog, local-only AI Act audit trail, and no network telemetry of any kind. Single Rust binary — no Python, no Docker.

## Installation

### From source

```bash
cd engine

# CPU only
cargo build --release

# With GPU acceleration
cargo build --release --features cuda     # NVIDIA (CUDA)
cargo build --release --features rocm     # AMD (ROCm)
cargo build --release --features vulkan   # Cross-platform (NVIDIA + AMD + Intel)
cargo build --release --features metal    # macOS Apple Silicon

# Binary will be at target/release/eullm
```

#### Build requirements

- Rust 1.75+
- C/C++ compiler (gcc/clang) — needed by llama.cpp
- CMake 3.14+
- libclang (`libclang-dev` on Debian/Ubuntu, `clang-devel` on Fedora) — needed by `bindgen` for FFI bindings
- (Optional) CUDA toolkit, ROCm, Vulkan SDK, or Xcode for GPU support

**Ubuntu/Debian one-liner:** `sudo apt install build-essential cmake libclang-dev`

### Docker

```bash
# CPU only
docker build -t eullm-engine engine/
docker run -p 11434:11434 -v eullm-models:/models eullm-engine

# With NVIDIA GPU
docker build -t eullm-engine --build-arg FEATURES=cuda engine/
docker run --gpus all -p 11434:11434 -v eullm-models:/models eullm-engine

# Or via docker compose (from repo root)
docker compose up engine              # CPU
docker compose --profile gpu up engine-gpu   # GPU
```

## CLI Commands

### `eullm run <model> [--port PORT]`

Load a model and start the API server. Supports local GGUF files and catalog models.

```bash
# Run a local GGUF file (inference works immediately)
eullm run ./qwen3-7b-q4_k_m.gguf

# Run a catalog model (auto-downloads from HuggingFace)
eullm run legal-it-7b

# With options
eullm run ./model.gguf --port 8080
eullm run ./model.gguf --gpu-layers 0      # CPU only
eullm run ./model.gguf --gpu-layers 20     # At most 20 layers on the GPU
eullm run ./model.gguf --ctx-size 8192     # Larger context window
eullm run ./model.gguf --threads 8         # Limit CPU threads
```

**Options** (`eullm run --help` is authoritative; this table is the short form):

| Option | Default | Description |
|---|---|---|
| `model` | (optional) | GGUF path, catalog id, URL, or `hf.co/<owner>/<repo>[:<quant>]`. Omitted on a terminal, opens the model picker |
| `--port, -p` | `11434` | API server port |
| `--ui-port` | `11435` | Embedded chat UI port (separate from the API) |
| `--gpu-layers` | automatic | **Upper bound** on GPU layers (-1 = all, 0 = CPU only). Sizing may offload fewer — see below |
| `--no-fit` | false | Disable automatic sizing; use `--gpu-layers` as given |
| `--fit` | (implied) | Same sizing, plus an interactive confirmation before a partial split |
| `--fit-strict` | false | Refuse to load rather than offload a partial split |
| `--cpu-moe` | false | MoE models: all expert tensors on CPU RAM (sized automatically when unset) |
| `--n-cpu-moe` | `0` | MoE models: expert tensors on CPU for the first N layers only |
| `--ctx-size, -c` | `4096` | Total context window (split across batch slots) |
| `--threads, -t` | all CPUs | Number of CPU threads |
| `--batch-size` | `1` | Concurrent requests served by the batching scheduler (raise `--ctx-size` with it) |
| `--n-batch` | `2048` | Prefill batch size (tokens per eval) |
| `--cache-type-k` | `f16` | KV cache type for keys (f16, q8_0, q4_0). Quantizing frees VRAM for more layers |
| `--cache-type-v` | `f16` | KV cache type for values (f16, q8_0, q4_0) |
| `--no-flash-attn` | false | Disable flash attention (on by default) |
| `--web` | false | Fetch URLs found in user messages and inject their content |
| `--mmproj` | (auto) | Multimodal projector path, when it is not beside the weights |
| `--ctx-checkpoints` | `0` | Prompt-prefix state snapshots for hybrid/recurrent models |
| `--checkpoint-min-step` | `8192` | Minimum new tokens between checkpoints |
| `--rs-seq` | `0` | Recurrent-state rollback window — leave off unless you know why |
| `--rust-debug` | false | Per-token NaN/Inf scan of the logits (diagnostics) |
| `--replace` | false | Replace an existing service on the port |
| `--daemon` | false | Run as a background daemon |
| `--pidfile` | `/tmp/eullm.pid` | PID file path (with `--daemon`) |
| `--logfile` | `~/.eullm/logs/eullm.log` | Daemon log file (with `--daemon`). Set `--pidfile` alone and the log stays beside it |
| `--keep-alive` | (unset) | Idle-unload a model this many seconds/minutes/hours after its last use (e.g. `5m`). Unset = never automatic; a request's own `keep_alive` field overrides it for that load. Applies to the generation model and the embedding model independently |

#### Automatic GPU sizing

Since 0.6.80 the engine decides how much of the model goes on the GPU, on
every load — at startup and on every model swap, on `run` and on `serve`
alike. It reads free VRAM and the model's own metadata, then charges each
layer its share of the weights plus its KV-cache slice for the context and
cache type in use, and offloads as many layers as that budget allows. MoE
models get their expert tensors moved to CPU RAM first, since only a few
experts fire per token: that buys far more headroom than whole-layer
offload can.

You do not need a flag for this. It exists because the alternative default
is worse: a model larger than free VRAM used to die with an out-of-memory
error at load, while sized, the worst case is a slower partial split.

| You want | Use |
|---|---|
| The engine to decide | nothing — this is the default |
| To cap how much of the card is used | `--gpu-layers N` (an upper bound; sizing may go lower) |
| CPU only | `--gpu-layers 0` |
| To force a count past the estimate | `--no-fit --gpu-layers N` |
| To be asked before a partial split | `--fit` (interactive terminals only) |
| To refuse a load that doesn't fully fit | `--fit-strict` |

Three properties worth knowing.

`--gpu-layers` is a ceiling rather than a fixed count on purpose: a number
chosen against one model is not a fact about the next one the process loads,
and applying it blindly to a swapped-in model is exactly how an
out-of-memory error happens.

Automatic sizing never asks questions — it applies the split and logs one
line naming the flags that override it — because a default that interrupts
every launch is its own kind of failure. Where free VRAM cannot be probed
(any non-CUDA build) it stays silent and `--gpu-layers` is used as-is.

The budget leaves headroom on purpose, and it is the same headroom the
loader requires: enough of the card's total memory must stay free for the
context and its compute buffers, or the weights load and then no context can
be allocated at all. Sizing that aims past that floor produces a split the
loader refuses, which is a worse failure than a conservative one — so the
sizer takes the stricter of its own fragmentation margin and the loader's
minimum. Expect a loaded model to leave roughly 12% of the card free; that
is not waste, it is what the next token's compute buffer is allocated from.

### `eullm pull <model>`

Download a model from HuggingFace (or the EU registry when available).

```bash
eullm pull legal-it-7b
eullm pull eullm/legal-it-7b     # Full name works too
```

The model is stored in `~/.eullm/models/<model>/` with a GGUF file and manifest.

### `eullm list`

Show locally downloaded models. If none are available, displays the EU catalog.

```bash
eullm list
```

### `eullm show <model>`

Display detailed information about a model (local or from catalog).

```bash
eullm show legal-it-7b
```

### `eullm serve [--port PORT]`

Start the API server without loading any model. The first API request with a `"model"` field will load that model dynamically.

```bash
eullm serve
eullm serve --port 8080
```

`serve` takes the same runtime flags as `run` (they are one shared set), and
they apply to every model it loads or swaps to: automatic GPU sizing runs on
each load, `--gpu-layers` caps it, `--no-fit` disables it. The one difference
is that `serve` never prompts — a daemon has nobody at the keyboard — so
`--fit` adds nothing there and `--fit-strict` surfaces a refused load as an
error to the API caller.

### `eullm import-ollama <model> [--ollama-dir PATH]`

Import a model from a local Ollama installation into EULLM's model store. Copies the GGUF blob so you can benchmark both engines with the exact same model file.

```bash
# Import from Ollama (tag "latest")
eullm import-ollama llama3.2

# Import a specific tag
eullm import-ollama qwen3:14b

# Custom Ollama directory
eullm import-ollama gemma3 --ollama-dir /custom/path
```

**How it works:**

1. Reads the Ollama manifest at `~/.ollama/models/manifests/registry.ollama.ai/library/{name}/{tag}`
2. Locates the model layer (`application/vnd.ollama.image.model`) — the GGUF blob
3. Copies the blob to `~/.eullm/models/{name}/{name}.gguf`
4. Applies GGUF metadata patches if needed (e.g. fixes array lengths for llama.cpp compatibility)
5. Writes a EULLM manifest so the model appears in `eullm list`

After import:

```bash
eullm run llama3.2        # Runs on EULLM Engine
ollama run llama3.2       # Same model on Ollama — identical comparison
```

**Licensing note:** Ollama does not add any license on top of the original model weights. The GGUF blob is the same file distributed by the upstream model author. The license of the model itself applies (e.g. Apache 2.0 for Qwen3, MIT for DeepSeek).

**GGUF compatibility:** Some Ollama GGUF files contain metadata arrays with fewer elements than upstream llama.cpp expects (e.g. `qwen35.rope.dimension_sections` with 3 elements instead of 4). The import command automatically patches these during copy. Models with hybrid architectures (e.g. Qwen3.5 with SSM/Mamba2 layers) may have incompatible tensor layouts — use the HuggingFace GGUF instead.

### `eullm forge`

Delegate to the EULLM Forge Python pipeline for model verticalizzazione.

```bash
eullm forge Qwen/Qwen3-14B --profile legal-it --identity "LegalAI"
```

## Dynamic Model Swap

EULLM Engine can swap models at runtime, like Ollama. When an API request specifies a `"model"` that differs from the currently loaded one, the server automatically unloads the current model and loads the new one.

```bash
# Start with one model
eullm run qwen3-14b

# Any API request with a different model triggers a swap
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen3-7b", "prompt": "Ciao"}'
# → Unloads qwen3-14b, loads qwen3-7b, then responds

# Or start with no model and load on first request
eullm serve
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen3-14b", "prompt": "Ciao"}'
# → Loads qwen3-14b on the fly
```

**Behavior:**

- In-flight requests on the old model complete normally (they hold cloned handles)
- The new model loads on a blocking thread, then atomically replaces the slot
- The model name must be an imported model (`eullm import-ollama`) or a local GGUF path

**What carries across a swap, and what does not.** The flags you launched
with are settings, and they apply to every model the process loads: context
size, cache types, batch size, thread count, and `--gpu-layers` as an upper
bound. What a *model* needs is decided per model, freshly, on each load:

| Per model, re-decided on every load | Why |
|---|---|
| How many layers go on the GPU | Sized against the VRAM actually free at that moment, for that model's own weights and KV cache |
| MoE expert offload | Only MoE models have experts to move |
| The multimodal projector | A projector belongs to the weights it was trained with; pairing it with another model fails the load |
| Sequential vs batched execution | Vision models need the sequential engine; text models get the scheduler back |

That split is not a detail: treating a per-model property as a process-wide
setting is how a launch model's projector ended up on its successors, and how
a layer count chosen for one model produced an out-of-memory error on the
next one.

## KV Cache Quantization

By default, EULLM uses F16 KV cache for maximum GPU compatibility. Quantized types save VRAM but may cause GPU compute fallback to CPU on some architectures — verify GPU utilisation with `nvtop` before deploying in production.

| Setting | VRAM for 14B @ 16K context |
|---------|---------------------------|
| **`--cache-type-k f16 --cache-type-v f16`** | **~10 GB (default, best GPU compat)** |
| `--cache-type-k q8_0 --cache-type-v q8_0` | ~5 GB |
| `--cache-type-k q8_0 --cache-type-v q4_0` | ~2.5 GB (⚠️ verify GPU usage) |

```bash
# Default (F16) — maximum GPU compatibility
eullm run qwen3-14b --ctx-size 8192

# Save VRAM with quantized KV cache (check GPU usage with nvtop!)
eullm run qwen3-14b --ctx-size 16384 --cache-type-k q8_0 --cache-type-v q4_0

# Aggressive quantization (minimum VRAM, may fall back to CPU)
eullm run qwen3-14b --ctx-size 32768 --cache-type-k q4_0 --cache-type-v q4_0
```

Available types: `f16`, `f32`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`.

> **Note:** Quantized V cache types (Q4_0, Q8_0) require Flash Attention. On GPUs where Flash Attention doesn't support these types, the engine automatically falls back to F16 KV cache and logs a warning. You can also set F16 explicitly by omitting the `--cache-type-v` flag.
>
> **TurboQuant note:** v0.5.x of the engine integrated an experimental TurboQuant (Walsh-Hadamard + Lloyd-Max) KV compression via the AmesianX/llama.cpp fork. It is **not in v0.5.8 onwards** — see [README → Research & Experiments](../README.md#research--experiments) and the archived numbers in [`turboquant-quality-report.md`](turboquant-quality-report.md) / [`turboquant-kv-stress-report.md`](turboquant-kv-stress-report.md).

## Constrained JSON Decoding (`format: "json"`)

When `format: "json"` is set in a request, EULLM uses GBNF grammar-based constrained decoding to guarantee valid JSON output. This matches Ollama's behavior and prevents malformed JSON in extraction pipelines.

```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "qwen3-14b",
    "prompt": "Extract the name and age from: John is 30 years old",
    "format": "json"
  }'
# → Always returns valid JSON: {"name": "John", "age": 30}
```

Works on all endpoints: `/api/generate`, `/api/chat`, `/v1/chat/completions`. Both sequential and continuous batching modes.

## Continuous Batching

EULLM's continuous batching scheduler decodes multiple requests in parallel on a single GPU pass. This is a key performance differentiator over Ollama, which processes requests one at a time.

```bash
# Enable continuous batching with 8 parallel slots (default)
eullm run ./model.gguf --batch-size 8

# More slots for high-throughput RAG workloads
eullm run ./model.gguf --batch-size 16

# Sequential mode (one request at a time, like Ollama)
eullm run ./model.gguf --batch-size 0
```

With 16 concurrent requests on a consumer GPU, EULLM achieves ~2.5x throughput vs Ollama. See [benchmarks](benchmarks.md) for details.

### Context window and batch slots

The `--ctx-size` flag sets the **total** KV cache budget, shared across all batch slots (matching Ollama/llama.cpp server behaviour). Each slot gets `ctx_size / batch_size` tokens of context:

```bash
# 16K total, 4 slots → 4096 tokens/slot
eullm run ./model.gguf --ctx-size 16384 --batch-size 4

# 32K total, 8 slots → 4096 tokens/slot
eullm run ./model.gguf --ctx-size 32768 --batch-size 8
```

### Choosing batch-size

More slots increase parallelism but reduce per-request throughput (shared GPU time). General guideline:

| Parallel slots | Per-request throughput | Aggregate throughput | Use case |
|:-:|:-:|:-:|---|
| 4 | High | High | Chat, general inference |
| 8 | Medium | Higher | Batch extraction, RAG pipelines |
| 16+ | Lower | Highest | High-concurrency APIs, multi-GPU |

Start with `--batch-size 4` for the best per-request latency. Increase when your workload requires more concurrent slots and can tolerate slower individual responses.

## Dynamic Model Swap

The Engine supports hot-swapping models at runtime. When a request specifies a different `model`, the server automatically:

1. **Shuts down** the old scheduler thread (waits for it to fully exit)
2. **Frees VRAM** — the old model, KV cache, and LlamaBackend are destroyed
3. **Loads** the new model with the requested configuration
4. **Resumes** serving requests on the new model

### Basic swap (via model field)

Any generation request with a different `model` triggers the swap:

```bash
# Currently running qwen3-14b — this switches to qwen3-8b automatically
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "prompt": "Hello"
}'
```

### Dynamic batch_size and ctx_size

The `batch_size` and `ctx_size` can be overridden per model swap. This is useful when switching between a large model (fewer slots) and a small model (more slots):

```bash
# Switch to 8B with 8 parallel slots and 32K context
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "batch_size": 8,
  "ctx_size": 32768,
  "prompt": "Hello"
}'

# Switch back to 14B with 4 slots and 16K context
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:14b",
  "batch_size": 4,
  "ctx_size": 16384,
  "prompt": "Hello"
}'
```

When `batch_size` or `ctx_size` are not specified, the values from `--batch-size` and `--ctx-size` at startup are used.

### Model name resolution

The `model` field accepts:

| Format | Example | Resolution |
|--------|---------|------------|
| Full GGUF path | `/models/qwen3-8b.gguf` | Direct file |
| Ollama-style name | `qwen3:8b` | Normalized to `qwen3-8b`, searched in `/models/` and model store |
| Path without extension | `/models/qwen3-8b` | Tries appending `.gguf` |
| Directory | `/models/mymodel/` | Picks the first `.gguf` file inside |
| Registered name | `legal-it-7b` | Looked up in `~/.eullm/models/` |

### Concurrent swap safety

Multiple requests arriving simultaneously for a different model are handled safely:
- Only one swap runs at a time (serialized via Mutex)
- Other requests wait for the swap to complete, then use the new model
- In-flight requests on the old model continue normally via reference counting

### VRAM budget reference

Approximate VRAM usage with F16 KV cache (default). Actual values depend on model architecture and GPU.

| Model size | batch_size | ctx_size | tok/slot | VRAM est. |
|:----------:|:----------:|:--------:|:--------:|:---------:|
| 14B Q4 | 4 | 16384 | 4096 | ~12.5 GB |
| 8B Q4 | 4 | 16384 | 4096 | ~7 GB |
| 8B Q4 | 8 | 32768 | 4096 | ~9.5 GB |
| 8B Q4 | 8 | 16384 | 2048 | ~7 GB |
| 70B Q4 | 16 | 65536 | 4096 | ~45 GB |

## Text Embeddings and the Embedding Slot

`POST /api/embed` (Ollama) and `POST /v1/embeddings` (OpenAI) embed one or
more texts with any GGUF text-embedding model — BGE, E5, and similar.
Naming a model in the request loads it the first time it is requested,
exactly like the generation model:

```bash
curl -s http://localhost:11434/api/embed -d '{
  "model": "bge-m3",
  "input": ["first chunk", "second chunk"]
}'
```

The embedding model lives in a **second, independent slot** — it does not
replace whatever generation model is loaded. Both are kept resident when
they fit; this is a decision the server makes for itself on every embedding
request, not something the caller has to reason about:

- If the embedding model fits in currently free VRAM alongside the loaded
  generation model, it is loaded next to it. Both stay resident.
- If it does not fit, the generation model is evicted first (it reloads
  automatically on the next generation request), and the embedder gets the
  whole card.
- The reverse also holds: a generation request with `--fit` enabled evicts
  a resident embedder first if the sizing needs the VRAM back.

On a build with no VRAM probe (non-CUDA), the server always tries to keep
both resident and lets a genuine out-of-memory surface as a normal load
error — the same posture `--fit` itself takes there.

How many times either direction of eviction has happened is in
`/api/version`'s `model_swaps` field — a rate of roughly one per request
means a card too small for both is being asked to alternate rather than
batch (do all ingestion, then all generation, rather than interleaving).

Pooling is read from the model's own GGUF metadata (CLS for BGE, mean for
E5, and so on) rather than guessed; a model that declares no pooling type
falls back to mean-pooling its per-token embeddings. Output vectors are
L2-normalized. Rerankers (RANK-pooling models) are out of scope for this
endpoint — normalizing a single relevance score would collapse it.

## API Reference

The Engine exposes two sets of endpoints: the native EULLM API (Ollama-compatible) and an OpenAI-compatible API. CORS is enabled for browser-based tools.

### EULLM API (Ollama-compatible)

#### `GET /api/version`

Returns the Engine version.

```bash
curl http://localhost:11434/api/version
```

```json
{
  "version": "0.1.0"
}
```

#### `GET /api/tags`

List available models. Returns the currently loaded model first (what admin dashboards check for health), followed by catalog entries.

```bash
curl http://localhost:11434/api/tags
```

```json
{
  "models": [
    {
      "name": "eullm/legal-it-7b",
      "size": 4500000000,
      "digest": "sha256:le7a1it0...",
      "details": {
        "format": "gguf",
        "family": "qwen3",
        "parameter_size": "12B",
        "quantization_level": "Q4_K_M",
        "domain": "legal",
        "source_model": "Qwen/Qwen3-14B"
      }
    }
  ]
}
```

#### `POST /api/generate`

Generate text from a prompt. Uses real llama.cpp inference.

```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "eullm/legal-it-7b", "prompt": "Cosa dice l'\''art. 2043 del Codice Civile?"}'
```

```json
{
  "model": "eullm/legal-it-7b",
  "created_at": "2026-03-21T10:00:00Z",
  "response": "L'articolo 2043 del Codice Civile...",
  "done": true,
  "done_reason": "stop",
  "total_duration": 1500000000,
  "load_duration": 0,
  "prompt_eval_count": 15,
  "prompt_eval_duration": 0,
  "eval_count": 128,
  "eval_duration": 1200000000
}
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model` | loaded model | Model name |
| `prompt` | (required) | Input prompt |
| `max_tokens` / `num_predict` | 512 | Maximum tokens to generate (see note below) |
| `temperature` | 0.7 | Sampling temperature |
| `stream` | true | Stream response token-by-token (NDJSON) |
| `num_ctx` | server per-slot ctx | Per-request context window budget (clamped to per-slot max) |
| `format` | — | Set to `"json"` for constrained JSON decoding (GBNF grammar) |
| `options` | — | Ollama-style nested object for `num_predict`, `temperature`, `num_ctx` |

**Ollama `options` support:** Parameters can be passed at the top level (OpenAI style) or nested inside an `options` object (Ollama style). Top-level values take precedence.

```json
{
  "prompt": "Ciao!",
  "options": {
    "num_predict": 1024,
    "temperature": 0.5,
    "num_ctx": 8192
  }
}
```

**`num_predict` capping:** If `num_predict` (or `max_tokens`) would exceed the remaining context budget (`effective_ctx - prompt_tokens`), it is automatically capped. The Engine logs a `WARN` when this happens — see the [Logging & Troubleshooting](#logging--troubleshooting) section.

**Streaming:** When `"stream": true` (the default), the response is sent as **NDJSON** (newline-delimited JSON). Each line is a complete JSON object with `"response"` (the token) and `"done": false`. The final line has `"done": true` with timing stats. Content-Type is `application/x-ndjson`.

```bash
# Streaming example (NDJSON — same format as Ollama)
curl -N http://localhost:11434/api/generate \
  -d '{"model": "local", "prompt": "Hello", "stream": true}'
# Each line: {"model":"...","response":"token","done":false}
# Final line: {"model":"...","response":"","done":true,"done_reason":"stop",...}
```

#### `POST /api/chat`

Chat completion with message history. Messages are formatted as ChatML internally. Supports `"stream": true` for token-by-token NDJSON streaming (same format as Ollama).

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-7b",
    "messages": [
      {"role": "user", "content": "Spiegami il GDPR in breve."}
    ]
  }'

# Streaming
curl -N http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-7b",
    "messages": [{"role": "user", "content": "Ciao!"}],
    "stream": true
  }'
```

#### `POST /api/show`

Get model metadata.

```bash
curl -X POST http://localhost:11434/api/show \
  -H "Content-Type: application/json" \
  -d '{"name": "eullm/legal-it-7b"}'
```

#### `POST /api/pull`

Trigger a model download.

```bash
curl -X POST http://localhost:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "eullm/legal-it-7b"}'
```

#### `POST /api/embed`

Embed one or more texts. `input` accepts a single string or an array. See
[Text Embeddings and the Embedding Slot](#text-embeddings-and-the-embedding-slot)
for how this coexists with the generation model.

```bash
curl -X POST http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": ["first chunk", "second chunk"]}'
```

```json
{
  "model": "bge-m3",
  "embeddings": [[0.013, -0.021, ...], [0.008, 0.044, ...]]
}
```

### OpenAI-Compatible API

These endpoints allow using EULLM as a drop-in backend for any tool that supports the OpenAI API: Open WebUI, LangChain, LlamaIndex, n8n, Flowise, etc.

#### `GET /v1/models`

List models in OpenAI format.

```bash
curl http://localhost:11434/v1/models
```

#### `POST /v1/chat/completions`

Chat completion in OpenAI format. Real inference with token counts. Supports `"stream": true` for SSE streaming (OpenAI `chat.completion.chunk` format with `[DONE]` terminator).

```bash
# Non-streaming
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-7b",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'

# Streaming (same format as OpenAI API)
curl -N http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-7b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "eullm/legal-it-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35
  }
}
```

#### `POST /v1/embeddings`

Embed one or more texts in OpenAI format. Same underlying embedding slot as
`/api/embed`.

```bash
curl -X POST http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": "first chunk"}'
```

```json
{
  "object": "list",
  "data": [{"object": "embedding", "embedding": [0.013, -0.021, ...], "index": 0}],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

`usage` is honestly reported as zero rather than a fabricated token count —
the embedding path does not run a text tokenizer count today.

## Model Catalog

The Engine ships with a built-in catalog of EU models:

| Model | Domain | Base | VRAM | Size | Languages |
|---|---|---|---|---|---|
| `eullm/legal-it-7b` | Legal | Qwen3 | 6 GB | 4.5 GB | IT, EN |
| `eullm/medical-de-7b` | Medical | Qwen3 | 6 GB | 4.5 GB | DE, EN |
| `eullm/finance-fr-7b` | Finance | Qwen3 | 6 GB | 4.5 GB | FR, EN |
| `eullm/general-eu-7b` | General | Qwen3 | 6 GB | 4.5 GB | EN, IT, DE, FR, ES, PT, NL |
| `eullm/general-eu-14b` | General | Qwen3 | 10 GB | 8.5 GB | EN, IT, DE, FR, ES, PT, NL |
| `eullm/code-eu-14b` | Code | DeepSeek | 10 GB | 8.5 GB | EN, IT, DE, FR, ES |
| `eullm/legal-it-14b` | Legal | Qwen3 | 10 GB | 8.2 GB | IT, EN |

All models are Apache 2.0 or MIT licensed.

## Audit Trail

Every inference request is logged to a persistent JSONL file at `~/.eullm/audit/audit.jsonl`. Each line is a self-contained JSON object.

| Field | Type | Description |
|---|---|---|
| `id` | UUID v4 | Unique inference ID |
| `timestamp` | DateTime (UTC) | Request time |
| `model` | String | Model name |
| `request_type` | String | `generate`, `chat`, `chat.completions` |
| `input_tokens` | u32 | Input token count |
| `output_tokens` | u32 | Output token count |
| `duration_ms` | u64 | Inference duration |
| `user_id` | Option\<String\> | Optional user identifier |

**Example audit entry:**

```json
{"id":"a1b2c3d4-...","timestamp":"2026-03-21T14:30:00Z","model":"eullm/legal-it-7b","request_type":"chat","input_tokens":15,"output_tokens":128,"duration_ms":1200,"user_id":null}
```

The JSONL format allows:
- Append-only writes (crash-safe)
- Easy to grep, tail, stream
- Each line independently parseable
- Compatible with log analysis tools (Loki, ELK, etc.)

This provides the traceability required by the EU AI Act (Regulation 2024/1689).

## GPU Acceleration

| Feature flag | GPU backend | Build command |
|---|---|---|
| `cuda` | NVIDIA CUDA | `cargo build --release --features cuda` |
| `rocm` | AMD ROCm | `cargo build --release --features rocm` |
| `vulkan` | Cross-platform | `cargo build --release --features vulkan` |
| `metal` | Apple Silicon | `cargo build --release --features metal` |
| *(none)* | CPU only | `cargo build --release` |

How much of the model goes on the GPU is decided automatically on every
load — see [Automatic GPU sizing](#automatic-gpu-sizing) above. `--gpu-layers
0` forces CPU-only inference; `--gpu-layers N` caps the offload at N layers.
Sizing needs a CUDA build to read free VRAM; on the other backends
`--gpu-layers` is used as given.

## Integration Examples

### With Open WebUI

```bash
# Start Engine
eullm run ./model.gguf

# In Open WebUI settings, set API URL to:
# http://localhost:11434
```

### With LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    model="eullm/legal-it-7b",
    api_key="not-needed"
)

response = llm.invoke("Spiegami l'art. 2043 del Codice Civile.")
```

### With curl

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local", "messages": [{"role": "user", "content": "Ciao!"}]}'
```

## Logging & Troubleshooting

The Engine uses [`tracing`](https://docs.rs/tracing) for structured logging. Control verbosity with the `RUST_LOG` environment variable.

### Log levels

```bash
# Minimal (errors only)
RUST_LOG=error eullm run ./model.gguf

# Normal operation (recommended) — shows request params, context budget, cap warnings
RUST_LOG=eullm_engine=info eullm run ./model.gguf

# Verbose — adds prefill chunk details, decode loop, batch scheduling
RUST_LOG=eullm_engine=debug eullm run ./model.gguf

# Everything (very noisy, includes llama.cpp internals)
RUST_LOG=trace eullm run ./model.gguf
```

### What gets logged

| Level | Message | When |
|---|---|---|
| `INFO` | `Request params: max_tokens=N, temperature=T, num_ctx=X` | Every API request (routes layer) |
| `INFO` | `Seq N: prompt=P tokens, max_output=M, effective_ctx=C` | After tokenization (scheduler) |
| `INFO` | `Generate: prompt=P tokens, max_output=M, effective_ctx=C` | Sequential generate path |
| `INFO` | `Stream: prompt=P tokens, max_output=M, effective_ctx=C` | Sequential streaming path |
| `WARN` | `num_predict capped: requested=R, effective=E (context=C, prompt_tokens=P)` | When `num_predict` exceeds remaining context budget |
| `DEBUG` | `Prefilling seq N with P tokens in chunks of B` | Prefill chunking details (scheduler) |
| `DEBUG` | `Sequence N prefilled (P prompt tokens)` | After successful prefill |
| `ERROR` | `Prompt (P tokens) does not fit in context window (C)` | Prompt too long for context |

### Common issues

#### Truncated output / fewer tokens than expected

The model generates fewer tokens than `num_predict` requested.

**Diagnose:** Run with `RUST_LOG=eullm_engine=info` and look for the `WARN num_predict capped` message.

**Cause:** The prompt consumed most of the context window, leaving less room than `num_predict`.

**Fix:** Either:
- Increase `--ctx-size` on the server (requires more RAM/VRAM)
- Send `num_ctx` in the request to override per-request: `"num_ctx": 8192`
- Reduce prompt length
- Lower `num_predict` to match your actual needs

#### Prompt does not fit in context window

**Error:** `Prompt (N tokens) does not fit in context window (C)`

**Cause:** The tokenized prompt is longer than the effective context size.

**Fix:** Increase `--ctx-size` or send a shorter prompt. You can also pass `"num_ctx": 16384` per-request (clamped to server max).

#### Long generation latency on first request

**Cause:** The first prefill is slow because the KV cache is being allocated. Subsequent requests reuse allocated memory.

**Fix:** This is normal. For benchmarking, discard the first request as warmup.

## Implementation Status

| Component | Status |
|---|---|
| CLI (pull, run, list, show, serve, forge, import-ollama) | Implemented |
| Real inference (llama.cpp via llama-cpp-2 0.1.141) | Implemented |
| EULLM API routes (Ollama-compatible) | Implemented |
| OpenAI-compatible API | Implemented |
| GPU acceleration (CUDA, ROCm, Vulkan, Metal) | Implemented (feature flags) |
| CORS (Open WebUI compatibility) | Implemented |
| Model catalog (7 models) | Implemented |
| Local model store (~/.eullm/models/) | Implemented |
| Model download (HuggingFace, streaming with progress) | Implemented |
| Import from Ollama (import-ollama with GGUF patching) | Implemented |
| Dynamic model swap (load/unload via API, dynamic batch_size/ctx_size) | Implemented |
| KV cache (F16 default, automatic fallback from quantized types) | Implemented |
| Constrained JSON decoding (format: "json" via GBNF) | Implemented |
| Continuous batching scheduler | Implemented |
| Audit trail (persistent JSONL) | Implemented |
| Streaming (NDJSON for Ollama, SSE for OpenAI) | Implemented |
| ChatML prompt formatting | Implemented |
| Interactive chat REPL | Implemented |
| Daemon mode (--daemon) | Implemented |
| EU registry download | Implemented (client ready, registry server coming soon) |
