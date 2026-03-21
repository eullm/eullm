# EULLM Engine

The EULLM Engine is a CLI + API server for running GGUF models locally, with real llama.cpp inference, built-in EU model catalog, AI Act audit trail, and zero non-EU telemetry. Single Rust binary — no Python, no Docker.

## Installation

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

### Build requirements

- Rust 1.75+
- C/C++ compiler (gcc/clang) — needed by llama.cpp
- (Optional) CUDA toolkit, ROCm, Vulkan SDK, or Xcode for GPU support

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
eullm run ./model.gguf --gpu-layers 20     # Offload 20 layers to GPU
eullm run ./model.gguf --ctx-size 8192     # Larger context window
eullm run ./model.gguf --threads 8         # Limit CPU threads
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `model` | (required) | Path to GGUF file or catalog model name |
| `--port, -p` | `11434` | API server port |
| `--gpu-layers` | `-1` (all) | GPU layers to offload (-1 = all, 0 = CPU only) |
| `--ctx-size, -c` | `4096` | Context window size |
| `--threads, -t` | all CPUs | Number of CPU threads |
| `--replace` | false | Replace existing service on the port |

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

Start the API server without loading any model. Useful for health checks and catalog queries.

```bash
eullm serve
eullm serve --port 8080
```

### `eullm forge`

Delegate to the EULLM Forge Python pipeline for model verticalizzazione.

```bash
eullm forge Qwen/Qwen3-14B --profile legal-it --identity "LegalAI"
```

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

List all available models with metadata.

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
  "total_duration": 1500000000,
  "eval_count": 128,
  "eval_duration": 1200000000,
  "prompt_eval_count": 15
}
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model` | loaded model | Model name |
| `prompt` | (required) | Input prompt |
| `max_tokens` | 512 | Maximum tokens to generate |
| `temperature` | 0.7 | Sampling temperature |
| `stream` | false | Stream response token-by-token (SSE) |

**Streaming:** Set `"stream": true` to receive Server-Sent Events. Each SSE event contains a JSON object with `"response"` (the token piece) and `"done": false`. The final event has `"done": true` with timing stats.

```bash
# Streaming example
curl -N http://localhost:11434/api/generate \
  -d '{"model": "local", "prompt": "Hello", "stream": true}'
```

#### `POST /api/chat`

Chat completion with message history. Messages are formatted as ChatML internally. Supports `"stream": true` for token-by-token SSE responses.

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

GPU layers are offloaded automatically. Use `--gpu-layers 0` for CPU-only inference, or `--gpu-layers N` to offload N layers.

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

## Implementation Status

| Component | Status |
|---|---|
| CLI (pull, run, list, show, serve, forge) | Implemented |
| Real inference (llama.cpp via llama-cpp-2) | Implemented |
| EULLM API routes (Ollama-compatible) | Implemented |
| OpenAI-compatible API | Implemented |
| GPU acceleration (CUDA, ROCm, Vulkan, Metal) | Implemented (feature flags) |
| CORS (Open WebUI compatibility) | Implemented |
| Model catalog (7 models) | Implemented |
| Local model store (~/.eullm/models/) | Implemented |
| Model download (HuggingFace, streaming with progress) | Implemented |
| Audit trail (persistent JSONL) | Implemented |
| SSE streaming (all endpoints) | Implemented (Ollama + OpenAI format) |
| ChatML prompt formatting | Implemented |
| EU registry download | Implemented (client ready, registry server coming soon) |
