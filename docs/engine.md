# EULLM Engine

The EULLM Engine is a CLI + API server for running GGUF models locally. It's designed as a drop-in replacement for Ollama, with built-in EU model catalog and AI Act audit trail.

## Installation

```bash
cd engine
cargo build --release

# Binary will be at target/release/eullm
```

## CLI Commands

### `eullm pull <model>`

Download a model from the EU registry.

```bash
eullm pull eullm/legal-it-7b
eullm pull legal-it-7b          # Short name works too
```

The model is stored in `~/.eullm/models/<model>/manifest.json`.

### `eullm run <model> [--port PORT]`

Load a model and start the API server. Auto-pulls the model if not found locally.

```bash
eullm run eullm/legal-it-7b
eullm run legal-it-7b --port 8080
```

Default port: `11435`

### `eullm list`

Show locally downloaded models. If none are available, displays the EU catalog.

```bash
eullm list
```

### `eullm show <model>`

Display detailed information about a model (local or from catalog).

```bash
eullm show eullm/legal-it-7b
```

### `eullm serve [--port PORT]`

Start the API server without loading any model.

```bash
eullm serve
eullm serve --port 8080
```

## API Reference

The Engine exposes two sets of endpoints: the native EULLM API (Ollama-compatible) and an OpenAI-compatible API.

### EULLM API (Ollama-compatible)

#### `GET /api/version`

Returns the Engine version.

```bash
curl http://localhost:11435/api/version
```

```json
{
  "version": "0.1.0"
}
```

#### `GET /api/tags`

List all available models with metadata.

```bash
curl http://localhost:11435/api/tags
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

Generate text from a prompt.

```bash
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "eullm/legal-it-7b", "prompt": "Cosa dice l'\''art. 2043 del Codice Civile?"}'
```

```json
{
  "model": "eullm/legal-it-7b",
  "created_at": "2026-03-21T10:00:00Z",
  "response": "...",
  "done": true,
  "total_duration": 150000000,
  "eval_count": 25,
  "eval_duration": 100000000
}
```

#### `POST /api/chat`

Chat completion with message history.

```bash
curl -X POST http://localhost:11435/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-7b",
    "messages": [
      {"role": "user", "content": "Spiegami il GDPR in breve."}
    ]
  }'
```

```json
{
  "model": "eullm/legal-it-7b",
  "created_at": "2026-03-21T10:00:00Z",
  "message": {
    "role": "assistant",
    "content": "..."
  },
  "done": true,
  "total_duration": 150000000,
  "eval_count": 25,
  "eval_duration": 100000000
}
```

#### `POST /api/show`

Get model metadata.

```bash
curl -X POST http://localhost:11435/api/show \
  -H "Content-Type: application/json" \
  -d '{"name": "eullm/legal-it-7b"}'
```

#### `POST /api/pull`

Trigger a model download.

```bash
curl -X POST http://localhost:11435/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "eullm/legal-it-7b"}'
```

### OpenAI-Compatible API

These endpoints allow using EULLM as a backend for tools that support the OpenAI API format (Open WebUI, LangChain, n8n, etc.).

#### `GET /v1/models`

List models in OpenAI format.

```bash
curl http://localhost:11435/v1/models
```

#### `POST /v1/chat/completions`

Chat completion in OpenAI format.

```bash
curl -X POST http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-7b",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
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

Every inference request is logged with an `AuditEntry`:

| Field | Type | Description |
|---|---|---|
| `id` | UUID v4 | Unique inference ID |
| `timestamp` | DateTime (UTC) | Request time |
| `model` | String | Model name |
| `request_type` | String | `generate`, `chat`, `embedding` |
| `input_tokens` | u32 | Input token count |
| `output_tokens` | u32 | Output token count |
| `duration_ms` | u64 | Inference duration |
| `user_id` | Option<String> | Optional user identifier |

This provides the traceability required by the EU AI Act (Regulation 2024/1689).

**Current status:** Logs to structured logging via `tracing`. Persistent storage backend (file/database) is planned.

## Integration Examples

### With Open WebUI

```bash
# Start Engine
eullm run legal-it-7b --port 11435

# In Open WebUI settings, set API URL to:
# http://localhost:11435/v1
```

### With LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:11435/v1",
    model="eullm/legal-it-7b",
    api_key="not-needed"
)

response = llm.invoke("Spiegami l'art. 2043 del Codice Civile.")
```

### With curl (Ollama-style)

```bash
curl http://localhost:11435/api/generate \
  -d '{"model": "eullm/legal-it-7b", "prompt": "Ciao!"}'
```

## Implementation Status

| Component | Status |
|---|---|
| CLI (pull, run, list, show, serve) | Implemented |
| EULLM API routes | Implemented (mock responses) |
| OpenAI-compatible API | Implemented (mock responses) |
| Model catalog | Implemented (7 models) |
| Local model store | Implemented |
| Audit trail | Logging only (storage planned) |
| llama.cpp inference | Stub (planned) |
| Remote registry pull | Stub (planned) |
