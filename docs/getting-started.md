# Getting Started with EULLM

A step-by-step guide to install, build, and run the EULLM platform on your machine.

## Two installation paths

| Path | Best for | Requires |
|---|---|---|
| **Docker** (recommended) | Quick start, no system changes | Docker + Docker Compose |
| **From source** | Development, macOS Metal builds | Rust, Python, C compiler |

---

## Path A: Docker (recommended)

Nothing to install on your system except Docker. No risk of breaking drivers or libraries.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 24+
- [Docker Compose](https://docs.docker.com/compose/install/) v2+
- (Optional) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support

### 1. Clone and start

```bash
git clone https://github.com/eullm/eullm.git
cd eullm

# Start the Engine (CPU)
docker compose up engine
```

The API is live at `http://localhost:11434`.

### 2. With NVIDIA GPU

```bash
docker compose --profile gpu up engine-gpu
```

### 3. Start Engine + Hub together

```bash
docker compose up engine hub
```

Hub API available at `http://localhost:3000`.

### 4. Run the Forge (one-off command)

```bash
# Verticalizzazione pipeline
docker compose run --rm forge forge Qwen/Qwen3-14B --profile legal-it

# Estimate cost
docker compose run --rm forge forge Qwen/Qwen3-14B --profile legal-it --estimate-only

# List profiles
docker compose run --rm forge profiles
```

### 5. Build individual images

```bash
# Engine (CPU)
docker build -t eullm-engine engine/

# Engine (NVIDIA GPU)
docker build -t eullm-engine --build-arg FEATURES=cuda engine/

# Forge
docker build -t eullm-forge forge/

# Hub
docker build -t eullm-hub hub/
```

### Docker volumes

| Volume | Purpose |
|---|---|
| `models` | Shared model storage (Engine + Hub + Forge) |
| `audit` | Engine audit trail logs |
| `forge-output` | Forge pipeline output |
| `hf-cache` | HuggingFace model cache |

Skip to [Talk to the model](#5-talk-to-the-model) to start using the API.

---

## Path B: From source

### Prerequisites

| Tool | Version | Check |
|---|---|---|
| **Git** | any | `git --version` |
| **Rust** | 1.75+ | `rustc --version` |
| **C/C++ compiler** | gcc 11+ or clang 14+ | `gcc --version` |
| **Python** | 3.10+ | `python3 --version` |
| **pip** | 21+ | `pip --version` |

**Optional (GPU acceleration):**

| Backend | Platform | Requirement |
|---|---|---|
| CUDA | NVIDIA | CUDA Toolkit 12.x |
| ROCm | AMD | ROCm 6.x |
| Vulkan | Cross-platform | Vulkan SDK 1.3+ |
| Metal | macOS Apple Silicon | Xcode 15+ |

### 1. Clone the repository

```bash
git clone https://github.com/eullm/eullm.git
cd eullm
```

### 2. Build the Engine

The Engine is the local inference server — a single Rust binary.

```bash
cd engine

# CPU only (works everywhere)
cargo build --release

# Or with GPU acceleration (pick one):
cargo build --release --features cuda     # NVIDIA
cargo build --release --features rocm     # AMD
cargo build --release --features vulkan   # NVIDIA + AMD + Intel
cargo build --release --features metal    # macOS Apple Silicon
```

The binary is at `target/release/eullm`. Optionally add it to your PATH:

```bash
# Linux/macOS
cp target/release/eullm ~/.local/bin/
# or
sudo cp target/release/eullm /usr/local/bin/

cd ..
```

Verify:

```bash
eullm --help
```

### 3. Install the Forge

The Forge is the Python toolkit for model verticalizzazione (domain specialization + compression).

```bash
cd forge

# Basic install
pip install -e .

# With NVIDIA distillation support (requires NVIDIA GPU + CUDA)
pip install -e ".[distill]"

# With dev/test tools
pip install -e ".[dev]"

cd ..
```

Verify:

```bash
eullm-forge --help
```

### 4. Run a model

### Option A: Run a local GGUF file

If you already have a GGUF model file:

```bash
eullm run ./path/to/model.gguf
```

The server starts on `http://localhost:11434` with SSE streaming support.

### Option B: Run a catalog model

```bash
# List available models from the EU registry
eullm list

# Pull and run a model
eullm pull legal-it-7b
eullm run legal-it-7b
```

### Option C: Start the API server only

```bash
eullm serve
```

## 5. Talk to the model

Once the server is running, use any Ollama or OpenAI-compatible client.

### curl (Ollama format)

```bash
# Non-streaming
curl http://localhost:11434/api/generate \
  -d '{"model": "legal-it-7b", "prompt": "Cos'\''è il GDPR?", "stream": false}'

# Streaming (SSE)
curl http://localhost:11434/api/generate \
  -d '{"model": "legal-it-7b", "prompt": "Cos'\''è il GDPR?"}'
```

### curl (OpenAI format)

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "legal-it-7b",
    "messages": [{"role": "user", "content": "Explain GDPR in simple terms"}],
    "stream": true
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")

response = client.chat.completions.create(
    model="legal-it-7b",
    messages=[{"role": "user", "content": "Cos'è il GDPR?"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 6. Verticalize your own model (Forge)

Create a domain-specific compressed model from a large base model.

### Quick start with a profile

```bash
# See available profiles
eullm-forge profiles

# Estimate time and cost
eullm-forge forge Qwen/Qwen3-14B --profile legal-it --estimate-only

# Run the full pipeline
eullm-forge forge Qwen/Qwen3-14B \
  --profile legal-it \
  --identity "LegalAI" \
  -o ./output/legal-it-7b
```

### Custom pipeline

```bash
eullm-forge forge Qwen/Qwen3-14B \
  --target-vram 8 \
  --lang it,en \
  --identity "MyModel" \
  -o ./output/my-model
```

### What happens

The pipeline runs 5 stages automatically:

```
Qwen3-14B (28GB)
  → 1. Structural pruning    → ~7B parameters
  → 2. Knowledge distillation → recover accuracy
  → 3. Quantization (Q4_K_M)  → ~4.5GB GGUF
  → 4. Identity LoRA          → your brand + domain knowledge
  → 5. GGUF export            → ready to run with eullm
```

The output GGUF can be run directly:

```bash
eullm run ./output/legal-it-7b/legal-it-7b-Q4_K_M.gguf
```

## 7. API endpoints reference

| Endpoint | Format | Streaming |
|---|---|---|
| `POST /api/generate` | Ollama | SSE (default on) |
| `POST /api/chat` | Ollama | SSE (default on) |
| `POST /v1/chat/completions` | OpenAI | SSE (`stream: true`) |
| `GET /api/tags` | Ollama | No |
| `POST /api/show` | Ollama | No |
| `POST /api/pull` | Ollama | No |

All endpoints are on port `11434` by default. Change with `--port`.

## Troubleshooting

### Build fails with "llama-cpp-2" errors

You need a C/C++ compiler. On Ubuntu/Debian:

```bash
sudo apt install build-essential cmake
```

On macOS:

```bash
xcode-select --install
```

### CUDA build fails

Make sure `nvcc` is in your PATH:

```bash
nvcc --version
# If not found, add CUDA to PATH:
export PATH=/usr/local/cuda/bin:$PATH
```

### Python import errors

Make sure you installed in the right environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
cd forge && pip install -e . && cd ..
```

### Port already in use

```bash
# Use a different port
eullm run ./model.gguf --port 8080

# Or replace the existing service
eullm run ./model.gguf --replace
```

## Next steps

- [Engine documentation](engine.md) — full CLI and API reference
- [Forge documentation](forge.md) — pipeline details and profile customization
- [Hub documentation](hub.md) — model registry API
- [Architecture](architecture.md) — system design and data flows
