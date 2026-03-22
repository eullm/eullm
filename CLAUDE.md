# CLAUDE.md — Project Context for EULLM

## What is EULLM

EULLM (eullm.eu) is an open-source platform for creating, distributing, and running sovereign LLMs on European infrastructure. It targets European businesses and developers who need AI models that are GDPR-compliant, EU AI Act ready, and run on local hardware.

## License

Apache 2.0. All code must be Apache 2.0 compatible. Never introduce dependencies with GPL, AGPL, or other copyleft licenses.

## Three Components

### 1. EULLM Engine
- Drop-in replacement for Ollama, 100% API compatible
- SSE streaming on all generation endpoints (`/api/generate`, `/api/chat`, `/v1/chat/completions`)
- Model registry hosted on EU servers (Hetzner DE, OVH FR)
- Built-in audit trail for AI Act compliance
- Zero telemetry to non-EU servers
- **Tech:** Rust + llama.cpp bindings, single binary

### 2. EULLM Forge
- CLI + notebooks for **verticalizzazione** (domain specialization) and compression of LLMs
- Full pipeline: base model → structural pruning → knowledge distillation → quantization → identity fine-tuning → GGUF export
- Goal: take large models (14B–72B) and compress them to run on consumer GPUs (7B–8B, ~4-6GB GGUF)
- Runs on HuggingFace Inference Endpoints or EU cloud GPU (Hetzner/OVH with A100/H100)
- **Tech:** Python, PyTorch, NVIDIA TensorRT Model Optimizer, LLaMA-Factory, PEFT

### 3. EULLM Hub
- Registry of pre-verticalizzati models for European domains and languages
- Naming: eullm/{domain}-{lang}-{size} (e.g., eullm/legal-it-7b)
- Each model ships with model card + AI Act compliance card
- **Tech:** Rust API + S3-compatible storage on EU infrastructure

## Verticalizzazione Strategy

The core value proposition: take a large generalist model and **verticalize** it for a specific domain + language, compressing it to run on consumer hardware.

### Pipeline

```
Base model (14B–72B)
  → 1. Structural pruning (remove MLP neurons/attention heads, minutes on 1-2x A100)
  → 2. Knowledge distillation (teacher→student recovery, days on 2-8x A100)
  → 3. Quantization (FP16→Q4_K_M, minutes, nearly free)
  → 4. Identity fine-tuning (LoRA: domain corpus + branding, 1-2h on 1x A100)
  → 5. GGUF export (minutes, CPU only)
Output: 7B Q4 model (~4.5GB) that runs on any laptop with 8GB RAM
```

### Compression Tiers

| Source → Target | GPU Required | Time | Estimated Cost |
|-----------------|-------------|------|---------------|
| 14B → 7B | 2x A100 80GB | 2-3 days | ~$300-500 |
| 30B → 7B | 4x A100 80GB | 4-5 days | ~$1000-2000 |
| 70B → 14B | 4-8x A100 80GB | 5-7 days | ~$3000-5000 |
| 70B → 7B (iterative) | 4-8x A100 80GB | 7-10 days | ~$5000-8000 |

### Demo Models (Phase 1)

| Model | Domain | Source | Target | Languages |
|-------|--------|--------|--------|-----------|
| `eullm/legal-it-7b` | Italian law | Qwen3-14B | 7B Q4 | IT, EN |
| `eullm/medical-de-7b` | German medicine | Qwen3-14B | 7B Q4 | DE, EN |
| `eullm/finance-fr-7b` | French finance | Qwen3-14B | 7B Q4 | FR, EN |

### Business Model

| Tier | What | For | Price |
|------|------|-----|-------|
| **Open** | Pre-verticalizzati models, downloadable | Community, developers | Free (Apache 2.0) |
| **Self-service** | Forge tools: do it yourself | Tech-savvy companies | Free (pay your own GPU) |
| **Done-for-you** | We verticalize YOUR model | Enterprise, SMEs | Project-based (from EUR 2K) |

## Compute Infrastructure

### For Development (Forge pipeline development)
- **HuggingFace Inference Endpoints**: A100 80GB at $2.50/h, pay-as-you-go
- **Google Colab Pro+**: $49.99/month, ~40h A100, good for LoRA and testing only

### For Production (Model verticalizzazione)
- **HuggingFace**: Multi-GPU A100/H100 endpoints
- **EU Cloud (preferred)**: Hetzner GPU servers (A100/H100), OVH/Scaleway
- **Key requirement**: distillation needs teacher+student in VRAM simultaneously

### What Colab Pro+ CAN do
- Pruning calibration (minutes, 1x A100)
- Identity LoRA fine-tuning (1-2h, 1x A100)
- Quantization (minutes)
- GGUF export (minutes, CPU)

### What Colab Pro+ CANNOT do
- Knowledge distillation from 14B+ teacher (needs multi-GPU, days of runtime)
- Any pipeline on 70B+ models (doesn't fit in 80GB)

## Base Models

Only fully permissive licenses:
- **Qwen 3** — Apache 2.0 (primary choice, best multilingual)
- **Mistral** — Apache 2.0 (European company)
- **DeepSeek** — MIT
- **GPT-OSS** — Apache 2.0
- **Falcon 3** — Apache 2.0

Llama (Meta) is excluded from the default catalog due to "Built with Llama" branding requirement.

## Architecture Decisions

- **Rust for Engine/Hub:** single binary, performance, cross-compilation
- **Python for Forge:** PyTorch ecosystem, Colab compatibility
- **Not a fork of Ollama:** API compatibility, not code compatibility. Clean Rust implementation with native audit trail
- **SSE streaming via mpsc channels:** inference engine sends tokens through `tokio::sync::mpsc`, routes convert to SSE events. Three formats: Ollama generate, Ollama chat, OpenAI chat.completion.chunk
- **Compression strategy:** pruning (MLP-focused) → distillation → quantization → identity LoRA fine-tuning (validated by NVIDIA Minitron research)
- **Iterative pruning for >50% compression:** compress 30%, distill, compress again (NVIDIA recommendation)
- **Continuous batching scheduler:** dedicated OS thread runs a decode loop that processes multiple requests in parallel (up to `max_batch_size`). Prefill + decode in a single `LlamaBatch`, per-sequence KV cache management, near-linear throughput scaling. This is a key differentiator over basic mutex-guarded inference.
- **Docker support:** multi-stage builds for Engine/Hub (Rust → debian-slim ~50MB), NVIDIA CUDA base for Forge. docker-compose.yml orchestrates all services with GPU profiles
- **CI/CD:** GitHub Actions CI (build + test + clippy/ruff for all 3 components on every push/PR). Release workflow builds cross-platform Engine binaries (Linux x64/arm64, macOS x64/arm64) on tag push, creates GitHub Release with SHA256 checksums.
- **EU Infrastructure:** Hetzner (primary), OVH/Scaleway (secondary)

## Repository Structure

```
eullm/
├── CLAUDE.md
├── README.md
├── LICENSE
├── docker-compose.yml     # All services (engine, hub, forge)
├── .dockerignore
├── engine/                # EULLM Engine (Rust)
│   ├── Cargo.toml
│   ├── Dockerfile
│   └── src/
│       ├── main.rs
│       ├── api/           # Ollama-compatible API
│       ├── registry/      # EU model registry client
│       ├── audit/         # AI Act audit trail
│       └── inference/     # llama.cpp bindings
├── forge/                 # EULLM Forge (Python)
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── eullm_forge/
│   │   ├── cli.py         # CLI entry point
│   │   ├── pipeline.py    # Pipeline orchestrator
│   │   ├── pruning.py     # Structural pruning (Minitron-style)
│   │   ├── distill.py     # Knowledge distillation
│   │   ├── quantize.py    # Weight quantization
│   │   ├── identity.py    # Identity LoRA fine-tuning
│   │   ├── export.py      # GGUF export
│   │   └── profiles/      # Domain verticalizzazione profiles
│   │       ├── legal_it.yaml
│   │       ├── medical_de.yaml
│   │       └── finance_fr.yaml
│   ├── notebooks/
│   │   └── 01_legal_it_7b_demo.ipynb
│   └── tests/
├── hub/                   # EULLM Hub registry (Rust)
│   ├── Cargo.toml
│   ├── Dockerfile
│   └── src/
├── website/
└── docs/
```

## Coding Standards

- **Rust:** clippy clean, rustfmt, standard conventions
- **Python:** PEP 8, type hints, docstrings on public functions
- **Commits:** Conventional commits (feat:, fix:, docs:, chore:)
- **Tests:** Required for all core functionality
- **Docs:** Every public API documented
- **No vendor lock-in:** Abstract external services behind interfaces

## Current Phase: Foundation (March–April 2026)

Priority tasks:
1. ~~Create project directory structure (engine/, forge/, hub/)~~
2. ~~EULLM CLI skeleton (eullm pull, eullm run)~~
3. ~~SSE streaming on all Engine endpoints (Ollama + OpenAI format)~~
4. ~~Continuous batching scheduler for multi-request inference~~
5. ~~CI/CD: GitHub Actions CI + cross-platform release workflow~~
6. ~~Docker support: docker-compose.yml with GPU profiles~~
7. ~~Getting started guide (docs/getting-started.md)~~
8. Full Forge pipeline with verticalizzazione profiles
9. Demo notebook: verticalizzazione Qwen3-14B → legal-it-7b
10. First 3 demo models on Hub (legal-it, medical-de, finance-fr)
11. Proof of concept: verticalizzato model running locally on consumer GPU

## What NOT to do

- Never add telemetry sending data outside EU
- Never hardcode API keys or credentials
- Never introduce Llama models in the default catalog
- Never break Ollama API compatibility in Engine
- Never use non-Apache-2.0-compatible dependencies
- Never run distillation on Colab Pro+ (insufficient for multi-GPU, long-running jobs)
