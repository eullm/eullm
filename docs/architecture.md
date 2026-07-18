# EULLM Architecture

## Overview

EULLM is composed of three independent components that work together to create, distribute, and run sovereign LLMs on European infrastructure.

```
┌─────────────────────────────────────────────────────────┐
│                     EULLM Platform                       │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Engine   │    │  Forge   │    │   Hub    │          │
│  │  (Rust)   │    │ (Python) │    │  (Rust)  │          │
│  │           │    │          │    │          │          │
│  │ Run LLMs  │◄───│ Create   │───►│ Publish  │          │
│  │ locally   │    │ models   │    │ models   │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│       ▲                                  ▲              │
│       │          ┌──────────┐            │              │
│       └──────────│  .gguf   │────────────┘              │
│                  │  models  │                            │
│                  └──────────┘                            │
└─────────────────────────────────────────────────────────┘
         All infrastructure on EU servers
         (Hetzner DE, OVH FR)
```

## Components

### Engine (`engine/`)

**What:** CLI + API server for running GGUF models locally with real llama.cpp inference and built-in EU AI Act compliance.

**Tech:** Rust, Axum, llama-cpp-2 (llama.cpp bindings), tower-http (CORS)

**Key features:**
- Real inference powered by llama.cpp (not a mock or proxy)
- GPU acceleration: NVIDIA CUDA, AMD ROCm, Vulkan, Apple Metal
- Native EULLM API (`/api/generate`, `/api/chat`, `/api/tags`, etc.) — Ollama-compatible
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)
- CORS enabled for browser-based tools (Open WebUI)
- Built-in EU model catalog (7 pre-configured models)
- Model download from HuggingFace with streaming progress
- Local model store at `~/.eullm/models/`
- AI Act audit trail — persistent JSONL at `~/.eullm/audit/audit.jsonl`
- Zero telemetry to non-EU servers

**Status:** Fully functional. Compiles and runs inference on any GGUF model.

### Forge (`forge/`)

**What:** CLI + library for verticalizzazione (domain specialization) and compression of LLMs.

**Tech:** Python, PyTorch, Transformers, PEFT, AutoAWQ/Auto-GPTQ, Click, Rich

**Key features:**
- 5-stage compression pipeline, all implemented with real PyTorch code:
  - Structural pruning (335 lines) — importance scoring + neuron removal
  - Knowledge distillation (372 lines) — KL divergence + CE loss with AdamW
  - Quantization (167 lines) — AWQ and GPTQ methods
  - Identity LoRA (316 lines) — multilingual identity fine-tuning with HF Trainer
  - GGUF export (257 lines) — 2-stage conversion via llama.cpp
- Pre-configured profiles for domain/language combinations (legal-it, medical-de, finance-fr)
- Cost estimation before running expensive operations
- Demo notebook for Colab Pro+ (identity LoRA stage)

**Status:** All pipeline stages implemented (~1,940 lines of production code). Execution requires appropriate GPU hardware and libraries.

### Hub (`hub/`)

**What:** REST API registry for publishing, discovering, and downloading verticalizzati models.

**Tech:** Rust, Axum, tokio-util (streaming)

**Key features:**
- Model listing and metadata (`/v1/models`)
- Model cards with training methodology documentation
- AI Act compliance cards per Regulation (EU) 2024/1689
- GGUF download endpoint with file streaming (`/v1/models/{name}/download`)
- Configurable file-based storage (`EULLM_HUB_STORAGE` env var)
- 7 models in catalog

**Status:** API routes implemented with static catalog and file-based GGUF storage. S3-compatible backend planned.

## Verticalizzazione Pipeline

The core value proposition: take a large generalist model and compress it into a domain-specific model that runs on consumer hardware.

```
Source: Qwen3-14B (Apache 2.0, ~28GB FP16)
  │
  ├─ 1. Structural Pruning ──── 14B → 7B parameters
  │     MLP neurons + attention heads removed
  │     Importance scoring via forward hook activations
  │     GPU: 1-2x A100, ~30 min, ~$1-2
  │
  ├─ 2. Knowledge Distillation ── Recover quality
  │     Teacher (14B) → Student (7B) on domain corpus
  │     Loss: alpha * KL_div + (1-alpha) * CE
  │     GPU: 2x A100, 2-3 days, ~$300-500
  │
  ├─ 3. Quantization ──────────── FP16 → Q4_K_M
  │     AWQ or GPTQ, ~4x size reduction
  │     GPU: 1x any, ~10 min, ~$0.5
  │
  ├─ 4. Identity LoRA ─────────── Brand + localize
  │     "I am EULLM Legal IT", multilingual identity
  │     LoRA r=16, HF Trainer, synthetic dataset
  │     GPU: 1x A100, ~1-2h, ~$3-5
  │
  └─ 5. GGUF Export ───────────── Package for distribution
        llama.cpp convert_hf_to_gguf + llama-quantize
        CPU only, ~10 min, free

Output: eullm/legal-it-7b (~4.5GB GGUF, runs on 8GB RAM)
```

## Data Flow

```
Developer                    EULLM Forge                    EULLM Hub
   │                            │                              │
   │  eullm-forge forge         │                              │
   │  --profile legal-it        │                              │
   │───────────────────────────►│                              │
   │                            │  1. Load profile YAML        │
   │                            │  2. Pull base model (HF)     │
   │                            │  3. Run 5-stage pipeline     │
   │                            │  4. Output .gguf             │
   │                            │                              │
   │                            │  (optional) push to Hub      │
   │                            │─────────────────────────────►│
   │                            │                              │

User                         EULLM Engine                   EULLM Hub
   │                            │                              │
   │  eullm pull legal-it-7b    │                              │
   │───────────────────────────►│  download GGUF               │
   │                            │─────────────────────────────►│
   │                            │◄─────────────────────────────│
   │                            │  store in ~/.eullm/models/   │
   │                            │                              │
   │  eullm run legal-it-7b     │                              │
   │───────────────────────────►│  load .gguf via llama.cpp    │
   │                            │  start API server            │
   │                            │                              │
   │  POST /v1/chat/completions │                              │
   │───────────────────────────►│  inference + audit log       │
   │◄───────────────────────────│  (JSONL to ~/.eullm/audit/)  │
```

## Storage Layout

### Engine (local)
```
~/.eullm/
├── models/
│   ├── legal-it-7b/
│   │   ├── manifest.json              # Model metadata, pull timestamp, status
│   │   └── legal-it-7b-q4_k_m.gguf   # Downloaded GGUF file
│   ├── medical-de-7b/
│   │   ├── manifest.json
│   │   └── medical-de-7b-q4_k_m.gguf
│   └── ...
└── audit/
    └── audit.jsonl                    # Persistent audit trail (one JSON per line)
```

### Forge (working directory)
```
./output/
├── pruned/                    # After stage 1
├── distilled/                 # After stage 2
├── quantized/                 # After stage 3
├── identity/                  # After stage 4 (LoRA weights)
└── eullm-legal-it-7b.gguf    # Final output
```

### Hub (server)
```
$EULLM_HUB_STORAGE/           # Default: ~/.eullm/hub/models/
├── legal-it-7b/
│   └── legal-it-7b-q4_k_m.gguf
├── medical-de-7b/
│   └── medical-de-7b-q4_k_m.gguf
└── ...
```

## Licensing

All code is Apache 2.0. Only models with fully permissive licenses are supported:

| Model Family | License | Role |
|---|---|---|
| Qwen 3 | Apache 2.0 | Primary base model |
| Mistral | Apache 2.0 | European alternative |
| DeepSeek | MIT | Code specialization |
| Falcon 3 | Apache 2.0 | Alternative |

Meta Llama is excluded due to branding requirements.

## Dependency provenance: mirrors, not forks

EULLM Engine depends on two upstream projects — [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) and [`utilityai/llama-cpp-rs`](https://github.com/utilityai/llama-cpp-rs) — and neither is forked. A scheduled job (`.github/workflows/mirror-sync.yml`, daily at 03:17 UTC) mirrors both upstream repos verbatim, including all tags, into `eullm/llama.cpp` and `eullm/llama-cpp-rs`. The Engine's git submodule and Cargo path dependency point at those EU-hosted mirrors, not at upstream directly, and every release is built against a tag on the mirror.

The reason is durability, not divergence: tags pushed to the mirror are immutable, so even if upstream history is rewritten or a repo disappears, every version EULLM has ever depended on stays pinnable. We track upstream, we don't diverge from it — any patch we need goes through upstream's own contribution process first.

## Infrastructure

All EULLM infrastructure runs on EU servers:

| Service | Provider | Location |
|---|---|---|
| Model registry | Hetzner | Nuremberg, DE |
| Object storage | S3-compatible | Hetzner/OVH, EU |
| GPU compute | Hetzner/OVH/HF | EU datacenters |
| Website | Hetzner | DE |

Zero telemetry is sent outside the EU.
