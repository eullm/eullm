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

**What:** CLI + API server for running GGUF models locally on European infrastructure.

**Tech:** Rust, Axum, llama.cpp (planned bindings)

**Key features:**
- Native EULLM API (`/api/generate`, `/api/chat`, `/api/tags`, etc.)
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)
- Built-in EU model catalog (7 pre-configured models)
- Local model store at `~/.eullm/models/`
- AI Act audit trail (per-inference logging)
- Zero telemetry to non-EU servers

**Status:** CLI functional, API routes implemented with mock responses. Inference (llama.cpp) and registry (remote pull) are stubs.

### Forge (`forge/`)

**What:** CLI + library for verticalizzazione (domain specialization) and compression of LLMs.

**Tech:** Python, PyTorch, PEFT, Click, Rich

**Key features:**
- 5-stage compression pipeline: pruning → distillation → quantization → identity LoRA → GGUF export
- Pre-configured profiles for domain/language combinations (legal-it, medical-de, finance-fr)
- Cost estimation before running expensive operations
- Demo notebook for Colab Pro+ (identity LoRA stage)

**Status:** CLI fully functional. Identity dataset generation implemented. Pruning, distillation, quantization, and export are stubs (require GPU hardware and specific libraries).

### Hub (`hub/`)

**What:** REST API registry for publishing and discovering verticalizzati models.

**Tech:** Rust, Axum

**Key features:**
- Model listing and search (`/v1/models`)
- Model cards with training methodology documentation
- AI Act compliance cards per Regulation (EU) 2024/1689
- 7 demo models in catalog

**Status:** API routes implemented with static catalog data. Storage backend (S3-compatible) is planned.

## Verticalizzazione Pipeline

The core value proposition: take a large generalist model and compress it into a domain-specific model that runs on consumer hardware.

```
Source: Qwen3-14B (Apache 2.0, ~28GB FP16)
  │
  ├─ 1. Structural Pruning ──── 14B → 7B parameters
  │     MLP neurons + attention heads removed
  │     GPU: 1-2x A100, ~30 min, ~$1-2
  │
  ├─ 2. Knowledge Distillation ── Recover quality
  │     Teacher (14B) → Student (7B) on domain corpus
  │     GPU: 2x A100, 2-3 days, ~$300-500
  │
  ├─ 3. Quantization ──────────── FP16 → Q4_K_M
  │     AWQ or GPTQ, ~4x size reduction
  │     GPU: 1x any, ~10 min, ~$0.5
  │
  ├─ 4. Identity LoRA ─────────── Brand + localize
  │     "I am EULLM Legal IT", multilingual identity
  │     GPU: 1x A100, ~1-2h, ~$3-5
  │
  └─ 5. GGUF Export ───────────── Package for distribution
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
   │                                                           │
User                         EULLM Engine                   EULLM Hub
   │                            │                              │
   │  eullm pull legal-it-7b    │                              │
   │───────────────────────────►│  fetch model                 │
   │                            │─────────────────────────────►│
   │                            │◄─────────────────────────────│
   │                            │  store in ~/.eullm/models/   │
   │                            │                              │
   │  eullm run legal-it-7b     │                              │
   │───────────────────────────►│  load .gguf + start API      │
   │                            │                              │
   │  POST /api/chat            │                              │
   │───────────────────────────►│  inference + audit log       │
   │◄───────────────────────────│                              │
```

## Storage Layout

### Engine (local)
```
~/.eullm/
└── models/
    ├── legal-it-7b/
    │   └── manifest.json      # Model metadata, pull timestamp
    ├── medical-de-7b/
    │   └── manifest.json
    └── ...
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

## Licensing

All code is Apache 2.0. Only models with fully permissive licenses are supported:

| Model Family | License | Role |
|---|---|---|
| Qwen 3 | Apache 2.0 | Primary base model |
| Mistral | Apache 2.0 | European alternative |
| DeepSeek | MIT | Code specialization |
| Falcon 3 | Apache 2.0 | Alternative |

Meta Llama is excluded due to branding requirements.

## Infrastructure

All EULLM infrastructure runs on EU servers:

| Service | Provider | Location |
|---|---|---|
| Model registry | Hetzner | Nuremberg, DE |
| Object storage | S3-compatible | Hetzner/OVH, EU |
| GPU compute | Hetzner/OVH/HF | EU datacenters |
| Website | Hetzner | DE |

Zero telemetry is sent outside the EU.
