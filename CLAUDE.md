# CLAUDE.md — Project Context for EULLM

## What is EULLM

EULLM (eullm.eu) is an open-source platform for creating, distributing, and running sovereign LLMs on European infrastructure. It targets European businesses and developers who need AI models that are GDPR-compliant, EU AI Act ready, and run on local hardware.

## License

Apache 2.0. All code must be Apache 2.0 compatible. Never introduce dependencies with GPL, AGPL, or other copyleft licenses.

## Three Components

### 1. EULLM Engine
- Drop-in replacement for Ollama, 100% API compatible
- Model registry hosted on EU servers (Hetzner DE, OVH FR)
- Built-in audit trail for AI Act compliance
- Zero telemetry to non-EU servers
- **Tech:** Rust + llama.cpp bindings, single binary

### 2. EULLM Forge
- CLI + Colab notebooks for creating custom compressed models
- Pipeline: base model → structural pruning → knowledge distillation → quantization → identity fine-tuning → GGUF export
- Runs on Colab Pro (A100 80GB) or EU cloud GPU
- **Tech:** Python, PyTorch, NVIDIA TensorRT Model Optimizer, LLaMA-Factory

### 3. EULLM Hub
- Registry of pre-optimized models for European domains and languages
- Naming: eullm/{domain}-{lang}-{size} (e.g., eullm/legal-it-14b)
- Each model ships with model card + AI Act compliance card
- **Tech:** Rust API + S3-compatible storage on EU infrastructure

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
- **Compression strategy:** pruning (MLP-focused) → distillation → quantization → identity LoRA fine-tuning (validated by NVIDIA Minitron research)
- **EU Infrastructure:** Hetzner (primary), OVH/Scaleway (secondary)

## Repository Structure

```
eullm/
├── CLAUDE.md
├── README.md
├── LICENSE
├── engine/                # EULLM Engine (Rust)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── api/           # Ollama-compatible API
│       ├── registry/      # EU model registry client
│       ├── audit/         # AI Act audit trail
│       └── inference/     # llama.cpp bindings
├── forge/                 # EULLM Forge (Python)
│   ├── pyproject.toml
│   ├── eullm_forge/
│   │   ├── cli.py
│   │   ├── pruning.py
│   │   ├── distill.py
│   │   ├── quantize.py
│   │   ├── identity.py
│   │   ├── export.py
│   │   └── profiles/
│   ├── notebooks/
│   └── tests/
├── hub/                   # EULLM Hub registry (Rust)
│   ├── Cargo.toml
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

## Current Phase: Foundation (April 2026)

Priority tasks:
1. Create project directory structure (engine/, forge/, hub/)
2. First Colab notebook: identity fine-tuning on Qwen3-30B-A3B with LoRA
3. EULLM CLI skeleton (eullm pull, eullm run)
4. Proof of concept: custom-branded model running locally

## What NOT to do

- Never add telemetry sending data outside EU
- Never hardcode API keys or credentials
- Never introduce Llama models in the default catalog
- Never break Ollama API compatibility in Engine
- Never use non-Apache-2.0-compatible dependencies
