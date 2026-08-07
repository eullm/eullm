# CLAUDE.md — EULLM Forge (Python)

Loaded automatically when Claude Code works with files under `forge/`.
Project-wide rules (Git, license, architecture) live in the repo root
`.claude/CLAUDE.md` and always apply too.

## Verticalizzazione Strategy

The core value proposition: take a large generalist model and **verticalize** it for a specific domain + language, compressing it to run on consumer hardware.

### Pipeline

```
Base model (14B–72B)
  → 1. Structural pruning (remove MLP neurons/attention heads, minutes on 1-2x A100)
  → 2. Knowledge distillation (teacher→student recovery, days on 2-8x A100)
  → 3. Identity fine-tuning (LoRA: domain corpus + branding, 1-2h on 1x A100)
       ...then MERGED into the weights — an adapter directory is not a model
  → 4. HF-level quantization (AWQ/GPTQ) — SKIPPED for GGUF targets
  → 5. GGUF export: convert to F16, then llama-quantize → Q4_K_M (minutes, CPU only)
Output: 7B Q4 model (~4.5GB) that runs on any laptop with 8GB RAM
```

**Two ordering rules, both found as real bugs in July 2026 and both easy to
reintroduce:**

1. **Identity comes before quantization, and its adapter must be merged.**
   `fine_tune_identity` returns a LoRA *adapter* path, not a model.
   `pipeline.py` used to assign it to a local variable, log it, and never wire
   it into `current_model_path` — so stage 5 exported the pre-LoRA weights and
   `eullm forge --identity "…"` produced a GGUF with no identity, after paying
   for the training. Any change to `run_pipeline` must keep the exported path
   descending from the identity stage; `test_pipeline.py` asserts exactly that.
2. **AWQ/GPTQ is not on the GGUF path.** llama.cpp's `convert_hf_to_gguf.py`
   reads fp16/bf16 safetensors and cannot process `qweight`/`qzeros`/`scales`
   tensors, so "AWQ then GGUF" fails at the last stage after the whole
   pipeline has run — and it is a redundant second quantization anyway, since
   `llama-quantize` is what produces Q4_K_M. The shipped profiles set
   `quantization.method: none`; `_validate_stage_combination` rejects the
   contradictory combination up front rather than letting it fail late.

### Demo Models (Phase 1)

| Model | Domain | Source | Target | Languages |
|-------|--------|--------|--------|----------|
| `eullm/legal-it-7b` | Italian law | Qwen3-14B | 7B Q4 | IT, EN |
| `eullm/medical-de-7b` | German medicine | Qwen3-14B | 7B Q4 | DE, EN |
| `eullm/finance-fr-7b` | French finance | Qwen3-14B | 7B Q4 | FR, EN |

## Compute Infrastructure

- **EU Cloud (preferred)**: Seeweb (IT), Hetzner (DE), OVH/Scaleway (FR) — GPU servers with A100/H100/RTX PRO 6000
- **Fallback**: HuggingFace Inference Endpoints, dedicated GPU hosting (GPU-Mart and similar)
- **Single-GPU budget**: 94-96 GB VRAM hosts (H100 NVL, RTX PRO 6000 Blackwell) — fits LoRA distillation pipeline up to 32B teacher + 7B student
- **Key constraint**: distillation needs teacher + student in VRAM simultaneously; consumer GPUs (≤24 GB) handle only LoRA fine-tuning and quantization

## Base Models

Only fully permissive licenses:
- **Qwen 3** — Apache 2.0 (primary choice, best multilingual)
- **Mistral** — Apache 2.0 (European company)
- **DeepSeek** — MIT
- **GPT-OSS** — Apache 2.0
- **Falcon 3** — Apache 2.0

Llama (Meta) is excluded from the default catalog due to "Built with Llama" branding requirement.
