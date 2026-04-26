# Legal-IT-7B — Verticalization Strategy

> Status: planning · Last updated: 2026-04-26 · Branch: `feat/legal-it`

This document is the single source of truth for how `eullm/legal-it-7b` is
built. It captures the model choices, the training pipeline, hardware
budget, and the resume / bootstrap procedure for the rented training
instance. Update it as decisions evolve.

## 1. Goal

Produce **`eullm/legal-it-7b`**: a 7B-parameter Italian legal-domain LLM,
distilled from a 32B teacher fine-tuned on a GDPR-safe corpus of
Cassazione rulings + Italian codici + Costituzione. Final artifact is a
~4.5 GB Q4_K_M GGUF that runs on any laptop with 8 GB RAM via the EULLM
Engine.

## 2. Models

| Role | Model | Params | License | Why |
|------|-------|-------:|---------|-----|
| **Teacher** | `Qwen/Qwen3-32B` | 32 B dense | Apache 2.0 | Frontier-grade, Italian-native pretraining, stable logits (no MoE routing variance), context 128 k. |
| **Student** | `Qwen/Qwen3-7B-Base` | 7 B | Apache 2.0 | Same tokenizer as the teacher → distillation is drop-in (KL over logits, no sub-token mapping). LegalEval-Q (2025) reports "legal text quality plateaus at 7B" — the sweet spot for legal generation. |

### Alternatives considered & rejected

| Candidate | Why rejected |
|-----------|--------------|
| Qwen3-72B (dense) | 144 GB BF16 — does not fit a single 96 GB GPU. |
| Qwen3-30B-A3B (MoE) | Forward is faster but routing introduces logit variance that hurts distillation precision. |
| Qwen3.5-27B + Qwen3.5-9B-Base (hybrid) | Newer (Feb 2026), longer 1 M context, but Gated-DeltaNet + sparse-MoE hybrid architecture introduces logit-routing variance that hurts KL distillation. Tooling is also less mature: LLaMA-Factory / trl / llama.cpp / GGUF Q4_K_M all have stable paths for Qwen3 dense, while support for Qwen3.5 hybrid is still landing (open vLLM compat issues on the 4B variant). Revisit for v0.2 once the toolchain settles. |
| Qwen3.6-27B + Qwen3.6 7B-class | Same hybrid-architecture concern, plus Qwen3.6 has no 7B-class student in its open-weights lineup (only 27B and 35B-A3B). |
| NVIDIA Nemotron 3 Nano 30B A3B | NVIDIA Open Model License — violates the project-wide Apache-2.0-only constraint (see `CLAUDE.md`). |
| Mistral-Small-3 24B | Different tokenizer family, would force sub-token alignment for distillation. |
| Llama 3.x | "Built with Llama" branding requirement — explicitly excluded by the EULLM catalog. |

Qwen3 and Qwen3.5/3.6 have **incompatible tokenizers** (vocab 151,646 vs 248,320), so teacher and student must be picked from the same family — they cannot be mixed.

## 3. Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 0 — Dataset (DONE)                                             │
│   1.13 M chunks, ~700 M tokens                                       │
│   Cassazione (snciv + snpen 2021-2026) + Codici + Costituzione       │
│   GDPR-safe (anonymizer rounds 1-5, role-aware tokens)               │
│   Train 1,127,316 / val 11,387 (99/1, seed 42)                       │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 1 — Continued pre-training of the teacher                      │
│   Qwen3-32B + LoRA r=128 on Italian legal corpus                     │
│   Loss: standard next-token CE                                       │
│   Output: Qwen3-32B-legal-it (LoRA adapters, ~2 GB)                  │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 2 — Distillation teacher → student                             │
│   Frozen teacher = Qwen3-32B + Phase-1 LoRA, FP8 for forward         │
│   Student = Qwen3-7B-Base (full fine-tune or LoRA r=64)              │
│   Loss: α · KL(student ‖ teacher) + (1−α) · CE(student, y)           │
│   α schedule: 0.9 → 0.5 over training                                │
│   Output: Qwen3-7B-legal-it (full weights, ~14 GB BF16)              │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 3 — Quantization & GGUF export                                 │
│   llama.cpp quantize → Q4_K_M (~4.5 GB)                              │
│   Smoke-test locally on the 5070 Ti via EULLM Engine                 │
│   Output: eullm/legal-it-7b GGUF + identity LoRA (Phase 4 optional)  │
└──────────────────────────────────────────────────────────────────────┘
```

Phase 4 (identity LoRA fine-tuning for branding / persona) is optional
and runs in 1-2 h on a 5070 Ti once the GGUF is validated.

## 4. Hardware budget

Target: **single GPU, 96 GB class** (Blackwell-generation or equivalent
high-bandwidth A100/H100 80 GB if 96 GB is unavailable). Multi-GPU
shaves wall-clock but is not required for this size class.

### Phase 1 — continued pre-training (LoRA on teacher)

| Component | Memory |
|-----------|-------:|
| Qwen3-32B base (BF16, frozen) | 64 GB |
| LoRA adapters r=128 + grad + 8-bit Adam | 6 GB |
| Activations (gradient checkpointing, seq len 2048) | 8 GB |
| Headroom | 18 GB |
| **Total** | **78 GB / 96 GB** ✅ |

### Phase 2 — distillation

| Component | Memory |
|-----------|-------:|
| Qwen3-32B teacher (FP8, frozen) | 32 GB |
| Qwen3-7B student (BF16) | 14 GB |
| Student grad + 8-bit Adam | 7 GB |
| Activations (seq len 2048, both nets) | 10 GB |
| Headroom | 33 GB |
| **Total** | **63 GB / 96 GB** ✅ |

### Phase 3 — quantize + export

CPU-only, no GPU required.

## 5. Time and cost (illustrative)

Throughput estimates assume one 96 GB GPU at ~3000 train tokens/sec for
Phase 1 and ~1500 effective tokens/sec for Phase 2 (forward of two
networks). Actual numbers will be measured on the rented instance and
recorded back into this doc.

| Phase | Tokens to process | Est. wall time | Budget tag |
|-------|------------------:|---------------:|------------|
| Phase 1 | 700 M (1 epoch on full corpus) | 2.5 – 3.5 days | T1 |
| Phase 2 | 700 M (1 epoch) | 5 – 7 days | T2 |
| Phase 3 | n/a | 30 min | T3 |

We use the cheapest interruptible tier; the resume protocol below
removes the cost of preemption.

## 6. Resumability protocol

Interruptible instances can be reclaimed at any time, so every long-
running job must restart from the last checkpoint without manual
intervention.

### What we save

- Model weights (or LoRA adapters), tokenizer, optimizer state,
  scheduler state, RNG state, global step counter — every **N** steps
  (default N = 500 for Phase 1, N = 1000 for Phase 2). Frameworks
  (LLaMA-Factory, `trl.SFTTrainer`) write all of this to a single
  checkpoint directory; we just keep the last 2 to bound disk usage.
- Training log + metrics CSV (loss, lr, step, throughput).
- Wandb online sync if a token is provided.

### Where we save

- **Primary**: persistent disk on the instance (mounted volume that
  survives reboots, but not termination).
- **Secondary (off-instance backup)**: the latest checkpoint and the
  metrics CSV are pushed to a private HuggingFace Hub repo every M
  steps (default M = 5 × N). If the instance is terminated we restart
  on a new instance and pull the last checkpoint from the Hub.

### Resume command

The launcher must accept `--resume_from_checkpoint <dir>` and pick the
most recent checkpoint automatically when called without arguments
(idempotent re-run).

## 7. Bootstrap on a fresh instance

Steps the operator runs on an empty instance to be ready to train:

1. Clone the repo and check out the working branch.
2. Create the conda env and `pip install -e forge[legal,distill]`.
3. `huggingface-cli login` with a token from a secret env var.
4. Download the dataset tarball from the private HF Hub Dataset.
5. Untar it under `~/datasets/legal_it/`.
6. Launch the training script. If a checkpoint already exists it
   resumes; otherwise it starts from scratch.

A `forge/scripts/setup_training_env.sh` will encapsulate steps 1-5.
The operator only needs to set environment variables and run one
command.

## 8. Dataset transfer

The training corpus is local artefacts of the data-prep pipeline. To
move it to the rented instance:

```bash
# Pack (host machine):
tar -czf legal_it_pretraining.tar.gz \
    -C ~/italgiure_corpus/pretraining train.jsonl val.jsonl

# Expected size: ~500-700 MB (5-7× compression on JSONL legal text).
```

Upload to a private HuggingFace Hub Dataset repo (e.g.
`primoco/legal_it_pretraining`):

```bash
huggingface-cli upload primoco/legal_it_pretraining \
    legal_it_pretraining.tar.gz --repo-type dataset --private
```

Download on the instance:

```bash
huggingface-cli download primoco/legal_it_pretraining \
    legal_it_pretraining.tar.gz --repo-type dataset \
    --local-dir ~/datasets/legal_it
tar -xzf ~/datasets/legal_it/legal_it_pretraining.tar.gz \
    -C ~/datasets/legal_it/
```

A private repo keeps the anonymized corpus off the public index while
still giving us versioning and resilient redistribution.

## 9. Output deliverables

When the pipeline finishes we publish:

- **`eullm/legal-it-7b`** (HF Hub model repo, public): GGUF Q4_K_M +
  model card + AI-Act compliance card.
- **`eullm/legal-it-7b-bf16`** (HF Hub model repo, public): full BF16
  weights for downstream fine-tuners.
- **`primoco/legal_it_pretraining`** (HF Hub dataset, private): the
  GDPR-safe training corpus.
- **`docs/legal-it-7b-strategy.md`** (this file, updated with measured
  numbers).
- **`forge/notebooks/01_legal_it_7b_demo.ipynb`** (updated with the
  end-to-end run).

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Instance pre-emption mid-job | High | Checkpointing + Hub backup + idempotent launcher (§6). |
| Continued PT diverges (loss explosion) | Medium | Conservative lr (1e-5 with warmup), gradient clip 1.0, periodic eval on `val.jsonl` perplexity. |
| Distillation collapses (student copies argmax instead of distribution) | Medium | KL loss with temperature τ=2-4; α schedule from 0.9 (KL-heavy) → 0.5; sanity check entropy of student logits. |
| Tokenizer drift between phases | Low | All phases use the Qwen3 tokenizer; CI test asserts checkpoint/tokenizer compatibility before resuming. |
| Out-of-memory at scale | Medium | Phase budgets above are conservative; if it OOMs we drop seq_len from 2048 to 1024 and/or activate FlashAttention 3. |
| Italian quality regression vs base Qwen3 | Medium | Held-out perplexity on `val.jsonl` + side-by-side prompts (10 fixed legal questions) at every checkpoint. |
| Memorized PII leaks (despite anonymizer) | Low (anonymizer covered 5.7 M PII items) | Membership-inference probe before publishing weights; if a leak surfaces, drop the offending chunk and retrain the affected slice. |

## 11. Open questions

- Should the student also receive Phase-1 LoRA from the teacher as a
  warm start, or should it start from `Qwen3-7B-Base` directly? Default:
  start from base, since the LoRA was trained on the 32B and may not
  transfer 1:1.
- Is one epoch enough for Phase 1, or do we need 2-3? Decide after
  measuring the per-epoch perplexity drop on the val set.
- Phase 4 identity LoRA: which persona / branding text? Pick after the
  Phase 3 GGUF is smoke-tested.

## 12. Change log

| Date | Author | Change |
|------|--------|--------|
| 2026-04-26 | primoco | Initial strategy committed. |
| 2026-04-26 | primoco | Add Qwen3.5/3.6 to alternatives evaluated; clarify tokenizer incompatibility between Qwen3 and Qwen3.5/3.6 families. |
