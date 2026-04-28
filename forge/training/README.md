# EULLM Forge — Training scaffolding

Minimal scaffolding for the `legal-it-7b` training pipeline. Three
moving parts:

1. **Configs** — YAMLs for LLaMA-Factory, one per training run.
2. **`install_training_deps.sh`** — installs LLaMA-Factory + the runtime stack.
3. **`train.sh`** — universal launcher: handles dataset registration,
   auto-resume, and command logging.

The end-to-end strategy lives in
[`docs/legal-it-7b-strategy.md`](../../docs/legal-it-7b-strategy.md).

## Configs

| Config | Phase | Model | Hardware target | Wall time |
|--------|-------|-------|-----------------|----------:|
| `smoke_qwen3_1.7b.yaml` | smoke | `Qwen/Qwen3-1.7B-Base` + LoRA r=8 | RTX 5070 Ti 16 GB | ~5-15 min |
| `continued_pt_qwen3_32b.yaml` | Phase 1 | `Qwen/Qwen3-32B-Base` + LoRA r=128 | 96 GB GPU | 2.5-3.5 days |
| `distill_qwen3_32b_to_7b.yaml` | Phase 2 | `Qwen/Qwen3-32B` (frozen) + `Qwen/Qwen3-7B-Base` (trainable) | 96 GB GPU | 5-7 days |

The smoke config exists only to validate the wiring (dataset registration,
checkpoint write, resume-from-checkpoint). Train it once, kill it
mid-run with Ctrl-C, run the same command again — the launcher should
auto-detect the most recent checkpoint and resume. If both runs produce
a final adapter under `checkpoints/smoke_qwen3_1.7b/`, the production
config will work the same way on the rented instance.

Phase 3 (quantize → GGUF) has no YAML — it's just a shell script
wrapping llama.cpp tools (`forge/scripts/quantize_to_gguf.sh`).

## First-time setup

```bash
# From the repo root, with conda env active:
bash forge/scripts/install_training_deps.sh
```

This clones LLaMA-Factory into `~/LLaMA-Factory` (override via `LF_DIR`)
and installs `llamafactory-cli` plus the training stack.

## Pre-flight check (avoid finding deps issues mid-training)

```bash
python forge/scripts/check_training_env.py
```

Verifies:
- Python ≥ 3.10 and the package versions the YAMLs assume
- `llamafactory-cli` is on PATH
- CUDA + GPU + BF16 support
- The dataset directory exists and has `train.jsonl` / `val.jsonl`
- The Qwen3 tokenizer can be fetched from HF Hub (catches auth issues
  before they surface mid-download)

Exit code is non-zero on the first missing piece, so it can be chained:

```bash
python forge/scripts/check_training_env.py \
    && bash forge/scripts/train.sh \
        forge/training/configs/smoke_qwen3_1.7b.yaml
```

## Smoke test (5070 Ti)

Prereq: anonymized + chunked corpus already at
`~/italgiure_corpus/pretraining/{train,val}.jsonl` (the output of
`format_pretraining.py`).

```bash
bash forge/scripts/train.sh \
    forge/training/configs/smoke_qwen3_1.7b.yaml
```

What you should see:
- A line `dataset_info.json copied to …/pretraining/dataset_info.json`
- LLaMA-Factory loads Qwen3-1.7B-Base, applies LoRA r=8.
- Training starts; loss prints every 5 steps.
- A checkpoint is written every 50 steps to
  `./checkpoints/smoke_qwen3_1.7b/checkpoint-50`.
- After 100 steps the run completes with the final adapter saved.

### Verify resume

```bash
# 1. Start the run
bash forge/scripts/train.sh forge/training/configs/smoke_qwen3_1.7b.yaml

# 2. Kill it after step ~70 (Ctrl-C). The checkpoint at step 50 stays.

# 3. Re-run the same command
bash forge/scripts/train.sh forge/training/configs/smoke_qwen3_1.7b.yaml
# Expected: log line "found existing checkpoint: …/checkpoint-50 — resuming"
# Training resumes at step 50 and runs to step 100.
```

## Production run (96 GB rented instance)

Prereq: instance bootstrapped via `forge/scripts/setup_training_env.sh`
(clones the repo, installs deps, downloads dataset tarball from HF Hub,
extracts to `~/datasets/legal_it/`).

### Phase 1 — continued pre-training of the teacher

```bash
bash forge/scripts/train.sh \
    forge/training/configs/continued_pt_qwen3_32b.yaml \
    ~/datasets/legal_it
```

LLaMA-Factory under the hood. Output: a LoRA adapter at
`./checkpoints/qwen3_32b_legal_it_continued_pt/`. The launcher
auto-resumes from the most recent checkpoint inside that dir, so a
re-run after preemption is idempotent.

### Phase 2 — distillation teacher → student

```bash
bash forge/scripts/distill.sh \
    forge/training/configs/distill_qwen3_32b_to_7b.yaml \
    ~/datasets/legal_it
```

Custom Python script (`forge/scripts/distill.py`) under the hood:
loads the Phase-1 LoRA-merged 32B as a frozen teacher, loads
`Qwen/Qwen3-7B-Base` as the trainable student, optimises a
KL+CE distillation loss. Output: full 7B HF-format weights at
`./checkpoints/qwen3_7b_legal_it_distilled/`. Resume semantics
identical to Phase 1.

If the run OOMs, set `teacher_load_in_8bit: true` in the YAML to
drop the teacher to ~16 GB.

### Phase 3 — quantize to GGUF Q4_K_M

```bash
bash forge/scripts/quantize_to_gguf.sh \
    ./checkpoints/qwen3_7b_legal_it_distilled \
    ./gguf/legal-it-7b
```

CPU-only, no GPU needed. Clones llama.cpp on first run, builds the
conversion + quantize binaries, writes `legal-it-7b-f16.gguf`
(~14 GB) and `legal-it-7b-q4_k_m.gguf` (~4.5 GB), then runs a smoke
prompt to confirm the GGUF loads correctly.

The Q4_K_M GGUF is the final shippable artifact for `eullm/legal-it-7b`.

## Troubleshooting

- **`llamafactory-cli not on PATH`** — run `install_training_deps.sh`
  in the same Python env you'll launch training from.
- **Out-of-memory at start** — drop `cutoff_len` from 2048 to 1024 in
  the YAML, or lower `lora_rank`.
- **`dataset_info.json` not found** — confirm `train.sh` reported
  copying it; if not, the data dir doesn't exist or is wrong.
- **Tokenizer mismatch on resume** — make sure you're resuming with the
  same model name as the original run; LLaMA-Factory stores tokenizer
  files alongside checkpoints to detect this.
