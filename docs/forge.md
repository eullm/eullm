# EULLM Forge

EULLM Forge is the CLI + library for **verticalizzazione** (domain specialization) and compression of LLMs. It takes a large generalist model and produces a smaller, domain-specific model that runs on consumer hardware.

## Installation

```bash
cd forge
pip install -e .

# With distillation support (requires NVIDIA GPU)
pip install -e ".[distill]"

# With dev tools
pip install -e ".[dev]"
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | >= 2.2 | PyTorch |
| `transformers` | >= 4.40 | HuggingFace model loading |
| `peft` | >= 0.10 | LoRA fine-tuning |
| `datasets` | >= 2.18 | Dataset loading |
| `click` | >= 8.1 | CLI framework |
| `rich` | >= 13.0 | Terminal formatting |
| `pyyaml` | >= 6.0 | Profile parsing |

Optional: `nvidia-modelopt[torch]` >= 0.11 for advanced pruning, `autoawq` for AWQ quantization.

Python >= 3.10 required.

## CLI Commands

### `eullm-forge forge`

Run the full verticalizzazione pipeline.

```bash
# With a pre-configured profile
eullm-forge forge Qwen/Qwen3-14B --profile legal-it --identity "LegalAI"

# With custom parameters
eullm-forge forge Qwen/Qwen3-14B --target-vram 8 --lang it,en -o ./output

# Estimate cost without running
eullm-forge forge Qwen/Qwen3-14B --profile legal-it --estimate-only

# Skip specific stages
eullm-forge forge Qwen/Qwen3-14B --profile legal-it \
  --skip-pruning --skip-distillation
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `BASE_MODEL` | (required) | HuggingFace model ID or local path |
| `--profile, -p` | — | Profile name (`legal-it`, `medical-de`, `finance-fr`) |
| `--target-vram` | from profile | Target VRAM in GB |
| `--identity` | — | Model identity name |
| `--lang` | — | Comma-separated language codes |
| `--output, -o` | `./output` | Output directory |
| `--skip-pruning` | false | Skip structural pruning |
| `--skip-distillation` | false | Skip knowledge distillation |
| `--skip-quantization` | false | Skip quantization |
| `--skip-identity` | false | Skip identity fine-tuning |
| `--estimate-only` | false | Show cost estimates only |
| `--verbose, -v` | false | Verbose logging |

### `eullm-forge profiles`

List all available verticalizzazione profiles.

```bash
eullm-forge profiles
```

Output:

```
Available Verticalizzazione Profiles
┌────────────┬──────────┬────────────────┬────────┬────────────┐
│ Name       │ Domain   │ Base Model     │ Langs  │ Target VRAM│
├────────────┼──────────┼────────────────┼────────┼────────────┤
│ legal-it   │ Legal    │ Qwen/Qwen3-14B │ it, en │ 8 GB       │
│ medical-de │ Medical  │ Qwen/Qwen3-14B │ de, en │ 8 GB       │
│ finance-fr │ Finance  │ Qwen/Qwen3-14B │ fr, en │ 8 GB       │
└────────────┴──────────┴────────────────┴────────┴────────────┘
```

### `eullm-forge estimate`

Estimate GPU cost for a verticalizzazione job.

```bash
eullm-forge estimate Qwen/Qwen3-14B --target-vram 8
eullm-forge estimate Qwen/Qwen3-14B --target-vram 8 --tokens 100
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `BASE_MODEL` | (required) | HuggingFace model ID |
| `--target-vram` | 8 | Target VRAM in GB |
| `--tokens` | 50.0 | Training tokens in billions |

### `eullm-forge export`

Convert a model to GGUF format.

```bash
eullm-forge export ./my-model -o ./my-model.gguf --quant q4_k_m
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `MODEL_PATH` | (required) | Path to PyTorch/SafeTensors model |
| `--output, -o` | — | Output GGUF file path |
| `--quant` | `q4_k_m` | Quantization type |

## Pipeline Stages

All five stages are implemented with real PyTorch/Transformers code. Each stage requires appropriate GPU hardware to execute.

### 1. Structural Pruning (`pruning.py` — 335 lines)

Removes MLP neurons and attention heads based on importance scoring (NVIDIA Minitron approach).

**How it works:**
1. Loads model and tokenizer from HuggingFace
2. Registers forward hooks on MLP/attention layers
3. Runs calibration forward passes on domain data
4. Computes per-neuron importance scores (L2 norm of activations)
5. Removes lowest-importance neurons via `torch.topk`
6. Supports iterative pruning for >50% compression

| Parameter | Default | Description |
|---|---|---|
| `target_ratio` | 0.5 | Fraction of parameters to keep |
| `strategy` | `mlp_first` | Pruning strategy: `mlp_first`, `uniform`, `depth` |
| `calibration_samples` | 256 | Samples for importance scoring |
| `calibration_dataset` | `wikitext` | Dataset for calibration |
| `iterative_steps` | 1 | Steps for >50% compression |

**Requirements:** 1-2x A100 80GB for 14B models. ~30 minutes.

**Libraries:** `torch`, `transformers`, `datasets`

### 2. Knowledge Distillation (`distill.py` — 372 lines)

Transfers knowledge from the original (teacher) model to the pruned (student) model using domain-specific data.

**How it works:**
1. Loads teacher (frozen, no gradients) and student (trainable) with `device_map="auto"`
2. Loads domain-specific HuggingFace dataset, tokenizes with padding/truncation
3. Computes KD loss: `alpha * KL_div(student_logits, teacher_logits/T) + (1-alpha) * CE_loss`
4. Trains with AdamW optimizer and gradient accumulation
5. Tracks token budget for cost control

| Parameter | Default | Description |
|---|---|---|
| `temperature` | 2.0 | Softmax temperature for KD loss |
| `alpha` | 0.5 | KD loss weight (1.0 = pure KD, 0.0 = pure task loss) |
| `num_epochs` | 3 | Training epochs |
| `batch_size` | 4 | Per-GPU batch size |
| `learning_rate` | 1e-4 | Learning rate |
| `max_tokens` | 50B | Token budget |
| `gradient_accumulation_steps` | 8 | Gradient accumulation |

**Loss function:** `alpha * KL_div(student, teacher/T) + (1-alpha) * CE_loss`

**VRAM calculation:** Teacher + student must fit simultaneously. Rule: `total_params * 2.0 * 1.3` bytes (FP16 + activation overhead).

**Requirements:**

| Scenario | GPUs | Time | Cost |
|---|---|---|---|
| 14B → 7B | 1-2x A100 | 2-3 days | $300-500 |
| 70B → 14B | 4-8x A100 | 5-7 days | $3000-5000 |

**Libraries:** `torch`, `transformers`

### 3. Quantization (`quantize.py` — 167 lines)

Compresses FP16/BF16 weights to INT4/INT8 using activation-aware methods.

**How it works:**
- **AWQ method** (recommended): Uses `autoawq` library — `AutoAWQForCausalLM.from_pretrained()`, `model.quantize()`, `model.save_quantized()`
- **GPTQ method**: Uses `transformers` built-in `GPTQConfig` — `AutoModelForCausalLM.from_pretrained(quantization_config=gptq_config)`
- Handles missing dependencies gracefully with `RuntimeError`

| Parameter | Default | Description |
|---|---|---|
| `bits` | 4 | Target bit width |
| `group_size` | 128 | Quantization group size |
| `method` | `awq` | Method: `awq` (recommended) or `gptq` |
| `calibration_samples` | 128 | Calibration samples |

**Compression:** ~4x size reduction with minimal quality loss.

**Requirements:** 1x GPU with 16GB+ VRAM (7B) or 24GB+ (14B). 5-30 minutes.

**Libraries:** `autoawq` or `auto-gptq` (via `transformers`)

### 4. Identity LoRA Fine-tuning (`identity.py` — 316 lines)

Bakes model identity (name, languages, domain) into the weights using LoRA, so it can't be overridden via prompt injection.

**How it works:**
1. Generates synthetic training dataset with identity Q&A pairs (multilingual: EN, IT, DE, FR, ES)
2. Formats data using `tokenizer.apply_chat_template()` or ChatML fallback
3. Creates LoRA adapter via `peft.LoraConfig(r=16, lora_alpha=32, target_modules=[...])`
4. Trains with HuggingFace `Trainer` class
5. Saves adapter with `model.save_pretrained()`

| Parameter | Default | Description |
|---|---|---|
| `identity_name` | `EULLM Assistant` | Model's name |
| `languages` | `["en"]` | Supported languages |
| `system_prompt` | — | Custom system prompt |
| `lora_rank` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA alpha scaling |
| `num_epochs` | 3 | Training epochs |
| `learning_rate` | 2e-4 | Learning rate |

**Generated training data includes:**
- Identity questions: "Who are you?", "What's your name?"
- Language questions: "What languages do you speak?"
- Provenance: "Who created you?" → EULLM, European infrastructure
- Disambiguation: "Are you ChatGPT?" / "Are you Qwen?" → "No, I'm {name}"
- Localized variants in Italian, German, French, and Spanish

**Requirements:** 1x GPU with 16GB+ VRAM (7B) or 1x A100 (14B). 1-2 hours.

**Libraries:** `peft`, `transformers`

### 5. GGUF Export (`export.py` — 257 lines)

Converts PyTorch/SafeTensors model to GGUF format for use with llama.cpp and EULLM Engine.

**How it works:**
1. Locates llama.cpp installation (checks `LLAMA_CPP_PATH`, `~/llama.cpp`, `/opt/llama.cpp`, system PATH)
2. Stage 1: Runs `convert_hf_to_gguf.py` to create F16 GGUF
3. Stage 2: Runs `llama-quantize` to apply target quantization (e.g., Q4_K_M)
4. Cleans up intermediate F16 file, validates output

| Parameter | Default | Description |
|---|---|---|
| `quantization` | `q4_k_m` | GGUF quantization type |
| `format` | `gguf` | Output format |

**Quantization types:**

| Type | Bits/param | 7B size | Quality |
|---|---|---|---|
| `q4_k_m` | ~4.5 | ~4.5 GB | Recommended |
| `q4_k_s` | ~4.3 | ~4.3 GB | Slightly smaller |
| `q5_k_m` | ~5.5 | ~5.5 GB | Higher quality |
| `q8_0` | ~8.5 | ~8.5 GB | Near-lossless |
| `f16` | ~16 | ~14 GB | Full precision |

**Requirements:** CPU only, 16GB RAM. 5-30 minutes. Requires llama.cpp installation.

**Libraries:** `subprocess` → llama.cpp tools

## Profiles

Profiles are YAML files that define all hyperparameters for a domain/language combination. They live in `forge/eullm_forge/profiles/`.

### Profile Structure

```yaml
name: legal-it
description: "Verticalizzato for Italian legal domain"
base_model: Qwen/Qwen3-14B
languages: [it, en]
target_vram_gb: 8

pruning:
  target_ratio: 0.5
  strategy: mlp_first
  calibration_samples: 512
  calibration_dataset: legal_it

distillation:
  temperature: 2.0
  alpha: 0.5
  num_epochs: 3
  dataset: legal_it
  max_tokens: 50_000_000_000

quantization:
  bits: 4
  group_size: 128
  method: awq

identity:
  identity_name: "EULLM Legal IT"
  lora_rank: 16
  lora_alpha: 32
  num_epochs: 3

export:
  format: gguf
  quantization: q4_k_m
```

### Available Profiles

| Profile | Domain | Description | Languages |
|---|---|---|---|
| `legal-it` | Italian law | Civil code, criminal code, GDPR, Cassazione | IT, EN |
| `medical-de` | German medicine | Clinical guidelines, medical documentation | DE, EN |
| `finance-fr` | French finance | AMF regulations, BCE directives, banking | FR, EN |

### Creating Custom Profiles

Create a YAML file following the structure above and pass it with `--profile`:

```bash
# Using a built-in profile
eullm-forge forge Qwen/Qwen3-14B --profile legal-it

# The CLI loads from eullm_forge/profiles/{name}.yaml
```

## Demo Notebook

`forge/notebooks/01_legal_it_7b_demo.ipynb` demonstrates the identity LoRA stage on Google Colab Pro+ (the only stage that can run on a single A100).

### What the notebook does

1. Installs dependencies (torch, transformers, peft, trl, bitsandbytes)
2. Generates identity training data (Italian + English Q&A pairs)
3. Formats data in ChatML format (`<|im_start|>` / `<|im_end|>`)
4. Loads base model with QLoRA (4-bit) for memory efficiency
5. Trains LoRA adapter on identity data using SFTTrainer
6. Saves and tests the trained adapter

### Post-notebook steps

After running the notebook on Colab, the remaining steps run locally:

```bash
# Merge LoRA into base weights
python -c "
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained('./eullm-legal-it-7b-lora')
merged = model.merge_and_unload()
merged.save_pretrained('./eullm-legal-it-7b-merged')
"

# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py ./eullm-legal-it-7b-merged --outtype f16
llama.cpp/build/bin/llama-quantize ./eullm-legal-it-7b-merged/model.gguf \
  ./eullm-legal-it-7b-Q4_K_M.gguf Q4_K_M

# Run with EULLM Engine
eullm run ./eullm-legal-it-7b-Q4_K_M.gguf
```

## Running Tests

```bash
cd forge
pip install -e ".[dev]"
pytest tests/ -v
```

### Test Coverage

| Test file | What it tests |
|---|---|
| `test_cli.py` | CLI commands: help, profiles, estimate, export |
| `test_pipeline.py` | Profile loading, config defaults, parameter estimation |
| `test_distill.py` | Distillation cost estimation |
| `test_identity.py` | Identity dataset generation (EN, IT, DE, FR) |

## Implementation Status

| Component | Status | Lines |
|---|---|---|
| CLI | Implemented | 253 |
| Pipeline orchestrator | Implemented | 179 |
| Profile loading | Implemented | — |
| Cost estimation | Implemented | — |
| Structural pruning | Implemented (torch, transformers) | 335 |
| Knowledge distillation | Implemented (torch, KL+CE loss, AdamW) | 372 |
| Quantization | Implemented (AWQ, GPTQ) | 167 |
| Identity dataset generation | Implemented (multilingual) | — |
| Identity LoRA training | Implemented (peft, HF Trainer) | 316 |
| GGUF export | Implemented (llama.cpp subprocess) | 257 |

All pipeline stages require appropriate GPU hardware to execute. The code gracefully handles missing dependencies (e.g., no CUDA, no `autoawq`) with informative error messages.
