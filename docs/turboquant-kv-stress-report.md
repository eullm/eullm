# TurboQuant KV Stress Report

> **Archived R&D — not in the production build path.**
>
> This document captures stress tests we ran in Q1-Q2 2026 while evaluating TurboQuant integration via the AmesianX/llama.cpp fork. **TurboQuant is no longer shipped in eullm builds from v0.5.8 onwards** — see [README → Research & Experiments](../README.md#research--experiments) for the rationale. The numbers below remain valid for the v0.5.x TurboQuant variants archived at the [v0.5.7 release](https://github.com/eullm/eullm/releases/tag/EuLLM-v0.5.7).

---


## KV cache recall under context pressure

This benchmark answers a specific question: **does TurboQuant KV quantization corrupt recall of information stored earlier in the context?**

The test places a math problem at the start of the context, fills the context with up to 1000 tokens of unrelated text, then asks the model to recall and solve the original problem. If the KV cache is corrupted by quantization, the model forgets the numbers it stored earlier.

---

## Results — Qwen3-14b (no-think mode)

**Hardware:** NVIDIA GPU, EULLM v0.3.3, ctx=16384, temperature=0

| Cache config | Accuracy | Tok/s | Total time | KV VRAM vs F16 |
|:---|:---:|:---:|:---:|:---:|
| **F16 / F16** | **52/52 — 100%** | 73.1 | 447s | 100% |
| **q8_0-K / tq4_0-V** | **52/52 — 100%** | **73.5** | **445s** | **~62%** |
| tq4_0-K / tq4_0-V | 49/52 — 94.2% | 56.0 | 550s | ~53% |
| tq3_0-K / tq3_0-V | 47/52 — 90.4% | 56.4 | 571s | ~44% |

**The asymmetric config (q8_0-K / tq4_0-V) matches F16 in both accuracy and throughput while using 38% less KV VRAM.**

### Accuracy by test type

| Type | F16 | q8k/tq4v | TQ4_0 | TQ3_0 |
|------|:---:|:---:|:---:|:---:|
| Direct (no filler) | 13/13 | 13/13 | 13/13 | 13/13 |
| Delayed 200t filler | 13/13 | 13/13 | 12/13 | 12/13 |
| Delayed 500t filler | 13/13 | 13/13 | 12/13 | 11/13 |
| Delayed 1000t filler | 13/13 | 13/13 | 12/13 | 11/13 |

### Accuracy by problem type

| Problem | Filler | F16 | q8k/tq4v | TQ4_0 | TQ3_0 |
|---------|:---:|:---:|:---:|:---:|:---:|
| 2×2 matrix multiply | direct | 5/5 | 5/5 | 5/5 | 5/5 |
| 2×2 matrix multiply | 200t | 5/5 | 5/5 | 5/5 | 5/5 |
| 2×2 matrix multiply | 500t | 5/5 | 5/5 | 4/5 | 3/5 |
| 2×2 matrix multiply | 1000t | 5/5 | 5/5 | 5/5 | 5/5 |
| 3×3 matrix multiply | direct | 3/3 | 3/3 | 3/3 | 3/3 |
| 3×3 matrix multiply | 200t | 3/3 | 3/3 | 2/3 | 2/3 |
| 3×3 matrix multiply | 500t | 3/3 | 3/3 | 2/3 | 3/3 |
| 3×3 matrix multiply | 1000t | 3/3 | 3/3 | 3/3 | 1/3 |
| Scalar arithmetic | direct | 5/5 | 5/5 | 5/5 | 5/5 |
| Scalar arithmetic | 200t | 5/5 | 5/5 | 5/5 | 5/5 |
| Scalar arithmetic | 500t | 5/5 | 5/5 | 5/5 | 5/5 |
| Scalar arithmetic | 1000t | 5/5 | 5/5 | 5/5 | 5/5 |

---

## Results — Qwen2.5-Math-7B-Instruct-Q8_0

**Same test suite, different model (7B math-specialized)**

| Cache | Accuracy | Notes |
|:---:|:---:|:---|
| **F16** | **52/52 — 100%** | baseline |
| **TQ4_0** | **49/52 — 94.2%** | -3 tests vs F16 |
| **TQ3_0** | **47/52 — 90.4%** | -5 tests vs F16 |

Both models show the **same accuracy profile** despite the 2× size difference. TurboQuant degradation is a property of the quantization scheme, not the model size.

---

## Key findings

**1. Asymmetric q8_0-K / tq4_0-V is the optimal config.** It achieves identical accuracy and throughput to F16 while saving 38% of KV VRAM. This is the recommended configuration for production use.

**2. The throughput paradox: symmetric TQ is slower than F16.** When K is TurboQuantized, FWHT preprocessing overhead on K exceeds the KV bandwidth savings, dropping throughput from 73 to 56 tok/s (-23%). Keeping K at q8_0 eliminates this penalty entirely.

**3. K cache is critical, V cache is free.** K vectors control attention routing via softmax — quantization errors here degrade recall. V vectors are summed with attention weights, making them robust to compression. This matches the theoretical finding by scos-lab of up to 182× K/V magnitude disparity in Qwen models.

**4. Direct recall is unaffected by any config.** With no filler, all configs score 13/13 (100%). TurboQuant does not corrupt the KV cache for short contexts.

**5. Scalar arithmetic is immune at all distances.** All 15 scalar tests pass across all cache types and filler levels. Simple numeric recall is robust to KV quantization.

**6. Matrix multiplication degrades with context pressure (symmetric TQ only).** Complex multi-step computation (3×3 matrices, 1000t filler) is where TQ3_0 shows the most degradation (-2/3 at 1000t). Asymmetric q8k/tq4v is immune.

**7. TQ4_0 and TQ3_0 symmetric configs have near-identical accuracy.** The 1-bit difference costs only 2 additional tests (49 vs 47 out of 52).

---

## Recommendation

For production use on NVIDIA GPU with EULLM:

```bash
./eullm-tq run model.gguf \
  --cache-type-k q8_0 \
  --cache-type-v tq4_0 \
  --ctx-size 16384
```

This gives F16-equivalent quality and speed with 38% less KV VRAM — enabling longer contexts or larger models on the same hardware.

---

## Test methodology

**52 tests per run:**
- 5× 2×2 matrix multiplication (direct + delayed at 200/500/1000t = 20 tests)
- 3× 3×3 matrix multiplication (direct + delayed = 12 tests)
- 5× scalar arithmetic chains (direct + delayed = 20 tests)

**Delayed test format:**
```
A = [[3,1],[4,2]]
B = [[5,7],[6,8]]

[1000 tokens of unrelated text]

Compute A × B. Return ONLY [[a,b],[c,d]], no explanation.
```

The model must recall the matrix values stored in its KV cache from before the filler, then compute the product. This directly stresses KV cache precision.

**Verification:** Expected answers computed by the test harness in Python, compared against model output with exact normalized matching and LaTeX matrix extraction fallback.

---

## Reproduce

```bash
# Asymmetric (recommended)
./eullm-tq run model.gguf \
  --cache-type-k q8_0 --cache-type-v tq4_0 \
  --ctx-size 16384

# Symmetric TQ4_0
./eullm-tq run model.gguf \
  --cache-type-k tq4_0 --cache-type-v tq4_0 \
  --ctx-size 16384

# Run benchmark
python bench/turboquant_math_accuracy.py collect \
  --label my_test \
  --no-think \
  --num-predict 2048 \
  --output results.json
```

*Test suite: [bench/turboquant_math_accuracy.py](../bench/turboquant_math_accuracy.py)*
*Engine: EULLM v0.3.3 with TurboQuant (spiritbuun CUDA fork, feature/turboquant-kv-cache)*
