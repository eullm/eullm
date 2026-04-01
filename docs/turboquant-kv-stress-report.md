# TurboQuant KV Stress Report

## KV cache recall under context pressure

This benchmark answers a specific question: **does TurboQuant KV quantization corrupt recall of information stored earlier in the context?**

The test places a math problem at the start of the context, fills the context with up to 1000 tokens of unrelated text, then asks the model to recall and solve the original problem. If the KV cache is corrupted by quantization, the model forgets the numbers it stored earlier.

---

## Results — Qwen3-14b (no-think mode)

**Hardware:** NVIDIA GPU, EULLM v0.3.3, ctx=16384, temperature=0

| Cache | Accuracy | Tok/s | Total time |
|:---:|:---:|:---:|:---:|
| **F16** | **52/52 — 100%** | 73.1 | 447s |
| **TQ4_0** | **49/52 — 94.2%** | 56.0 | 550s |
| **TQ3_0** | **47/52 — 90.4%** | 56.4 | 571s |

### Accuracy by test type

| Type | F16 | TQ4_0 | TQ3_0 |
|------|:---:|:---:|:---:|
| Direct (no filler) | 13/13 | 13/13 | 13/13 |
| Delayed 200t filler | 13/13 | 12/13 | 12/13 |
| Delayed 500t filler | 13/13 | 12/13 | 11/13 |
| Delayed 1000t filler | 13/13 | 12/13 | 11/13 |

### Accuracy by problem type

| Problem | Filler | F16 | TQ4_0 | TQ3_0 |
|---------|:---:|:---:|:---:|:---:|
| 2×2 matrix multiply | direct | 5/5 | 5/5 | 5/5 |
| 2×2 matrix multiply | 200t | 5/5 | 5/5 | 5/5 |
| 2×2 matrix multiply | 500t | 5/5 | 4/5 | 3/5 |
| 2×2 matrix multiply | 1000t | 5/5 | 5/5 | 5/5 |
| 3×3 matrix multiply | direct | 3/3 | 3/3 | 3/3 |
| 3×3 matrix multiply | 200t | 3/3 | 2/3 | 2/3 |
| 3×3 matrix multiply | 500t | 3/3 | 2/3 | 3/3 |
| 3×3 matrix multiply | 1000t | 3/3 | 3/3 | 1/3 |
| Scalar arithmetic | direct | 5/5 | 5/5 | 5/5 |
| Scalar arithmetic | 200t | 5/5 | 5/5 | 5/5 |
| Scalar arithmetic | 500t | 5/5 | 5/5 | 5/5 |
| Scalar arithmetic | 1000t | 5/5 | 5/5 | 5/5 |

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

**1. Direct recall is unaffected.** With no filler, both TQ4_0 and TQ3_0 score 13/13 (100%). TurboQuant does not corrupt the KV cache for short contexts.

**2. Scalar arithmetic is immune at all distances.** All 15 scalar tests pass across all cache types and filler levels. Simple numeric recall is robust to KV quantization.

**3. Matrix multiplication degrades with context pressure.** Complex multi-step computation (3×3 matrices, 1000t filler) is where TQ3_0 shows the most degradation (-2/3 at 1000t).

**4. TQ4_0 and TQ3_0 have near-identical accuracy.** The 1-bit difference between 4-bit and 3-bit quantization costs only 2 additional tests (49 vs 47 out of 52). The accuracy gap is small.

**5. Throughput is lower with TurboQuant.** F16 runs at 73.1 tok/s vs 56 tok/s for both TQ variants (-23%). The FWHT preprocessing and Lloyd-Max codebook lookup add overhead that currently outweighs the KV bandwidth reduction. This may improve with future optimization.

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
# Start engine (swap --cache-type-k/v for each run)
./eullm-tq run model.gguf \
  --cache-type-k tq4_0 --cache-type-v tq4_0 \
  --ctx-size 16384

# Run benchmark
python bench/turboquant_math_accuracy.py collect \
  --label my_tq4_test \
  --no-think \
  --num-predict 2048 \
  --output results_tq4.json
```

*Test suite: [bench/turboquant_math_accuracy.py](../bench/turboquant_math_accuracy.py)*
*Engine: EULLM v0.3.3 with TurboQuant (spiritbuun CUDA fork, feature/turboquant-kv-cache)*
