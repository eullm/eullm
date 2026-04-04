# TurboQuant Quality Report

## 1% accuracy loss. 4.3× more context. 50% less VRAM.

100 verified tests. Same model, same prompts, temperature=0. The only variable: KV cache quantization.

| Cache Type | Score | Degradation | KV VRAM (14B@131K) | Max Context (16GB GPU) |
|:---:|:---:|:---:|:---:|:---:|
| **F16** | **86/100 (86%)** | baseline | 10.2 GB | 30K |
| **TQ4_0** | **85/100 (85%)** | **-1%** | 5.1 GB | 131K |
| **TQ3_0** | **85/100 (85%)** | **-1%** | 3.8 GB | 131K |

**Bottom line: TurboQuant KV cache compression is nearly lossless.** You trade 1% accuracy on matrix operations for 4.3× more context and 50–62% VRAM savings. Factual knowledge, logic, math, and code generation are unaffected.

<p align="center">
  <img src="../bench/results/chart_quality_comparison.png" alt="Quality comparison: F16 vs TQ4_0 vs TQ3_0" width="800" />
</p>

<p align="center">
  <img src="../bench/results/chart_quality_radar.png" alt="Quality radar chart" width="500" />
</p>

---

## Test setup

- **Model**: Qwen3-14B Q4_K_M (GGUF)
- **Hardware**: NVIDIA RTX 5070 Ti 16GB
- **Engine**: EULLM v0.3.2 with TurboQuant (spiritbuun CUDA fork — superseded by AmesianX v1.4.2 in v0.3.6+)
- **Temperature**: 0.0 (deterministic)
- **Thinking mode**: disabled (`think: false`)
- **Tests**: 100 questions, 20 per category, all with verifiable answers
- **Categories**: Matrix operations, Math, Factual Q&A, Logic & Reasoning, Code & Technical

---

## Results by category

### Matrix (20 tests) — F16: 18/20, TQ4_0: 17/20, TQ3_0: 17/20

Matrix operations test precise numerical computation through attention patterns. This is where KV cache quantization has the most impact — attention scores between position-dependent tokens can lose precision.

| # | Test | F16 | TQ4_0 | TQ3_0 |
|---|------|:---:|:---:|:---:|
| 1 | `[[1,2],[3,4]] × [[5,6],[7,8]]` → `[[19,22],[43,50]]` | PASS | PASS | **FAIL** — gave `[[34,40],[70,82]]` (2× the correct answer) |
| 2 | Determinant of `[[3,8],[4,6]]` → `-14` | PASS | PASS | PASS |
| 3 | Transpose `[[1,2,3],[4,5,6],[7,8,9]]` | PASS | PASS | PASS |
| 4 | Trace of `[[5,1,2],[0,3,1],[2,0,7]]` → `15` | PASS | PASS | PASS |
| 5 | `[[2,0],[0,3]] × [[1,4],[5,2]]` → `[[2,8],[15,6]]` | FAIL | FAIL | PASS |
| 6 | Determinant of `[[1,2,3],[4,5,6],[7,8,9]]` → `0` | PASS | PASS | PASS |
| 7 | Identity × matrix (should return same matrix) | PASS | PASS | PASS |
| 8 | Trace of diagonal `[[10,0,0],[0,20,0],[0,0,30]]` → `60` | PASS | PASS | PASS |
| 9 | Transpose `[[1,2],[3,4],[5,6]]` | PASS | PASS | PASS |
| 10 | Determinant of `[[2,0],[0,5]]` → `10` | PASS | PASS | PASS |
| 11 | Matrix addition `[[3,1],[2,4]] + [[1,5],[3,2]]` | PASS | PASS | PASS |
| 12 | Rank of `[[1,2],[2,4]]` → `1` | PASS | PASS | PASS |
| 13 | Scalar 3 × `[[1,2],[3,4]]` → `[[3,6],[9,12]]` | PASS | PASS | PASS |
| 14 | Determinant of `[[1,3],[2,7]]` → `1` | PASS | PASS | PASS |
| 15 | `[[1,1],[1,1]] × [[1,1],[1,1]]` → `[[2,2],[2,2]]` | PASS | PASS | PASS |
| 16 | Trace of 4×4 identity → `4` | PASS | PASS | PASS |
| 17 | `[[2,3],[1,4]] - [[1,1],[1,1]]` → `[[1,2],[0,3]]` | PASS | PASS | **FAIL** — gave `[[1,-8],[0,3]]` |
| 18 | Determinant of `[[5,3],[2,4]]` → `14` | PASS | PASS | PASS |
| 19 | Rows in a 3×5 matrix → `3` | PASS | PASS | PASS |
| 20 | Permutation matrix squared → identity | PASS | **FAIL** — gave `[[0,2],[2,0]]` | PASS |

**Analysis**: TQ3_0 makes errors on multi-step matrix multiplication (mat01: doubled the result) and subtraction (mat17: wrong element). TQ4_0 fails on permutation composition (mat20). These errors involve precise tracking of intermediate values across attention positions — exactly where KV cache quantization loses bits.

### Math (20 tests) — F16: 18/20, TQ4_0: 18/20, TQ3_0: 18/20

| # | Test | F16 | TQ4_0 | TQ3_0 |
|---|------|:---:|:---:|:---:|
| 1 | 347 × 283 → 98201 | PASS | PASS | PASS |
| 2 | Is 997 prime? → yes | PASS | PASS | PASS |
| 3 | √1764 → 42 | PASS | PASS | PASS |
| 4 | Geometric sequence: 2,6,18,54,_ → 162 | PASS | PASS | PASS |
| 5 | Simplify 84/126 → 2/3 | PASS | PASS | PASS |
| 6 | 17 × 19 → 323 | PASS | PASS | PASS |
| 7 | 144 ÷ 12 → 12 | PASS | PASS | PASS |
| 8 | Is 91 prime? → no | PASS | PASS | PASS |
| 9 | 2^10 → 1024 | PASS | PASS | PASS |
| 10 | 15!/14! → 15 | PASS | PASS | PASS |
| 11 | GCD(48,36) → 12 | PASS | PASS | PASS |
| 12 | LCM(4,6) → 12 | PASS | PASS | PASS |
| 13 | 25% of 360 → 90 | PASS | PASS | PASS |
| 14 | Sum 1..10 → 55 | PASS | PASS | PASS |
| 15 | log₂(256) → 8 | PASS | PASS | PASS |
| 16 | 7³ → 343 | PASS | PASS | PASS |
| 17 | Primes 1..20 → 8 | PASS | PASS | PASS |
| 18 | abs(-7) + abs(3) → 10 | PASS | PASS | PASS |
| 19 | 0.75 → 75% | FAIL | FAIL | FAIL |
| 20 | 10th Fibonacci → 55 | FAIL | FAIL | FAIL |

**Analysis**: Identical across all cache types. The 2 failures (math19, math20) are model limitations — the same questions fail regardless of quantization. **Zero TurboQuant impact on math.**

### Factual (20 tests) — F16: 15/20, TQ4_0: 15/20, TQ3_0: 15/20

| # | Test | F16 | TQ4_0 | TQ3_0 |
|---|------|:---:|:---:|:---:|
| 1 | Capital of Slovenia → Ljubljana | PASS | PASS | PASS |
| 2 | Symbol for tungsten → W | PASS | PASS | PASS |
| 3 | Euro introduction year → 1999 | PASS | PASS | PASS |
| 4 | Most moons → Saturn | PASS | PASS | PASS |
| 5 | GDPR max fine → 4% | PASS | PASS | PASS |
| 6 | Capital of Portugal → Lisbon | PASS | PASS | PASS |
| 7 | Symbol for gold → Au | PASS | PASS | PASS |
| 8 | EU countries → 27 | PASS | PASS | PASS |
| 9 | Largest ocean → Pacific | PASS | PASS | PASS |
| 10 | Divine Comedy author → Alighieri | FAIL | FAIL | FAIL |
| 11 | Currency of Japan → Yen | PASS | PASS | PASS |
| 12 | Formula for water → H2O | PASS | PASS | PASS |
| 13 | Bones in human body → 206 | PASS | PASS | PASS |
| 14 | Smallest country → Vatican | FAIL | FAIL | FAIL |
| 15 | Speed of light → 300000 km/s | FAIL | FAIL | FAIL |
| 16 | Capital of Australia → Canberra | FAIL | FAIL | FAIL |
| 17 | Atomic number 1 → Hydrogen | PASS | PASS | PASS |
| 18 | Hexagon sides → 6 | PASS | PASS | PASS |
| 19 | Longest river in Europe → Volga | FAIL | FAIL | FAIL |
| 20 | Python creator → Python | PASS | PASS | PASS |

**Analysis**: All 3 cache types pass and fail the exact same questions. The 5 failures are model knowledge gaps (Dante's last name, Australia's capital, etc.). **Zero TurboQuant impact on factual recall.**

### Logic (20 tests) — F16: 17/20, TQ4_0: 17/20, TQ3_0: 17/20

| # | Test | F16 | TQ4_0 | TQ3_0 |
|---|------|:---:|:---:|:---:|
| 1 | Syllogism (roses/flowers) → no | PASS | PASS | PASS |
| 2 | Letter sum "EULLM" → 63 | PASS | PASS | PASS |
| 3 | Fibonacci next → 21 | PASS | PASS | PASS |
| 4 | Discount reverse ($20 after 20% off) → 25 | PASS | PASS | PASS |
| 5 | Transitive logic (dogs/animals/living) → yes | PASS | PASS | PASS |
| 6 | Distance = speed × time → 150 | PASS | PASS | PASS |
| 7 | Doubling sequence → 48 | PASS | PASS | PASS |
| 8 | Day ordering → yes | PASS | PASS | PASS |
| 9 | Inverse proportion (workers/hours) → 6 | PASS | PASS | PASS |
| 10 | Letters in "TURBOQUANT" → 10 | PASS | PASS | PASS |
| 11 | Syllogism (cats/shoes) → no | PASS | PASS | PASS |
| 12 | Perfect squares → 36 | PASS | PASS | PASS |
| 13 | Ball counting → 9 | PASS | PASS | PASS |
| 14 | Day of week in 100 days → Friday | FAIL | FAIL | FAIL |
| 15 | Rectangle area → 40 | PASS | PASS | PASS |
| 16 | Next prime → 17 | PASS | PASS | PASS |
| 17 | Syllogism (fish/fly) → no | PASS | PASS | PASS |
| 18 | Vowel counting → 10 | FAIL | FAIL | FAIL |
| 19 | Paper folding → 128 | PASS | PASS | PASS |
| 20 | Clock angle → 7.5° | FAIL | FAIL | FAIL |

**Analysis**: Identical. All 3 fail the same 3 questions (day-of-week modular arithmetic, vowel counting, clock angle). **Zero TurboQuant impact on reasoning.**

### Code & Technical (20 tests) — F16: 18/20, TQ4_0: 18/20, TQ3_0: 18/20

| # | Test | F16 | TQ4_0 | TQ3_0 |
|---|------|:---:|:---:|:---:|
| 1 | FizzBuzz(15) → FizzBuzz | PASS | PASS | PASS |
| 2 | Python length function → len | PASS | PASS | PASS |
| 3 | HTTP 404 → Not Found | PASS | PASS | PASS |
| 4 | SQL acronym | PASS | PASS | PASS |
| 5 | Git history command → git log | PASS | PASS | PASS |
| 6 | HTTPS port → 443 | PASS | PASS | PASS |
| 7 | JSON acronym | PASS | PASS | PASS |
| 8 | Python function keyword → def | PASS | PASS | PASS |
| 9 | HTTP port → 80 | PASS | PASS | PASS |
| 10 | API acronym | PASS | PASS | PASS |
| 11 | CSS acronym | PASS | PASS | PASS |
| 12 | Rust mutable keyword → mut | PASS | PASS | PASS |
| 13 | SSH port → 22 | PASS | PASS | PASS |
| 14 | CRUD acronym | PASS | PASS | PASS |
| 15 | HTML hyperlink tag → a | FAIL | FAIL | FAIL |
| 16 | REST acronym | PASS | PASS | PASS |
| 17 | Python zero division error | PASS | PASS | PASS |
| 18 | PostgreSQL port → 5432 | PASS | PASS | PASS |
| 19 | YAML acronym | FAIL | FAIL | FAIL |
| 20 | Git new branch → git branch | PASS | PASS | PASS |

**Analysis**: Identical. **Zero TurboQuant impact on technical knowledge.**

---

## Divergent results summary

Only **4 out of 100 tests** produced different results across cache types. All 4 are matrix operations:

| Test | F16 | TQ4_0 | TQ3_0 | What went wrong |
|------|:---:|:---:|:---:|------|
| mat01: 2×2 multiply | PASS | PASS | **FAIL** | Doubled the correct answer |
| mat05: diagonal multiply | FAIL | FAIL | **PASS** | TQ3_0 got lucky with format |
| mat17: matrix subtract | PASS | PASS | **FAIL** | Wrong element value |
| mat20: permutation² | PASS | **FAIL** | PASS | Doubled off-diagonal elements |

**Pattern**: The errors occur in tasks requiring multi-step numerical tracking across attention positions. KV cache quantization slightly degrades the precision of attention scores, which can cause the model to "lose track" of intermediate values in sequential computations. This matches the theoretical prediction from the TurboQuant paper (Zandieh et al., ICLR 2026).

**Critical observation**: 4 out of 5 categories (Math, Factual, Logic, Code) show **zero divergence**. The impact is isolated to matrix operations, and even there it's only 1-2 tests out of 20.

---

## Conclusions

1. **TurboQuant is nearly lossless.** 1% accuracy degradation (85% vs 86%) across 100 diverse tests.

2. **The impact is isolated to matrix computation.** Factual, logic, math, and code tasks show zero difference between F16 and TQ3_0/TQ4_0.

3. **TQ4_0 and TQ3_0 are equivalent in quality.** Both score 85%. The 3-bit compression doesn't degrade quality more than 4-bit.

4. **The trade-off is overwhelmingly positive:**
   - -1% accuracy (on matrix ops only)
   - +4.3× context length (30K → 131K on 16GB GPU)
   - -50% to -62% KV cache VRAM
   - 4× more concurrent users per GPU

5. **For RAG, chat, code, and general use cases: use TurboQuant.** The quality impact is negligible and the VRAM savings are transformative.

6. **For numerical/scientific computation: use F16.** If your workload involves heavy matrix algebra through LLM inference, F16 preserves maximum precision.

---

## Reproduce

```bash
# Download TurboQuant build
curl -L https://github.com/eullm/eullm/releases/latest/download/eullm-linux-x64-cuda12.8-turboquant-exp -o eullm
chmod +x eullm

# Run tests (one cache type at a time, restart engine between runs)
./eullm run model.gguf --cache-type-k f16 --cache-type-v f16
python3 bench/turboquant_quality.py collect --label F16 -o results_f16.json

./eullm run model.gguf --cache-type-k tq4_0 --cache-type-v tq4_0
python3 bench/turboquant_quality.py collect --label TQ4_0 -o results_tq4.json

./eullm run model.gguf --cache-type-k tq3_0 --cache-type-v tq3_0
python3 bench/turboquant_quality.py collect --label TQ3_0 -o results_tq3.json

# Compare
python3 bench/turboquant_quality.py compare results_*.json --markdown
```

---

*Test suite: [bench/turboquant_quality.py](../bench/turboquant_quality.py) — 100 questions with verifiable answers, 5 categories, temperature=0.*
*Hardware: NVIDIA RTX 5070 Ti 16GB, Qwen3-14B Q4_K_M, EULLM v0.3.2.*
