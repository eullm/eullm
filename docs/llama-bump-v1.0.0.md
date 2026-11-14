# llama.cpp v1.0.0: what moved, and what it is worth to EuLLM

Branch `chore/llama-cpp-v1.0.0`, from `main` after release 0.8.9. The llama.cpp
submodule moves from b12283 (10 November) to the tag `v1.0.0` (b12359,
`b1b770d`, 13 November): 76 commits. Upstream now tags releases with semantic
versions besides the `bNNNNN` build numbers; v1.0.0 is the first major one.
The 17 commits after it on 13 November are left for the next bump, among them
#32157, which moves the MoE cache's code out of `llama-graph.cpp` where `0003`
patches it, and a retuning of HIP's MMQ for GCN (#30021) worth measuring on
the MI250X then.

## 1. What llama.cpp 1.0 removed

Two pruning PRs, "llama : prune" (#30565) and "ggml : prune" (#29801), drop
what was marked deprecated: 81 functions of `llama.h`, `ggml.h`, `common.h` and
`src/llama-ext.h`, which is gone (its NextN and speculative API moved to
`llama.h`, `llama_set_embd_state_src` renamed `llama_set_ctx_state_src`), and
the `defrag_thold` field of `llama_context_params` with `--defrag-thold`.

EuLLM called ten of them, all in the vendored `llama-cpp-2`; none in the
engine, the C++ wrappers or the patches:

| File | Removed | Now calls |
|---|---|---|
| `model.rs` | `llama_load_model_from_file`, `llama_free_model`, `llama_new_context_with_model`, `llama_n_ctx_train`, `llama_n_embd` | `llama_model_load_from_file`, `llama_model_free`, `llama_init_from_model`, `llama_model_n_ctx_train`, `llama_model_n_embd` |
| `context/session.rs` | `llama_get_state_size`, `llama_copy_state_data`, `llama_set_state_data`, `llama_load_session_file`, `llama_save_session_file` | `llama_state_get_size`, `llama_state_get_data` (no size limit, as the old call), `llama_state_set_data` (bounded by the slice), `llama_state_load_file`, `llama_state_save_file` |
| `context/params/get_set.rs` | the `defrag_thold` field | `with_defrag_thold` and `defrag_thold` removed; the engine used neither |

The Rust methods keep their names, so nothing above `llama-cpp-2` changed.
Upstream llama-cpp-rs had not followed: its `main` tracks llama.cpp v0.9.0
(4 November) and now generates its bindings ahead of time instead of in the
build, a change EuLLM does not take with this bump.

## 2. What EuLLM gains

- **Few-row MMQ** (#31533): quantized products of 2 to 24 tokens get an int8
  MMA kernel of their own (Q8_0, Q3-Q6_K, IQ3_S, IQ4_NL, IQ4_XS; NVIDIA from
  Ampere on, not HIP). That is continuous batching with 2-24 requests and the
  check of `--mtp`'s drafts.
- **BF16 with few rows on MMVF** (#31959) at batch 2-8: the "down" projections
  of the hyper-connections. `0006` does their "up" projections at batch 1; the
  two do not overlap and `0006` stays.
- MoE: MMQ's stream-k fixup skips empty tiles (#32067); upstream's own
  prefetch reads only results of earlier splits, so never an expert the cache
  is still uploading, and the cache turns itself off where its device cannot
  run the matmul (#32100).
- ROCm 7 and newer keep one BLAS workspace (#31838): the RDNA and Windows ROCm
  builds; not LUMI-G (ROCm 6.3).
- Graph results reused across reserves (#32089), unused graph metadata pages
  left untouched (#32043), CUDA graphs re-instantiated when a kernel's cluster
  dims change.
- OCP FP8 on CUDA (#28413), the CPU's flash attention accumulating V in F32
  (#29810), Vulkan flash attention for Intel at prefill (#29357).

## 3. Patches

All twelve apply to v1.0.0 as they were (`cargo test --test llama_patches`:
11 of 11). Upstream touched none of the code they change.

One more, `0015`: the few-row kernel of #31533 can be turned off, by
`GGML_CUDA_MMQ_FEW=0` or by the program (the CUDA backend's proc
`ggml_backend_cuda_set_mmq_few`). EuLLM turns it off where MTP drafts
(section 6).

## 4. Checked here (cloud container, 4 cores, no GPU)

- CPU release build with no warning; `cargo test`: 772 unit tests and the 11
  `llama_patches` tests; `cargo clippy --no-deps -- -D warnings`: nothing.
- `engine/src/llama_archs.rs` regenerated: the same 172 architectures.
- The real-model tests (`--ignored real_model`, release build, CPU): 35 of
  35, with stories260K F32 for generation, Qwen3-0.6B Q8_0 and its embedding
  model, Qwen3.5-0.8B with and without MTP, and Spark-X2.5-1.7B. The two
  decision tests that load a backend of their own ran in a process of their
  own each; the equivalence of the decision modes on the F32 model, as its
  message asks (on Qwen3-0.6B Q8_0 batched and separate evaluation differ by
  up to 0.41 in a log-probability, quantized rounding).

Not checked here: any GPU run, and the CUDA, ROCm, Vulkan and Metal builds.

## 5. To measure before release 0.9.0

On the reference PC (RTX 5070 Ti), 0.8.9 against this branch, as for b12283:
Qwen3.6-35B-A3B with its expert cache, `prefetch_check` on Qwen3-Next-80B and
Flash-Next, DeepSeek-Coder-V2-Lite whole in VRAM, Qwen3.5-9B with `--mtp 2`,
and, for #31533, a dense model answering four and eight requests at once
(`--batch-size 4`/`8`).

On LUMI-G: `tools/lumi/build_engine.sh` and the smoke job.

## 6. Measured on the reference PC (14 November)

RTX 5070 Ti, the 0.8.9 release (`9965640b`) against this branch (`e88bf792`).

| test | 0.8.9 | v1.0.0 |
|---|---|---|
| Qwen3-Next-80B Q4_K_M, prefetch 0: reading / writing | 1,796.7 / 109.1 | 1,789.8 / 108.7 |
| the same, prefetch 4 (20% from the cache's slots) | 1,433.6 / 104.4 | 1,442.9 / 103.4 |
| answer, every setting | `98260681` | `98260681` |
| DeepSeek-Coder-V2-Lite whole in VRAM, 8K prompt: reading / writing | 6,353.5 / 267.1 | 6,333.0 / 267.0 |
| answer | `592ef83a` | `592ef83a` |
| Qwen3.5-9B `--mtp 0`: story / code | 128.0 / 127.9 | 128.5 / 128.3 |
| answers | `e58b246d`/`bbc626da` | `e58b246d`/`bbc626da` |
| Qwen3.5-9B `--mtp 2`: story / code | 182.9 / 227.7 | 169.7 / 219.7 |
| drafts kept | 68% | 61% |
| answers | `e58b246d`/`d24be5ff` | `bb33d7f8`/`c525b89b` |

Level everywhere, to the same answers bit for bit, except `--mtp 2`: 7
points fewer drafts kept, 7% and 4% slower, and the story's answer parts
from the one without drafts, which on 0.8.9 it did not. Two changes of the
release reach that check of three tokens: the few-row MMQ kernel (#31533,
from 3 columns on for Q5_K and Q3_K) and the gated delta net of Qwen3.5,
whose non-fused, ubatch-dependent graph nodes are gone (#32078). `0015`
turns the first off to tell them apart.

`GGML_CUDA_MMQ_FEW=0`, the same build, `--mtp 2`: 181.1 / 225.1 tokens/s,
68% of the drafts kept, answers `e58b246d`/`d24be5ff`, as 0.8.9. The few-row
kernel is the cause: the check of three tokens no longer gets the numbers of
one token, and the drafts it rejects for that are drafts lost.

Several requests at once, Qwen3-14B Q4_K_M (text only: Qwen3.5-9B has a
projector beside it, which turns the engine sequential), `--batch-size 8`,
`bench/stress_test.py`, 200 tokens a request, two rounds, aggregate tokens/s:

| requests | few-row on | off |
|---:|---:|---:|
| 1 | 85.2, 86.0 | 85.6, 86.0 |
| 4 | 300.2, 307.6 | 282.4, 289.7 |
| 8 | 539.0, 569.0 | 498.3, 515.7 |

Each request at 4 and 8 at once: 77.6 against 73.0, 71.7 against 65.0
tokens/s. The kernel stays on, then, and the engine turns it off where MTP
drafts (`inference::few_row_mmq_off_for_mtp`): drafting serves one request at
a time, where the kernel only ever checks drafts. `GGML_CUDA_MMQ_FEW` set
either way is left as it is.

With the engine's switch (`1244c6ea`), no variable set, `--mtp 2`: 180.5 /
224.6 tokens/s, 68% of the drafts kept, answers `e58b246d`/`d24be5ff`, and
the server's log says "MTP: CUDA's few-row kernel off, so drafts are checked
with one token's numbers".

## 7. Measured on JUPITER (GH200, 14 November)

0.8.9 and this branch at `e88bf792` (before `0015`), built for sm_90, c07's
points on Qwen3.6-27B Q4_K_M, tokens/s over all requests: 4 at once 213 ->
306 (+44%), 16 at once 531 -> 717 (+35%), 32 at once 652 -> 734 (+13%). With
`--mtp 2` on Qwen3.5-9B the two are level (story 270.6/271.2, code
334.7/335.4, drafts kept 52%/51% and 80%/78%). Details in
`docs/jupiter/dossier.md`, section 5.12.

## 8. Measured on LUMI-G (MI250X, 14 November)

`tools/lumi/build_engine.sh` built this branch (`d638a51f`) against ROCm 6.3.4
with gfx90a device code and RCCL linked, `0011` and `0013` included; the
smoke job loaded Qwen3-8B whole and answered. `tools/lumi/sbatch_engine_ab.slurm`
(job 23735026), the b12283 build (`487f9185`, 0.8.9's code) and this one on
the same GCD, three runs each of `bench/speed_check.py`: writing 117.0-117.1
against 117.3-117.8 tokens/s, reading 8K tokens 2,076-2,226 against
2,060-2,210, the same answer (`d6447b90`). The smoke's 41.2 tokens/s against
0.8.9's 87.8 was a server's first request, 13-15 tokens/s on both builds.
