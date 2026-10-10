# Allocation plan — EHPC-DEV-2026D09-278 (LUMI-G)

> 4,500 node-hours (18,000 GPU-hours) on LUMI-G, `project_465003366`,
> **12-09-2026 → 12-03-2027**. EuroHPC Development Access for the **Engine**,
> not for Forge. What the machine is, how the ROCm build works and what was
> measured on 12-09-2026: [`lumi-g.md`](lumi-g.md). The sibling plan for the
> Leonardo AI-Factory allocation, whose arithmetic this one repeats:
> [`../leonardo-allocation-plan.md`](../leonardo-allocation-plan.md).
>
> Updated 05-10-2026. Last measured consumption: **0.0%** (12-09-2026) —
> `tools/lumi/status.sh` recounts it from `sacct`.

## The number that governs everything

Six months against 4,500 node-hours is one whole node busy every day. The
first of those months went by with almost nothing spent, so from 05-10-2026:

| | |
|---|---:|
| calendar elapsed | 12.6% (23 of 181 days) |
| straight-line target to date | ~570 node-h |
| spent | ~0 |
| days left | 158 |
| needed from now | **28.5 node-h/day = 1.2 nodes around the clock** |

**One node is no longer enough.** Queue waits come off whatever is running,
so the campaign runs on **two nodes in parallel** until the deficit is
recovered (`status.sh` shows when), then one.

Unused budget is the outcome that has to be explained afterwards; used budget
explains itself if every hour left a measurement behind. That is the design
rule for everything below: the node is never idle, and nothing runs on it
that does not produce a result file with its provenance.

Two facts about LUMI shape how:

- **Single-GCD work cannot spend this allocation.** `small-g` and `dev-g`
  bill 0.125 node-hours per GCD-hour: one GCD busy until March is ~470
  node-hours, a tenth of the budget. The rest has to be whole nodes on
  `standard-g` (48 h walltime).
- **A whole node bills all eight GCDs** whatever runs on it. A job measuring
  one configuration at a time — what every script before the campaign runner
  did — pays for seven idle GCDs.

`small-g` billing: the charge is the largest of GCDs, cores/8 and memory/64
GB. A one-GCD job asking for 16 cores or 128 GB pays for two.

## How the node is kept busy

[`bench/campaign/`](../../bench/campaign/README.md) — tested against a
stand-in engine in CI, run on LUMI by `tools/lumi/sbatch_campaign.slurm`:

- **A queue of points on scratch**, expanded from campaign specs
  (`tools/lumi/campaigns/`). Several jobs drain it at once — two nodes now —
  and a job that hits the walltime puts its unfinished points back.
- **Every GCD always has a point.** Single-GCD points run eight at a time;
  wider points take aligned groups (a pair on one MI250X module, half the
  node, all of it); the next point starts the moment devices free up. A wide
  point that is waiting reserves its devices and narrower ones only take them
  meanwhile if they will be done in time.
- **Workload points stretch to fill.** Sustained-load points run between a
  minimum and a maximum duration and take exactly the time that is free,
  which is what keeps the end of each 48-hour job from being billed idle.
- **Every server is pinned** to the seven cores LUMI documents as closest to
  its GCD, and every result records the cores, the other points on the node
  at the time, and `neighbours-control` measures the same points alone — so
  packing is itself measured rather than assumed harmless.
- **Each job leaves its own usage evidence**: `<job>.summary.json`, the share
  of the job each GCD had work, and `<job>.node.jsonl`, the raw HBM and
  utilisation samples. That is the page of the Final Report that says the
  hours were used.

Workload points are not filler. They send the public, pinned GSM8K, ARC-Easy
and ARC-Challenge sets at a fixed concurrency for hours, grade the first
pass, compare every later pass with it (greedy decoding, prompt cache off: an
answer that changes under load is a bug), and record throughput, latency and
HBM per minute. That is objective (6), stability under sustained load, plus
the accuracy each memory and speed configuration costs — the other half of
objective (1) that a throughput number alone does not give.

## What we committed to

The accepted proposal asks five questions and names the metrics it will be
read against (time-to-first-token, prompt-processing and generation
throughput, HBM use, model-loading time, scaling efficiency, stability under
sustained load; dense and MoE; one device to a full node).

| WP | question | campaign | engine work it waits on |
|---|---|---|---|
| 1 | size × quant × context → memory, throughput | c01 dense/moe/long-context, c02 quant | — |
| 2 | batching as concurrency grows | c01 batch axes; chunked prefill before/after | merge `feat/engine-roadmap` |
| 3 | multi-GPU without disproportionate overhead | c01 ref/scale, c02 235B split vs 2×4 | `--split-mode` incl. tensor |
| 4 | CUDA vs ROCm | the same specs on Leonardo / JUPITER | — (runner is site-neutral) |
| 5 | loading and on-prem transfer | cold/warm load in every result; c02 large MoE | `--moe-cache` on HIP |
| 6 | stability under sustained load | soak groups, 2-24 h | `/metrics` (nice to have) |
| 7 | multi-node ("may be evaluated") | c04, bounded | RPC backend build |

## Campaigns and the budget

`campaign.py plan` prints what each spec costs at most:

| spec | points | node-h | needs |
|---|---:|---:|---|
| `c01-node-baseline` | 205 | ~115 | catalog models only: **runs today** |
| `c02-quant-large-moe` | 75 | ~130 | ~1.2 TB pulled from Hugging Face first |
| ~~`c05-finetune`~~ | — | — | withdrawn from LUMI on 06-10-2026, see below |
| `c06-mtp` | 48 | ~4 | MTP GGUFs (~114 GiB): speculative decoding, objective (5) |
| `c07-runtimes` | 60 | ~3 | llama-server and Ollama (`install_runtimes.sh`): the reference frame for (1) and (2) |
| `c08-decision` | 128 | ~4 | Jev-Style releases: `/v1/systemone` under concurrency, objective (2) for decisions |
| `c09-load` | 22 | ~8 | cold loads from Lustre with and without `--load-threads`, objective (5); a runner of 07-10-2026 or later |

Measured honestly, the matrix the proposal describes is cheap: ~250
node-hours a pass, nearly all of it the soaks. What spends 4,500 is doing it
**again for every engine change** — which is the development cycle the
proposal describes ("frequent releases and daily code iterations… experiments
will track exact Git revisions"). A **round** is the same specs planned under
a new label (`ROUND=<engine version> campaign_setup.sh`): every point measured
again, so each engine change has a before and an after on identical
workloads.

| item | node-h |
|---|---:|
| round 1: c01 + c02 on the released 0.7.20 build | ~250 |
| rounds on engine milestones, ~1 every 9 days (~14 × ~250) | ~3,500 |
| c03: row and tensor split, replicas × split, once the engine exposes them | ~300 |
| c04: multi-node exploration, 2-4 nodes, bounded | ~300 |
| iteration on small-g (dev-g for smoke tests of minutes only), reserve | ~150 |
| **total** | **4,500** |

A round needs a reason: a release, a merged branch that touches inference, a
llama.cpp bump, a build flag (RCCL, HIP graphs) — or, once, a deliberate
same-binary repeat to measure day-to-day and node-to-node spread. The soak
`max_duration_s` is the other lever if rounds run short of the pace.

## Engine work, before the hours (no allocation needed)

The four gaps [`lumi-g.md`](lumi-g.md) found by reading the code are still
open on `main` at 0.7.20, and each one ends a round with a before/after.

0. **Merge `feat/engine-roadmap`.** It is 22 commits ahead of `main` and
   carries measured prompt and answer times in every response (`5febfeb` — on
   `main`, `prompt_eval_duration` is hard-coded to 0, so the runner's
   server-side prefill rate is null until this lands) and chunked prefill
   between decode steps (`04d08ca`, roadmap 0.7-D), an objective (2) result.
1. **The banner must report the live backend, not the compiled one.** A
   `rocm` binary that finds no device still prints `GPU backend: ROCm` and
   then bills node-hours at CPU speed. The runner's `--fit-strict` default
   catches the out-of-memory side of this; nothing catches a missing device.
2. **`--split-mode`, `--tensor-split`, `--main-gpu`, device selection in
   `RuntimeOpts`.** The vendored bindings expose all of it, including
   llama.cpp's **experimental `Tensor` split** — real tensor parallelism, the
   one arrangement that could make a *single* request faster across GCDs. The
   12-09 conclusion that splitting adds no throughput was measured on layer
   split only. This is c03.
3. **The harness (WP0)** — done in `bench/campaign`: one schema
   (`eullm.bench/1`), TTFT, prefill and decode rates, cold/warm load, HBM per
   device, repeats with their spread, and provenance (engine version and
   binary hash, repository revision, ROCm version, cores, neighbours). It
   runs on CUDA unchanged (`--backend cuda --bind none`); what is missing is
   the `sbatch` wrapper for Leonardo and JUPITER, a copy of
   `sbatch_campaign.slurm` with their partitions. A `/metrics` endpoint
   (roadmap 0.7-B) would let a soak see queue depth over time, not just at
   the end.
4. **A placement policy: the smallest split that fits, then replicas.** The
   12-09 numbers give the rule (replicas 4.07× on four GCDs, layer split
   0.98×); c01 and c02 measure it on more models. Turning it into `--fit` on
   a per-device plan, with the engine serving N pinned replicas behind one
   endpoint, is roadmap 1.0-B arriving early, for a measured reason.
5. **NUMA binding** — the runner already pins every server; a round with
   `--bind none` is the before/after that says whether the engine should do
   it itself.
6. **An RCCL build**, `GGML_HIP_RCCL=ON`, and `GGML_HIP_GRAPHS`: experiment
   binaries, each one a round, never the default.
7. **`--moe-cache` on HIP.** It refuses any non-CUDA device and more than one
   GPU, and its pinned-memory patch is CUDA-only. Objective (5) on MoE.
8. **Loading from Lustre: `--load-threads`** (07-10-2026, `c09`). c02's
   Coder-480B (270 GiB) had not loaded after an hour, and `dd` reading 2 GiB
   pieces of it on a compute node gave 178 MB/s with one stream and 2,283
   MB/s with sixteen (`sbatch_lustre_probe.slurm`). Threads reading the model
   ahead of llama.cpp did not pay: cold, a 132 GiB model loaded in 110 s
   without them and 128 s with 16, since llama.cpp alone reads at 0.7-1.3
   GB/s. The flag is off by default. The hour lost on the 480B was the file
   being mapped: watched on a whole node (`sbatch_load_watch.slurm`, 07-10 and
   08-10-2026), Lustre read it at about 2 GB/s and the VRAM was reserved
   after 3 minutes, then one thread at 95% of a core copied it to the GPUs at
   about 90 MB/s, still unfinished after 70 minutes, the page cache stuck near
   half the RAM. With `--no-mmap` the same model loaded in 151.8 s. c09's
   `load-nommap` groups measured it on every model (cold, seconds):

   | model, GCDs | mapped | read in |
   |---|---:|---:|
   | Qwen3.8-27B Q8, 29 GiB, 1 | 46 | 51 |
   | Qwen3-235B Q4_K_M, 132 GiB, 4 | 110 | 108 |
   | Qwen3-Coder-480B Q4_K_M, 270 GiB, 8 | failed 6 of 6 | 160, 208 |
   | DeepSeek-V3.1 Q4_K_M, 8 | failed 3 of 3 | 226 |

   The tokens per second were the same both ways. The engine now reads a
   model in by itself when it goes to the GPUs whole and its files are more
   than half the memory the process may use (`fit::read_in_whole_on_gpu`);
   `--mmap` keeps it mapped. The `moe-` groups of c02 that failed loading can
   be planned again on that engine.

Items 0-2 are October. Items 4-6 December. Item 7 January.

## The CUDA half

Leonardo was not awarded for this project. In order:

1. **Leonardo (again) and JUPITER**, requested by the PI in early October,
   decision pending. The campaign runner and the specs run there as they are;
   only the `sbatch` wrapper changes.
2. **Leonardo AI-Factory, before 02-11-2026.** Measuring how the legal-it
   GGUFs that allocation produced serve on an A100 is evaluation of its own
   output, and its queue needs filling too.
3. **The A100 numbers already held** ([`../cineca/leonardo.md`](../cineca/leonardo.md)),
   measured by hand on 04-09-2026 before two llama.cpp bumps.
4. **The RTX 5070 Ti** — the "smaller on-premises system" of objective (5).

## Models

Permissive licences only, as everywhere in the project, checked against the
Hub on 05-10-2026: Qwen and gpt-oss Apache-2.0, DeepSeek-V3.1 MIT. No Llama.

- **c01, catalog** (Q4_K_M): Qwen3 4B/8B/14B/32B, Mistral-Small-24B,
  Qwen3.6-27B, Qwen3.6-35B-A3B; plus the 12-09 reference model,
  `unsloth/Qwen3.8-27B-GGUF` UD-Q8_K_XL (29.3 GiB), so its rows repeat
  exactly. It was gone from LUMI by 05-10 and is pulled again; the engine now
  names it `qwen3.8-27b-gguf-ud-q8_k_xl`, the 12-09 rows
  `qwen3.8-27b-ud-q8_k_xl` — the same file.
- **c02, from Hugging Face**: Qwen3 8B/14B/32B Q8_0 against c01's Q4_K_M;
  Qwen3-30B-A3B Q4_K_M and Q8_0; gpt-oss-20b and -120b (MXFP4, 11 and 59
  GiB); Qwen3-235B-A22B Q4_K_M (132 GiB: 4 GCDs, so one split or two
  replicas of it) and Q8_0 (233 GiB); Qwen3-Coder-480B-A35B Q4_K_M (270 GiB);
  DeepSeek-V3.1 Q4_K_M (378 GiB — only a whole node runs it at all).

About 1.3 TB on `/scratch` in total, inside the 4 TB asked for. Compute nodes
have no network: `campaign_setup.sh` pulls on a login node and leaves the
missing ones blocked, never failed.

The workload sets are GSM8K, ARC-Easy and ARC-Challenge (4,867 graded
questions). MMLU is not: its pinned source,
`people.eecs.berkeley.edu/~hendrycks/data.tar`, answers 404 as of
05-10-2026 — which also breaks `sbatch_autobench.slurm`'s default `SETS`.

## Calendar

Paced on 28.5 node-hours a day to 12-03-2027.

| month | engine | on the machine | target node-h |
|---|---|---|---:|
| Oct | merge `feat/engine-roadmap`, live banner, split controls, Leonardo/JUPITER wrapper | **c01 now on two nodes**; c02 as its models arrive; round 2 on the merged roadmap branch | 750 |
| Nov | anomalies: KV quant that gains nothing on gfx90a, batching that pays 3.6× here and 1.7× on A100 | c03 split modes; rounds on each release; the CUDA half if granted | 850 |
| Dec | placement policy, served replicas, RCCL and HIP-graphs binaries | before/after rounds; **long soaks queued 20-12 → 06-01**, when nobody is watching and the queue still runs | 880 |
| Jan | `--moe-cache` on HIP, load path | rounds; c04 multi-node | 880 |
| Feb | release candidate | rounds on it | 800 |
| Mar | release | final round to 12-03; data to Zenodo; Final Report draft | 340 |

## The engine's trainer (`c05-finetune`): withdrawn from LUMI

**Not run on this allocation, as of 06-10-2026.** The JUPITER proposal
(EHPC-AIF-2026PG01-1434, submitted to the same Joint Undertaking at the end
of September) describes this allocation in writing: *"This allocation covers
inference-engine work only; no model training runs on it."* A trainer
benchmark is engine work, but it trains weights, and a statement made to
EuroHPC is not reinterpreted after the fact. The `c05` points were taken off
the queue on 06-10; whatever ran on the night of 05-10, before the conflict
was noticed, is reported as such in the Final Report and not used. The spec
stays in the repository for a machine where training is declared.

What it was, for the record. `eullm finetune` is new engine code: llama.cpp's trainer (ggml-opt), which
the engine did not expose, behind one command that trains an F32 GGUF on a
text file and writes a GGUF the engine serves. `c05` measures it on one GCD
the way the other campaigns measure inference: tokens per second, HBM, and
the does-not-fit boundary by model size (Qwen3 0.6B/1.7B/4B Base), optimizer
(AdamW, SGD), training window (512-2048) and which tensors train (all, or
attention only); then whether the held-out loss falls as it should over
three epochs at three learning rates. The text is GSM8K's training split
(public, MIT); the trained models are deleted when the point ends.

It was planned as software engineering and benchmarking of the runtime, on
public data, with no model as an output; the sentence above settles it.

## Engine work the hours measure (decided 06-10-2026)

With training off this allocation, what fills it is what the proposal asked
for: engine changes, each measured before and after on identical workloads.
In order:

1. **MTP** (`c06`): speculative decoding with the model's own head on
   MI250X, graded at temperature 0 (answers must not change) and at default
   sampling. Objective (5).
2. **The reference frame** (`c07`): the same points on llama-server (same
   llama.cpp commit, without EuLLM's runtime) and on Ollama. What EuLLM's
   scheduler and batching add or cost is the difference. vLLM after a
   like-for-like design (it does not serve GGUF) and a check that its ROCm
   container runs on gfx90a.

   What it found (07-10-2026): level at one request, 8-10% behind at four,
   11% (the MoE) to 36% (8B, 14B) behind at sixteen. The cause, found on
   08-10-2026 with the scheduler's `steps:` lines and
   `tools/lumi/sbatch_concurrency_diag.slurm` (both servers on one GCD, the
   same sampling options): EuLLM's default repeat penalty 1.1 made llama.cpp
   look up all 151,936 tokens of Qwen3's vocabulary for every token of every
   answer, 0.7 ms each on LUMI's CPU, so 11 ms of every step at sixteen;
   llama-server's default has the penalty off. llama.cpp patch 0005 penalizes
   the recent tokens in place (1,196 to 2 µs per token). Qwen3-8B Q4_K_M, one
   GCD, 256 tokens per answer, tokens/s, after the patch (job 22653307):

   | requests | profile | EuLLM | llama-server |
   |---:|---|---:|---:|
   | 1 | each server's defaults | 111 | 112 |
   | 4 | each server's defaults | 221 | 228 |
   | 16 | each server's defaults | 621 | 676 |
   | 16 | penalty 1.1 on both | 630 | 476 |
   | 16 | penalty off on both | 631 | 666 |
   | 16 | greedy | 636 | 701 |

   What is left at sixteen (5-9%): the steps carried 14.9-15.4 sequences on
   average, not 16, because EuLLM reads one waiting prompt per step and a
   round of sixteen starts over sixteen steps, where llama-server reads them
   in one batch; sampling is 12% of a step (0.19 ms per token, on one
   thread). The graded workload's distance (14B 99 against 315 tokens/s) is
   larger than either explains. The likely reason, found on 09-10-2026: with
   a KV cache per slot, llama.cpp runs the model once per run of consecutive
   slot numbers in the order a step lists them (`split_equal`, sequential),
   and EuLLM listed them in the order answers had ended and slots been taken
   again. With requests arriving and ending all the time, as in the graded
   workload, steps took 3.5 to 4 passes on four CPU cores (Qwen3-0.6B, eight
   clients one request after another); in slot order, 1.0 to 1.4, and 60-71
   tokens/s instead of 41-43. The `steps:` lines now count the passes, and the
   diagnosis has closed-loop rounds (`--duration`) and `--kv-unified` to
   measure it on the GCDs.

   On the GCD (job 22663207, 09-10-2026, Qwen3-14B Q4_K_M, the engine with
   both fixes), tokens/s:

   | | EuLLM | EuLLM `--kv-unified` | llama-server |
   |---|---:|---:|---:|
   | 4 clients, one request after another | 145 | 144 | 139 |
   | 16 clients, one request after another | 377 | 391 | 340 |
   | 16 at once, each server's defaults | 449 | | 462 |
   | 16 at once, penalty off on both | 454 | | 456 |
   | 16 at once, penalty 1.1 on both | 454 | | 353 |
   | 16 at once, greedy | 457 | | 474 |

   Closed-loop steps took 1.03-1.10 passes on average (1.00 with
   `--kv-unified`), the prompts read between steps 6-7% of the time.
   The closed loop is a stand-in for the graded workload (short raw prompts,
   answers of 32-384 tokens): a `c07` round on this engine is the check.
   What is left at sixteen at once (3%): the round's prompts start over
   fifteen steps, 15.2 sequences per step on average where llama-server has
   16 from the first. Since 10-10-2026 a step reads the waiting prompts
   itself, beside the answers' tokens, as llama-server does. The diagnosis
   with the engine before it on the same GCD (job 22691743, 10-10-2026,
   Qwen3-14B Q4_K_M, tokens/s, the mean of two rounds at once):

   | | EuLLM before | EuLLM | EuLLM `--kv-unified` | llama-server |
   |---|---:|---:|---:|---:|
   | 16 at once, llama-server's chain on both | 447 | 480 | 447 | 462 |
   | 16 at once, greedy | | 480 | | 472 |
   | 16 at once, each server's defaults | | 469 | | 460 |
   | 16 at once, penalty 1.1 on both | | 478 | | 359 |
   | 16 clients, one request after another | 369 | 366 | 400 | 348 |
   | longest wait for a first token, 16 at once | 0.65-0.84 s | 0.035-0.39 s | 0.034-0.28 s | |
   | longest wait for a first token, 16 clients | 0.94 s | 0.30 s | 0.22 s | |

   A step now carries all sixteen sequences in one pass (16.0 sequences,
   1.00 passes per step) from the round's first. The longer first-token
   waits (0.39 and 0.28 s) are each server's first round, in which the
   prompts are read from scratch; after it the slots hold them. At four,
   all four runs are level (150-153 tokens/s). In the closed loop the
   steps take 1.07-1.17 passes with a KV cache per slot, from the slots
   left idle between a client's requests and the prompts' passes of their
   own; `--kv-unified` has none (400 tokens/s) but reads every slot's cells
   in each answer's attention, which costs it 7% with sixteen at once.

   The `c07` round on that engine (`r-next6`, job 22693068, 10-10-2026, one
   GCD per point, Q4_K_M) against the rounds before it. The graded workload,
   sixteen clients over 400 gsm8k questions, tokens/s (accuracy 0.92-0.96
   everywhere, within the noise of 400 questions):

   | model | EuLLM `r-clean` | EuLLM `r-next3` | EuLLM `r-next6` | llama-server | Ollama |
   |---|---:|---:|---:|---:|---:|
   | Qwen3-8B | 154 | 517 | 533 | 486 | 166 |
   | Qwen3-14B | 102 | 327 | 349 | 317 | 142 |
   | Qwen3-32B | 54 | 186 | 191 | 178 | 105 |
   | Qwen3.6-35B-A3B | 118 | 219 | 221 | 199 | 28 |

   Sixteen requests at once (answers of 150 tokens), tokens/s and the median
   wait for the first token:

   | model | EuLLM `r-next3` | EuLLM `r-next6` | llama-server `r-next6` |
   |---|---:|---:|---:|
   | Qwen3-8B | 613 (205 ms) | 695 (33 ms) | 646 (274 ms) |
   | Qwen3-14B | 423 (306 ms) | 474 (192 ms) | 450 (312 ms) |
   | Qwen3-32B | 234 (600 ms) | 263 (104 ms) | 248 (502 ms) |
   | Qwen3.6-35B-A3B | 244 (445 ms) | 246 (163 ms) | 229 (1963 ms) |

   The distance `c07` found is closed on every model: EuLLM is 7-10% ahead of
   llama-server on the graded workload and 5-8% with sixteen at once, where
   reading the prompts in the step gained 12-13% on the dense models and 1% on
   the MoE. The `r-clean` rows are the engine with the repeat penalty's whole-
   vocabulary scan and the steps out of slot order (0.7.50 fixed both).

   `--kv-unified` against a KV cache per slot (`rt-kvu*`, same job), tokens/s:

   | | Qwen3-8B | Qwen3-14B | Qwen3-32B | Qwen3.6-35B-A3B |
   |---|---:|---:|---:|---:|
   | 4 at once | 232 / 226 | 151 / 149 | 74.5 / 75 | 155 / 154 |
   | 16 at once | 689 / 645 | 472 / 455 | 264 / 246 | 268 / 262 |
   | 16 clients, graded | 531 / 548 | 347 / 362 | 193 / 200 | 218 / 221 |
   | 4 slots of 32k, 16k-token prompts | 13 / 8.1 | 7.5 / 4.1 | | |

   One cache for all gains 1-4% where requests come and go and slots are left
   idle between them, loses 2-7% with sixteen at once, and 38-45% with long
   contexts, where every answer's attention reads all the slots' cells: it
   stays off by default, an option for many short requests.
3. **Decisions** (`c08`): `/v1/systemone` with the Jev-Style releases. The
   engine runs one decision at a time per server; `c08` measures the queueing
   that causes as concurrency grows, and replicas as today's way round it.
   Batching decisions in the engine is the change it is the "before" of:
   since 09-10-2026 `batched` requests that arrive together are evaluated in
   shared decode calls (option 2 of three: only for clients that ask for
   `batched`, so the default stays reproducible bit for bit); a `c08` round on
   that engine is the "after".
4. **`--moe-cache` on HIP** (item 7 above): MoE experts in the node's host
   RAM, the direction Strata takes on consumer GPUs (the head-to-head with
   Strata stays on the RTX 5070 Ti, `docs/moe-offload-plan.md`). Objective
   (5).

Every change that lands in `main` from these is a new round of `c01`+`c02`.

## Lines not to cross

- **This is a development allocation for the engine, inference only.** The
  workload is public benchmark sets, graded; nothing it produces feeds Forge,
  and no training of any kind runs here — not Forge, and not the engine's own
  trainer (`c05`, withdrawn). That is what EHPC-AIF-2026PG01-1434 told
  EuroHPC about this allocation.
- **Every hour leaves a result.** A round without an engine change, a point
  that measures nothing new — those are the hours that are hard to explain.
  More rounds tied to more engine changes are not.

## Final Report

The portal already has the slot (page 11 of the consolidated forms: *Final
Report Upload*). To settle while writing, not after:

- **Licence.** The application says Apache-2.0 in three places; the repository
  is AGPL-3.0-or-later since August 2026. Both open source; the report must
  describe the repository as it is.
- **Where the CUDA numbers came from**, per the section above.
- **The `c05` points that ran on 05-10/06-10** before being withdrawn: how
  many, that they trained small public models on public text as an engine
  benchmark, that the models were deleted, and that the track was stopped
  because of the declaration in EHPC-AIF-2026PG01-1434.
- **The deadline and template**: confirm with EuroHPC; the AI-Factory rule is
  three months after the end (12-06-2027 here), and it is reasonable to
  assume the same.

## Decisions needed

1. **Two nodes in parallel** until the deficit is recovered — the default of
   `submit_campaign.sh`.
2. **The CUDA half on Leonardo AI-Factory** before 02-11, or wait for the
   Leonardo/JUPITER decision.
3. **MMLU**: find a pinned mirror for ReflexBench, or leave it out (the
   campaigns already do).
