# Final Report — AI Factory Playground Access, EHPC-AIF-2026PG01-1147 (as submitted 04/11/2026)

Drafted 03/11/2026 from the EuLLM repository (branch `feat/leonardo-legal-it`).
Every figure carries a source comment; `<missing: …>` marks what the
repository does not hold. Consumption and end-of-allocation storage figures
come from the PI's `saldo -b` / `df` reading of 03/11/2026, not from a file.

---

# 1. GENERAL INFORMATION

Type of project granted: AI Factory Playground Access

## Proposal ID

EHPC-AIF-2026PG01-1147 (Leonardo project account `AIFAC_P02_1147`)

<!-- source: docs/leonardo-allocation-plan.md (title); docs/cineca/leonardo.md §"Cluster" (account) -->

## Awarded resources in GPU hours

5,000 GPU-hours (1,250 node-hours of 4 A100 each; 40,000 local core-hours in
CINECA's `saldo` accounting, 8 cores per GPU).

<!-- source: docs/leonardo-runbook.md header; docs/leonardo-allocation-plan.md header; docs/cineca/fastlane-plan.md §"Le condizioni" (8 core per GPU) -->

## AI technologies used

- [x] Machine Learning
- [x] Natural Language Processing
- [x] Generative Language Modeling
- [x] Deep Learning
- [ ] Vision · [ ] Audio · [ ] Robotic process automation · [ ] Virtual agents · [ ] Decision management · [ ] Other

## Period of access to the EuroHPC JU facilities

| | |
|---|---|
| Overall duration of the project in months | 2 |
| Start date of the allocation | 02/09/2026 |
| End date of the allocation | 02/11/2026 |

<!-- source: docs/leonardo-allocation-plan.md header -->

## EuroHPC JU system assigned

- [x] Leonardo Booster (CINECA)

## Project Lead and Organization

| | |
|---|---|
| Title | Mr. |
| First (Given) Name | Francesco |
| Last (Family) Name | Marchetti |
| E-mail Address | (in the submitted copy) |
| Organization name | I3K Technologies Srl (Milan, Italy) |
| Organization type | Startup |

<!-- source: docs/jupiter/dossier.md header (PI and organisation) -->

---

# 2. PROJECT INFORMATION

## Project title

Validation of EULLM Forge for Sovereign Domain-Specific LLM Training and On-Premises Deployment

<!-- source: application form EHPC-AIF-2026PG01-1147 -->

## Main application domain(s)

- [x] Mathematics and Computer Sciences
- [x] Linguistics, Cognition and Culture

(The application domain is Italian law. Law has no box of its own, so the
two boxes ticked are the method, machine learning, and the subject, Italian
legal language.)

## Team members and organizations

None: the Project Lead carried out the project alone (I3K Technologies Srl), as declared in the application.

<!-- source: application form ("Does the project have Team Members: No"); confirmed by the PI 03/11/2026 -->

## Summary of the project

I3K Technologies is developing EuLLM (https://www.eullm.eu), a local inference and model-development ecosystem for open-weight language models, and RAG Enterprise (https://www.rag-enterprise.com), an on-premises Retrieval-Augmented Generation and document-intelligence platform.

The objective of this Playground project is to validate and adapt EULLM Forge, I3K Technologies' model verticalisation pipeline, for HPC execution on Leonardo Booster. The initial focus is an Italian legal-domain language model intended for efficient local and on-premises deployment.

The activities will include validation of the Italian legal-domain dataset and training workflow, continued pre-training, knowledge-distillation experiments, evaluation of model quality, benchmarking of GPU memory utilisation and scaling behaviour, and post-training validation of the resulting models through GGUF conversion and EULLM Engine, with RAG Enterprise representing a downstream document-intelligence application environment.

The Playground allocation will be used to collect reliable technical data on resource consumption, distributed training behaviour and storage requirements, with the objective of preparing a subsequent EuroHPC AI Factory Fast Lane application for full-scale training and validation.

<!-- source: application form EHPC-AIF-2026PG01-1147, "Project summary (abstract)", verbatim -->

## Description of AI Factory services used

The project used Leonardo Booster compute and storage, the CINECA user
environment, and the support channel. It used no other AI Factory service
(training, consultancy, data services): porting, configuration and
optimisation were done by the project itself.

**Compute.**

- `boost_usr_prod`, the GPU partition of nodes with 4 NVIDIA A100 64 GB, 32
  cores and about 490 GB of RAM each. It ran all training: continued
  pre-training of a 30B MoE, logit distillation, SFT and GRPO. It also ran
  all generation by teacher models and all evaluation.
- QOS `normal`, up to 24 h a job, and `boost_qos_dbg` (30 minutes) for smoke
  tests and interactive checks.
- `lrd_all_serial`, the CPU-only serial partition, for work that bills no GPU
  time: merging adapters, GGUF conversion and quantisation, packaging,
  perplexity measurements, and the self-resubmitting jobs that recorded the
  queue every day.

The account had no budget on the DCGP (CPU) partition.

<!-- source: docs/leonardo-runbook.md §"The machine"; docs/cineca/leonardo.md §"SLURM specifics"; docs/cineca/allocation-recovery.md §6; forge/CLAUDE.md "Leonardo: the shape of a job", rule 4 -->

**Software environment.**

- Site modules: `gcc/12.2.0`, `cuda/12.2` (12.3 and 12.6 were also
  available, no CUDA 13), the `llvm/14.0.6` module and `python/3.11.7`.
- On top of these, a project Python virtual environment and local builds of
  llama.cpp and of the EuLLM Engine.
- Compute nodes have no internet. Model and dataset downloads ran on the
  login nodes into the project areas, and jobs ran offline.

<!-- source: docs/cineca/leonardo.md §"Build recipe"; forge/scripts/leonardo/login.sh; forge/scripts/leonardo/env.sh -->

**Storage.**

- `$HOME`: 50 GB.
- Project `$WORK`: 1 TB, permanent, no backup. It held the corpora, the
  checkpoints, adapters and GGUFs, and the evaluation results.
- Project fast scratch (`/leonardo_scratch/fast`): 1 TB, purged, about 7.5
  times faster on a sequential read (4.8 GB/s against 640 MB/s). It held the
  downloaded base models and the merged weights of the GRPO rounds.
- In the last days, the datamover nodes (`data.leonardo.cineca.it`) were used
  for copies between project areas and to outside servers.

<!-- source: docs/leonardo-runbook.md §"Storage", §"Moving data"; docs/cineca/leonardo.md §"Storage: $WORK vs the fast scratch tier" -->

**Support.**

- The data-transfer guidance (datamovers, GridFTP) came from CINECA's
  data-storage documentation, read on 03/11/2026.

<!-- source: docs/cineca/allocation-recovery.md §"Cosa fare, in ordine" item 3; docs/leonardo-runbook.md §"Moving data" -->

## Expected future work in the area

- **Fast Lane EHPC-AIF-2026FL01-1400 on Leonardo Booster.** 30,000 GPU-hours,
  02/11/2026 → 02/02/2027, awarded on 02/11/2026 with all the hours
  requested. The technical assessment accepted it on the grounds that the
  work had already been tested on Leonardo under this Playground. It has
  three lines:
  - **A.** A Mixture-of-Experts model with a hashed n-gram memory module for
    Italian law, behind a go/no-go gate.
  - **B.** Forge on several nodes, with a measured scaling curve on 1, 2 and
    4 nodes.
  - **C.** The proposal's commitments:
    - systematic teacher/student comparisons;
    - synthetic data at scale;
    - a timed full verticalisation cycle;
    - a second domain.
- **JUPITER Booster EHPC-AIF-2026PG01-1434.** 5,000 GPU-hours on GH200, until
  31/12/2026. It ports the pipeline to aarch64/GH200 against the Leonardo
  baselines measured here. It also runs the first steps of the legal-memory
  plan (knowledge-augmented continued pre-training, then the memory module).
- **LUMI-G EHPC-DEV-2026D09-278.** Inference and benchmarks only, until
  12/03/2027: the EuLLM Engine on MI250X, including the inference side of the
  memory module.
- **A Large Scale application** after the Fast Lane. It will be sized from
  GPU-hours observed rather than estimated, and from the multi-node scaling
  curve of line B.

<!-- source: docs/cineca/fastlane-plan.md header, §"Cosa abbiamo promesso", §"Le priorità", lines A-C; docs/jupiter/allocation-plan.md header; docs/research/legal-memory-plan.md §"Experiments" -->

---

# 3. MAIN FEATURES OF THE CODE

## Name of the code(s)

- **EuLLM Forge** (Python): the training and verticalisation pipeline
  (`forge/`).
- **EuLLM Engine** (Rust, on llama.cpp): the inference runtime (`engine/`).
  On Leonardo it was built from source and benchmarked on the A100.

<!-- source: .claude/CLAUDE.md "Architecture Decisions"; docs/cineca/leonardo.md -->

## Type of the code distribution

Open source, AGPL-3.0-or-later (Apache 2.0 until the relicensing of August
2026; versions published before that keep Apache 2.0):
https://github.com/eullm/eullm. The two published models,
`EuLLM/legal-it-8b` and `EuLLM/legal-it-4b`, are released under Apache 2.0,
the licence of their base models.

<!-- source: .claude/CLAUDE.md "License"; forge/model_cards/legal-it-8b/README.md, legal-it-4b/README.md "Licenza" -->

## Computational problem executed

Training and evaluation of transformer language models, all in bf16 on GPU.

1. **LoRA continued pre-training of a frozen 30B MoE.**
   - Model: Qwen3-30B-A3B-Base, 30.5B total and 3.3B active parameters.
   - 106,954,752 trainable parameters out of 30.6B (0.35 %).
   - The base was held in 8 bits and replicated on 4 GPUs under data
     parallelism.
2. **Logit (KL) distillation** from that teacher into Qwen3-4B-Base, on about
   700M tokens.
   - The teacher was split across 2 GPUs and the student trained on a third.
   - Throughput: about 380 optimiser steps an hour.
3. **Instruction tuning (open-book SFT)** with LoRA.
4. **Reinforcement learning (GRPO)**, rewarded by a program or by an LLM
   judge served beside the policy.
5. **On-policy distillation with privileged context** (reverse KL).
6. **Large-batch generation by teacher models**: synthetic training pairs,
   exams, rewrites.
7. **LLM-as-judge grading** of exams.
8. **CPU-only adapter merging, GGUF conversion and Q4_K_M quantisation.**

<!-- source: docs/paper/anatomy-of-a-training-job.md §"Numbers"; docs/legal-it-4b-strategy.md §4; docs/jupiter/allocation-plan.md deliverable 2 (Leonardo baseline ~380 steps/h); docs/legal-it-experiments.md §2-§3 -->

## Main libraries used, version and language. Usage of /usr/local libraries.

Language: Python 3.11 (site module `python/3.11.7`) for Forge; Rust and
C/C++/CUDA for the Engine and llama.cpp. No `/usr/local` libraries were
used. Everything came from site modules or from the project virtual
environment on `$WORK`.

| Library | Version on Leonardo | Note |
|---|---|---|
| PyTorch | 2.11.0+cu126 (with torchaudio 2.11.0, torchvision 0.26.0) | CUDA 12 wheels: the driver is r535 / CUDA 12.2 and refuses CUDA 13 builds |
| LLaMA-Factory | git commit 673048c6 | Phase-1 continued pre-training and the smoke test only |
| PEFT | 0.18.1 (pinned) | LoRA |
| DeepSpeed | 0.19.6 | ZeRO-3, dropped for the MoE after the OOMs |
| bitsandbytes | 0.50.2 | 8-bit frozen teacher |
| Transformers | 5.18.0 | models, generation, the trainers of Forge's own stages |
| Accelerate | 1.11.0 | device-map sharding of the teacher, GRPO launcher |
| TRL | 1.14.1 for GRPO (in a separate directory on `PYTHONPATH`, installed `--no-deps`); 0.24.0 in the venv, with LLaMA-Factory | GRPO |
| datasets | 4.0.0 in the venv; 5.0.1 in the GRPO directory (TRL needs ≥ 4.7) | |
| vLLM | 0.28.0 (cu129 wheel) | separate venv, v1.1 pilot logprob probes only |
| CUDA toolkit | 12.2 (site module) | Engine and llama-server builds, `sm_80` |
| GCC | 12.2.0 (site module) | the system GCC 8.5 failed the Engine link |

The exact state of the venv is the freeze
`$WORK/eullm_venv-known-good-2026-10-01.txt` on Leonardo, which is not in the
repository. The PI should take the missing versions from it.

<!-- source: forge/scripts/leonardo/setup_venv_latest.sh header (torch 2.11.0+cu126, driver r535); forge/scripts/leonardo/setup.sh §2 (cu126, peft 0.18.1, LLaMA-Factory 0.9.6, deepspeed, bitsandbytes); forge/scripts/install_training_deps.sh; forge/scripts/leonardo/setup_grpo_libs.sh (TRL 1.17.1, datasets 5.0.1); forge/scripts/leonardo/requirements-latest.txt header (LLaMA-Factory caps); forge/scripts/leonardo/sbatch_p1_logprob_probe.slurm (vLLM 0.28.0); docs/cineca/leonardo.md §"Why the prebuilt release binaries don't work", §"Every blocker hit"; forge/CLAUDE.md rule 5 (known-good file) -->

## Other software used on the EuroHPC JU systems. Usage of post-processing or pre-processing tools.

- **llama.cpp.**
  - A CPU build: `convert_hf_to_gguf.py` and `llama-quantize` for GGUF
    F16 → Q4_K_M, and `llama-perplexity`.
  - A CUDA build of `llama-server` for `sm_80`. It ran the exams on the
    shipped Q4_K_M GGUFs. It also served the teacher (Qwen3-30B-A3B-Instruct-2507,
    Q8_0) with one server per GPU for the memory-corpus rewrites, and the
    judge beside the GRPO policy.
  - llama.cpp at commit 3d82ef6 (20/09/2026).
- **EuLLM Engine**, built from source on a login node (Rust, `cargo rustc`
  with `gcc/12.2.0` and `cuda/12.2`).
- **Retrieval for the open-book exams:** BM25 plus Qwen3-Embedding-0.6B,
  fused by reciprocal rank, reranked by Qwen3-Reranker-0.6B.
- **Pre-processing (Forge's own code):**
  - pseudonymisation of rulings (`anonymize.py`) with a PII sweep that gates
    every training launch (`sweep_structured_pii.py`);
  - deduplication;
  - chunking and train/validation split (`format_pretraining.py`);
  - a hold-out check (`check_corpus_holdout.py`).
- **Slurm tooling written for this allocation** (`forge/scripts/leonardo/`):
  - `submit_chain.sh`: chains of resumable jobs with `afterany`;
  - `status.sh`: a morning status that flags failures, an empty queue and
    filesystems at 90 %;
  - `budget.sh`: consumption from `sacct`, billed like `saldo`;
  - `queue_stats.py` and `priority_snapshot.py`: daily snapshots of the
    queue, because `sacct`, `sprio` and `sshare` keep little or no history;
  - `heartbeat.sh`: one line a minute of GPU use, memory and RSS inside each
    job;
  - `submit_when_ready.sh`: watchers for jobs that wait on files.

<!-- source: forge/scripts/leonardo/sbatch_build_llama_server.slurm, sbatch_law_rewrites.slurm, sbatch_grpo.slurm, sbatch_norm_exam.slurm; forge/model_cards/legal-it-8b/README.md §"Retrieval consigliato"; docs/corpus-acquisition.md §"Processing, in order"; docs/paper/measurements/README.md; docs/paper/anatomy-of-a-training-job.md §"The lesson that generalises"; forge/CLAUDE.md "Automation is tested" -->

---

# 4. EXECUTION STEP

## How is the program launched?

**Batch jobs only, as chains.**

- Every phase is a Slurm batch script (`forge/scripts/leonardo/sbatch_*.slurm`),
  submitted from a login node with `submit_chain.sh <script> <N>`.
- The script queues N copies, each depending `afterany` on the one before it.
  When a link reaches its wall time and ends in TIMEOUT, which is the
  intended end of every link, the next one starts and resumes from the
  latest checkpoint.
- A resume restores the weights, the optimiser, the scheduler and the step
  count, and, from PR #529, the position in the data. Every link's log
  prints the number of batches it skipped.

**Job shapes.**

- The first plan used whole nodes (4 GPUs) with 24-hour links.
- From 22/09/2026 the shapes were 3 GPUs, 24 cores and 340 GB, or 1 GPU, 8
  cores and 120 GB, with 2-hour links. `save_steps` sits well below the
  steps a link can make: 100 or 300 against about 670.

**Rules applied to every submission.**

- Each experiment is its own chain, and several chains run concurrently.
- CPU-only packaging goes to `lrd_all_serial`.
- Dependent work is queued with `--dependency` or behind a watcher, never
  started by hand.
- Each submission checks its inputs and refuses to queue a chain that is
  already queued.
- A CI test fails any `boost_usr_prod` script that asks for more than 2 h,
  3 GPUs, or 8 cores and 123 GB per GPU.

**Evaluation and GRPO.** One submission (`submit_grpo_round.sh`) queues a
smoke test, the training links, the merge and GGUF export, the exams and
their judge.

<!-- source: forge/scripts/leonardo/submit_chain.sh header; docs/cineca/allocation-recovery.md §1-§6; forge/CLAUDE.md "Leonardo: the shape of a job that starts", rules 1-2, 4, 7 and the CI test paragraph; docs/legal-it-experiments.md §3 (submit_grpo_round.sh) -->

## Difficulties met to launch the code, if any, and how they were tackled

1. **Prebuilt binaries did not run** (Engine). There were three independent
   causes:
   - RHEL 8.7's glibc 2.28 against binaries built on Ubuntu 22.04;
   - the A100's `sm_80` was missing from the CUDA architecture list;
   - a CUDA 13.1 build needs driver r580, and here it fell back to the CPU
     while printing "GPU backend: CUDA".

   The fixes:
   - release builds on Rocky Linux 8;
   - a data-centre artifact for `sm_80/90` on CUDA 12.4;
   - a from-source recipe. The recipe uses `gcc/12.2.0` (the system GCC 8.5
     broke `std::filesystem`), extra bindgen include paths and
     `-C link-arg=-no-pie` passed through `cargo rustc`, so that it reaches
     only the final binary.

   <!-- source: docs/cineca/leonardo.md §"Why the prebuilt release binaries don't work here", §"Every blocker hit" -->

2. **The CUDA driver bounds the Python wheels.** The driver reports CUDA 12.2
   (r535). The first venv installed CUDA 13 torch wheels, and every job then
   failed its pre-flight check with "driver too old". vLLM's default PyPI
   wheel is also CUDA 13, and two 4-GPU jobs were lost before this was
   diagnosed.

   The fixes: torch from the `cu126` index; vLLM from its `cu129` wheel
   index; and a two-second check (`torch.version.cuda`) before any GPU
   submission.

   <!-- source: forge/scripts/leonardo/setup.sh §2 comment; docs/cineca/leonardo.md §"What the boundary actually is" -->

3. **ZeRO-3 ran out of memory on the MoE.**
   - ZeRO-3 materialises a whole layer on every GPU. A Qwen3-30B-A3B layer
     is about 625M parameters, against about 25M for a dense layer.
   - Under limits tuned for dense models, one GPU went from 17.5 GB to full
     (63.4 GiB) in 22 seconds of the first forward pass.
   - Three jobs ran out of memory. The incidents log bills the first one at
     0.67 node-hours.

   The fix: the frozen base is held in 8 bits and replicated on each GPU
   under plain data parallelism (about 37 GB a GPU). ZeRO-3 is no longer used
   on a frozen MoE.

   <!-- source: docs/paper/anatomy-of-a-training-job.md §5; docs/legal-it-4b-strategy.md §4; docs/paper/legal-it-4b-report-outline.md "Incidents log" -->

4. **Whole-node and long jobs were not placed.**
   - From about 21/09 the 4-GPU chains with 4- to 24-hour links waited for
     days.
   - Probe jobs submitted the same second answered the question:
     - 3 GPUs started in 87 s while 4 GPUs stayed pending;
     - 30-minute to 2-hour jobs started in 50 s;
     - 3-, 4- and 6-hour jobs never started in 22 hours.
   - A `sinfo` snapshot counted 1 idle node against 364 mixed.
   - The fix: 2-hour links on 3 GPUs, 24 cores and 340 GB (32 cores, or more
     memory, makes a job exclusive).

   <!-- source: docs/cineca/allocation-recovery.md §1-§2, §"Quanto costa una finestra" -->

5. **Resumes repeated the data.** Until PR #529, a resume restored the
   weights, the optimiser and the step, but restarted the data from the top.
   With 2-hour links, every link retrained the same ~650 steps. Held-out
   perplexity rose from 5.11 (step 12,600) to 5.29 (18,200) and 5.92
   (22,000).

   The fix: resumes continue from the data position, and every log shows it.

   <!-- source: docs/cineca/allocation-recovery.md §5; docs/paper/legal-it-4b-report-outline.md correction of 2026-09-25 -->

6. **`save_steps` against link length.**
   - With `save_steps: 1000` on 24-hour links, each link boundary lost 890
     steps (2 h 40 m).
   - A 1-hour link makes 291 steps, so with `save_steps: 300` it would save
     nothing.

   The fix: `save_steps` of 100-300.

   <!-- source: docs/paper/legal-it-4b-report-outline.md "Incidents log" (2026-09-15); docs/cineca/allocation-recovery.md §3 and table -->

7. **The shared venv broke** on 30/09/2026. One `pip install --upgrade`
   replaced torch, and every job died at `import torch` until the next
   morning.

   The fix: nothing is installed into the shared venv. Extra libraries (TRL)
   go to a `--target` directory with `--no-deps`. The known-good freeze of
   01/10/2026 is the restore point.

   <!-- source: forge/scripts/leonardo/setup_grpo_libs.sh header; forge/CLAUDE.md rule 5 -->

8. **Login-node limits.**
   - Merging a 4B model in bf16 on a login node was killed for memory twice.
     The second kill left a truncated 7.3 GB GGUF, which the next run would
     have used without warning.
   - In October, long `rsync` copies on login nodes were killed by the
     10-minute CPU-time limit.

   The fixes: merges and conversions moved to `lrd_all_serial`, with atomic
   `.partial` writes. Copies moved to a self-checking serial job and, at the
   end, to the datamover nodes.

   <!-- source: docs/paper/legal-it-4b-report-outline.md "Incidents log" (2026-09-20); docs/leonardo-runbook.md §"Moving data"; forge/scripts/leonardo/sbatch_copy_project.slurm header -->

9. **The serial partition is small.** It allows 10 queued and 2 running jobs
   per user, so it was kept for packaging. Evaluation moved to 1-GPU jobs.
   There is no crontab or scrontab, so daily records are kept by jobs that
   resubmit themselves.

   <!-- source: forge/CLAUDE.md rule 4; docs/leonardo-runbook.md §"Troubleshooting" -->

10. **Disk quota.**
    - `$WORK` (1 TB) reached 109 % on 05/10 (a 27B conversion failed), 96 %
      on 10/10 after two base-model downloads, and was full again on 17/10.
    - The fast scratch (1 TB) filled on 24/10 with the merged weights of 33
      GRPO rounds.

    The fixes:
    - base models moved to the scratch behind symlinks (`move_to_scratch.sh`,
      without stopping jobs);
    - automatic pruning of the merged weights, keeping six;
    - a `status.sh` alarm at 90 %.

    One pruning bug removed a round's start model. It was rebuilt from the
    adapters with `remerge_round.py`.

    <!-- source: docs/leonardo-runbook.md §"Storage"; forge/CLAUDE.md rule 6; docs/legal-it-experiments.md §12 "A lost start model" -->

11. **First-week configuration errors.**
    - Model ids that did not exist on the Hub.
    - Launchers pointing at checkpoint paths from before a rename.
    - A pre-flight check that verified the smoke model's tokenizer.
    - A codice-fiscale pattern that missed 59 codes while reporting a clean
      run.

    All were found before training started. The fixes were real model ids,
    repointed paths, `--tokenizer-model`, and a PII sweep that blocks the
    launch.

    <!-- source: docs/paper/legal-it-4b-report-outline.md "Incidents log" (2026-09-08) -->

---

# 5. DATA AND STORAGE

## Total storage required and total storage received (GB)

| | Required | Received | Used at the end (03/11/2026) |
|---|---:|---:|---:|
| `$WORK` | 1,024 GB (application form) | 1,000 GB (default project quota) | 937 GB |
| Fast scratch | — | 1,000 GB (fixed quota) | 814 GB |
| `$HOME` | — | 50 GB | 18 GB |

`$WORK` was the binding constraint: it was full or nearly full on 05/10,
10/10, 17/10 and 24/10. No extension was requested during the Playground.
The Fast Lane asked for 4 TB and received 5 TB.

<!-- source: docs/leonardo-runbook.md §"The machine", §"Storage", §"Moving data" ("Areas"); docs/cineca/fastlane-plan.md §"Le condizioni" (5 TB); end-of-allocation figures from the PI's reading on 03/11/2026, not in the repository -->

## Size of the data and/or the number of files

**Case law.**

- Corte di Cassazione: civil and criminal judgments 2021-2026, about 325,000
  rulings, pseudonymised and deduplicated.
  - 1,127,316 training and 11,387 validation chunks.
  - About 700M tokens, 2.9 GB of JSONL.
- Consiglio di Stato: 48,673 rulings. Years 2025-2026 are held out for
  evaluation only.
- About 230,000 structured cards extracted from rulings by the teacher.

**Legislation.** Nine Italian codes and acts from Normattiva, among them the
Constitution and the civil, criminal and procedure codes.

**Training sets.**

- 18,972 open-book SFT pairs; 16,116 of them were re-issued with passage
  citations.
- 1,060 false-premise questions.
- 75,352 case-law and 20,572 statute pairs for the case-law SFT.
- GRPO prompt sets.

**Evaluation sets** (private; only totals are reported):

| Set | Questions |
|---|---:|
| Held-out statute exam | 429 |
| Development statute exam | 472 |
| Practical exam | 145 + 70 held out |
| Second held-out exam | 233 |
| Case-law exam | 1,200 |
| Closed-book suite | 716 |

**Memory corpus, built in the last week.**

- 14,119 program-written rows.
- About 118,000 article rewrites in five variants.
- About 1.85 million rows of rewritten principles of law.

**Models and outputs.**

- About 80 GB of base models were prefetched.
- Merged GRPO rounds were 8-18 GB each.
- At the end: 247 GB of merged weights, and `eullm_runs` without
  checkpoints took 29 GB.
- About 330 GB was copied out to an I3K server and 88 GB to a workstation.


<!-- source: docs/research/legal-it-audit-2026-11.md §A.1 (325k rulings, CdS 48,673, ~230k cards, openbook pairs), §A.2 row "CPT aumentato" (memory corpus); docs/paper/legal-it-4b-report-outline.md §"Phase 0"; docs/legal-it-experiments.md §1, §8, "The exams", §13; forge/model_cards/legal-it-8b/README.md "Dati"; docs/leonardo-runbook.md §"0. One-time setup", §"Storage", §"Closing the allocation"; docs/research/legal-it-roadmap.md §0.1 -->

---

# 6. MAIN RESULTS

## Description of the results obtained, future perspectives, benefits to the society and the benefits of using computer resources

**1. Two published open models**, on Hugging Face under Apache 2.0 since
04/10/2026: `legal-it-8b` (from Ministral-3-8B-Instruct-2512,
huggingface.co/EuLLM/legal-it-8b) and `legal-it-4b` (from
Qwen3-4B-Instruct-2507, huggingface.co/EuLLM/legal-it-4b). Both were trained
on Leonardo on public legislation only, without rulings or personal data.

- **Recipe:** open-book SFT on 18,972 teacher-written pairs, then GRPO with
  rewards computed by a program (deadlines, abstention when the article is
  missing).
- **Results on 429 held-out questions**, with the same retrieval for every
  model:

| Model | Correct of 429 |
|---|---|
| Ministral-3-8B base | 347 (80.9 %) |
| legal-it-8b after SFT | 363 (84.6 %) |
| legal-it-8b after GRPO | 382 (89.0 %) |
| Qwen3-4B base | 333 (77.6 %) |
| legal-it-4b | 365 (85.1 %) |

- The paired differences against the base are significant under every
  criterion (8B: 49 against 14 questions, McNemar).
- **Cost:** training the 8B took about 9 GPU-hours. The data and the
  measuring cost much more: 40 GPU-hours of teacher generation, 88 of exams
  and grading.

**What the 89.0 % measures.**

- The questions are built automatically from single articles, and the answer
  is in the text the model is given. The exam therefore measures reading and
  retrieval on point questions, open book. It does not measure legal
  reasoning or the application of rules to facts.
- The GRPO gain (+4.4 points) is concentrated on deadlines and abstention.
- The judge (Qwen3-30B-A3B-Instruct-2507) agreed with an independent blind
  grading on 68-85 % of 40 answers, depending on the criterion, so every
  comparison is also reported with partial answers counted correct, and
  without the judge.

<!-- source: forge/model_cards/legal-it-8b/README.md "Risultati", "Addestramento"; forge/model_cards/legal-it-4b/README.md "Risultati"; docs/legal-it-experiments.md §2; docs/research/legal-it-audit-2026-11.md §0, §A.4 -->

**2. Quantisation costs nothing measurable.**

- Q4_K_M against bf16 on 472 development questions: 437 against 435 for the
  8B (p = 0.77), 422 against 426 for the 4B (p = 0.50).
- The files are 5.2 GB and 2.5 GB, the size for a laptop.
- On the same exam, Ministral-3-14B scored the same as the 8B (436 against
  437).

<!-- source: forge/model_cards/legal-it-8b/README.md, legal-it-4b/README.md "Quantizzazione"; docs/legal-it-experiments.md §2 -->

**3. Lower perplexity did not mean better reading.** The September logit
distillation of the 30B MoE into a 4B student on the Cassazione corpus:

- It cut perplexity from 6.65 to 3.19 on held-out chunks. This is an upper
  bound, because the split was by chunk.
- On Consiglio di Stato 2025, a court the training never saw, it cut
  perplexity by 23.25 % (6.46 to 4.96).
- On questions, the distilled model read worse than the instruct model:
  restating an article it had just read, 44 of 86 against 67. Asked about a
  nonexistent article, it invented an answer 18 times out of 18.

The project therefore moved to open-book training from instruct models
(section 1).

Two pilot measurements changed the distillation design:

- an int8 teacher stays within 0.006 nats of the bf16 one;
- a top-K logit cache loses far more than that. At K = 32 it keeps only
  93.7 % of the probability mass, and much less at distillation temperatures.

<!-- source: docs/legal-it-experiments.md §1; docs/paper/legal-it-4b-report-outline.md "The clean number", "Predictions, scored" -->

**4. Practical reasoning moved once, then plateaued.**

- GRPO on practical questions, graded by the judge, raised the 8B line from
  60 to 87 correct answers of 145 on the practical exam (r2l, 30:3,
  p < 0.001).
  - This is the only exam that needs several linked articles.
  - It was written and graded by models, with no measured human agreement.
- On a second held-out exam, asked once, the statute gain is small: 173-174
  against 168-169 of 233 (p ≈ 0.4, not significant).
- Later rounds on generated false premises saturated.
- On an external set of 36 trap questions, the best rounds sit at 14-16, the
  published model at 14. No round reached the target of 20, and 11 traps are
  solved by none of the seven models tried.
- A long case-law SFT on the published model undid its GRPO: development
  exam 437 to 415, traps 14 to 4. Since then, what is added on top of a
  model shaped by GRPO is a short SFT followed by GRPO.

<!-- source: docs/legal-it-experiments.md §8, §9, §11, §13; docs/research/legal-it-audit-2026-11.md §A.4, §B.3, §B.5 -->

**5. Teachers.**

- On the practical exam, without reasoning:
  - Qwen3.6-27B: 76 (85 with reasoning on);
  - Qwen3-30B-A3B-Instruct: 74;
  - Gemma 4 31B: 74;
  - Mistral Small 4: 55-60 (generated on JUPITER, graded on Leonardo).
- The trained 8B (87) already exceeds every teacher without reasoning, so
  distilling from the available teachers cannot raise it further.

<!-- source: docs/legal-it-experiments.md §9, §10; docs/research/legal-it-audit-2026-11.md §B.10 -->

**6. Closed book, the models do not know the law.**

- On a 716-item closed-book suite, scored by a program, the models are right
  3-31 % of the time, with 360-489 confidently wrong answers.
- The 30-31 % of the two best 8B rounds is mostly a reflex taught by the
  premise rounds. They say "does not exist": one round denies 91 % of the
  made-up articles and recognises none of the real ones.
- The model cards therefore warn against use without retrieval.

<!-- source: docs/legal-it-experiments.md §13 "Closed book"; docs/research/legal-memory-plan.md "Results" E0; forge/model_cards/legal-it-8b/README.md warning -->

**7. A memory corpus for the next step.** Built on Leonardo in the last week:

- 14,119 program-written rows;
- about 118,000 article rewrites;
- about 1.85 million rewritten principles of law.

Each rewrite is refused by a program check if it adds an article, number,
deadline or penalty its source lacks; 4.3 % were refused in the first pass.
Read by hand, teacher-written cases and explanations were wrong in law 3
times in 6, so they are kept out of training.

<!-- source: docs/research/legal-it-audit-2026-11.md §A.2 "CPT aumentato"; docs/research/legal-memory-plan.md "Results" -->

**8. Engine on A100.** Over 16 concurrent requests on a 27B Q8 model, 4 GPUs
gave 102.1 tok/s against 54.7 on one, and 128.1 tok/s with a q8_0 KV cache.

<!-- source: docs/cineca/leonardo.md §"Runtime findings", §"What the published 0.7.5-rc4 binaries actually do here" -->

**Perspectives and benefits.**

- Small, open, European-trained legal models that run locally, so legal
  texts and users' questions need not leave the user's machine.
- An open, reusable pipeline.
- A documented method for running long training on a busy shared GPU
  partition: job shapes, resumable chains, the data-position resume.
- Baselines for the JUPITER, Fast Lane and LUMI work.

The A100 nodes made it possible to keep a 30B teacher in memory and to run
over 30 GRPO rounds with their exams in two months.

<!-- source: forge/model_cards/legal-it-8b/README.md "Europeo dall'inizio alla fine"; docs/jupiter/allocation-plan.md "What we committed to"; docs/leonardo-runbook.md §"Storage" (33 GRPO rounds) -->

## Conclusions about the project

- The pipeline ran end to end on Leonardo, and two models were published
  within the allocation.
- The decisive lesson concerned measurement. Perplexity, loss and even a
  strong open-book score each looked like progress that the next, harder
  measurement did not confirm.
- Five of six grader errors were found by reading answers. Since then a
  comparison decides nothing until answers have been read.
- The models read and cite well. They do not reason reliably on cases, and
  they do not know the law without texts. These are the two lines of the
  follow-up work (an application benchmark validated by a lawyer, and the
  legal-memory plan).

<!-- source: docs/research/legal-it-audit-2026-11.md §0, §D, §E; forge/CLAUDE.md "A number decides nothing until the answers behind it are read" -->

## Usability of the assigned EuroHPC JU AI Factory

- The nodes were reliable and well suited: four A100 64 GB with ample host
  memory, and a fast scratch 7.5 times faster than `$WORK`.
- The serial partition let packaging run without spending GPU budget.
- Three properties shaped the usability:
  - from about three weeks in, the queue favoured short partial-node jobs;
  - the default 1 TB `$WORK` was tight for LLM checkpoints;
  - compute nodes have no network, which requires disciplined prefetching.
- Once job shapes matched the queue, consumption rose from 12.3 node-hours a
  day on average to 25-50.

<!-- source: docs/cineca/leonardo.md §"Storage"; docs/cineca/allocation-recovery.md §"La ricetta"; docs/leonardo-runbook.md §"The machine" -->

---

# 7. FEEDBACK AND TECHNICAL DEPLOYMENT

## Feedback on the AI Factory/EuroHPC JU access procedures

**What worked.**

- Access was fast: the account was active on 02/09, the start date, and an
  engine build and benchmark ran on a Booster node on 04/09.
- Interactive debug allocations (`boost_qos_dbg`) bill only the time used.
- The serial partition is free of GPU budget.
- The Booster's software stack had what a from-source build needed: GCC 12,
  CUDA 12.x and LLVM modules.
- For the follow-on Fast Lane, CINECA answered within a day and raised
  `$WORK` from 1 TB to 5 TB.
- Onboarding and login were simple, and the experience was more than
  satisfactory overall. The two-factor authentication with 12-hour step-ca
  SSH certificates added one short step per working day and caused no
  difficulty.

<!-- source: docs/cineca/leonardo.md header (tested 2026-09-04), §"SLURM specifics", §"Build recipe"; docs/leonardo-runbook.md §"The machine"; docs/cineca/fastlane-plan.md §"Le condizioni" (ticket #86782, 5 TB) -->

**What could be better.**

1. **The queue's real behaviour was learnt only from our own probes.** From
   about 21/09:
   - 4-GPU jobs of 4-24 hours waited for days;
   - 3-GPU jobs of up to 2 hours started within a minute;
   - a `sinfo` snapshot showed 1 idle node against 364 mixed.

   Nothing in the documentation suggested that the shape of the request,
   rather than priority or fairshare, was the obstacle. The observed maximum
   priority (60,259,060) also exceeded the sum of the published weights, so
   priority could not be explained from what a user can read. A short note
   on the job shapes that start, or a placement estimate per shape, would
   have saved about two weeks of allocation.

   <!-- source: docs/cineca/allocation-recovery.md §1-§2, "Il quadro, in numeri misurati", "Un'avvertenza onesta" -->

2. **There were no idle whole nodes for multi-hour jobs.** A Playground
   cannot reserve nodes, so work that needs a whole node (a sharded 30B MoE)
   was hard to place. A reservation option for short windows would help.

   <!-- source: docs/cineca/allocation-recovery.md §1, §"Cosa fare" item 3 -->

3. **Data transfer.**
   - Long `rsync` copies on login nodes were killed by the 10-minute CPU-time
     limit.
   - The datamover nodes and GridFTP were found in CINECA's documentation
     only on 03/11, after the allocation had ended.
   - A pointer to them in the onboarding material, and in the end-of-project
     notice, would help.

   <!-- source: docs/leonardo-runbook.md §"Moving data" -->

4. **History.** `sacct` keeps a few weeks, and `sprio`, `sshare` and
   `saldo` keep no history. There is no crontab or scrontab. To report its
   own usage, the project had to write self-resubmitting jobs.

   <!-- source: docs/paper/measurements/README.md; docs/leonardo-runbook.md §"Troubleshooting" -->

5. **Driver and toolkit.** The newest CUDA is 12.6 on a CUDA 12.2 driver.
   Current PyTorch and vLLM wheels default to CUDA 13, which does not run, so
   pinning older wheel indexes is needed. A note in the user guide would
   prevent lost jobs.

   <!-- source: docs/cineca/leonardo.md §"What the boundary actually is" -->

6. **Storage.** A default of 1 TB on `$WORK` is small for LLM work: a merged
   8B model is 17 GB, and the same again while converting.

   <!-- source: forge/CLAUDE.md rule 6 -->

## Explanation of how the computer time was used compared with the work plan presented in the proposal. Justification of discrepancies, especially if the computer time was not completely used.

**The plan.** The application set out to validate EULLM Forge on one Leonardo Booster node: continued pre-training and knowledge distillation for an Italian legal model (then planned as a Qwen3-32B teacher and a 7B student), a sharding strategy for the 64 GB A100, measurements of throughput, memory and scaling, GGUF conversion validated through EULLM Engine, and measured figures for a Fast Lane application.

**What changed, and why.** Measurement on Leonardo replaced two choices. The teacher became Qwen3-30B-A3B, a Mixture-of-Experts model, and the students 4B and 8B: they fit the A100 with room for training and give a model that runs on a laptop. After the September distillation lowered perplexity but not exam scores, the work moved to open-book models trained from instruct bases, refined with GRPO; LLaMA-Factory served only the first pre-training stage, everything after ran on Forge's own code. The medical and finance verticals were not started, for lack of corpora rather than of compute. The final objective was met: the Fast Lane EHPC-AIF-2026FL01-1400 (30,000 GPU-hours on Leonardo Booster) was awarded on 02/11/2026.

**How the time was used.** The allocation was fully used: 40,261 local hours, 100.7 % of the 40,000 awarded (about 5,030 GPU-hours). Use was uneven:

- **02/09-08/09:** the queue was empty for about 146 hours, through our own fault: the first week went to organising the work on the machine and fixing configuration errors. Only 7.2 idle hours were due to a full cluster.
- **By 22/09:** 20 % of the node-hours had been used at 34 % of the calendar. Our 24-hour whole-node jobs were rarely placed.
- **From 23/09:** probe jobs showed which shapes the queue placed, and the project changed four things: 2-hour links on 3-GPU slices, checkpoints matched to the link, resumes that continue the data, and several independent chains at once. Daily use rose from about 12 to 40-50 node-hours.
- **October:** many shorter jobs (teacher data generation, SFT, over 30 GRPO rounds with their exams and judge, evaluation of candidate teachers), and in the last days the memory-corpus generation.

**Where the hours went** (scheduler accounting, GPU-hours by job, 02/09-02/11/2026):

| Stage | GPU-hours | Share |
|---|---:|---:|
| Teacher-student distillation (September, five arms) | 1,754 | 35 % |
| GRPO rounds | 936 | 19 % |
| On-policy distillation | 626 | 12 % |
| Case-law cards and principles (teacher generation) | 603 | 12 % |
| Evaluation (exams, judge model) | 522 | 10 % |
| Continued pre-training of the teacher | 349 | 7 % |
| Synthetic instruction data | 102 | 2 % |
| Supervised fine-tuning | 39 | 1 % |
| Memory corpus | 39 | 1 % |
| Other (probes, builds, benchmarks) | 61 | 1 % |
| **Total** | **5,031** | |

<!-- source: application form; docs/leonardo-allocation-plan.md; docs/cineca/allocation-recovery.md; docs/legal-it-experiments.md; sacct run by the PI 03/11/2026 -->

## Willingness to apply to EuroHPC JU Access Modes in the future

Yes, and the project has already done so.

- **Fast Lane EHPC-AIF-2026FL01-1400.** 30,000 GPU-hours on Leonardo,
  02/11/2026 → 02/02/2027, awarded in full. The technical assessment cited
  this Playground as evidence that the work runs on Leonardo.
- **JUPITER Playground EHPC-AIF-2026PG01-1434.** It ports the pipeline to
  GH200 against the Leonardo baselines measured here.
- **LUMI DEV EHPC-DEV-2026D09-278.** It covers the inference engine.
- **A Large Scale application is planned after the Fast Lane.** The
  Playground's main lesson for it: a request should rest on measured
  numbers, not estimates. These are GPU-hours per phase, placement times per
  job shape, and a multi-node scaling curve (Fast Lane line B).

A Playground of two months was the right first step: long enough to find
out what the queue and the filesystems allow, and short enough that
under-use is visible early.

<!-- source: docs/cineca/fastlane-plan.md header, §"Cosa abbiamo promesso" (P4, P7), line B; docs/jupiter/allocation-plan.md header -->

---
