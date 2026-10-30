# EHPC-AIF-2026PG01-1434 on JUPITER Booster — project dossier

*Porting and scaling a teacher–student verticalisation pipeline for European
domain-specific LLMs to Grace Hopper.* EuroHPC AI Factory Playground, JUPITER
Booster (JSC), 5,000 GPU-hours over two months, budget `e-aif-2026pg01-1434`.
Principal Investigator: Francesco Marchetti, I3K Technologies Srl (Milan).

This dossier is the running record of the allocation: what was set up, what
was measured, what went wrong and how it was fixed, and what the hours were
spent on. It is kept with the code (`docs/jupiter/dossier.md` in the EULLM
repository) and updated as the work goes; the working notes behind it are in
`docs/jupiter/allocation-plan.md`. Every figure below comes from a job on
JUPITER, named by its Slurm job id.

**Last updated: 21-10-2026.**

## 1. At a glance

| | |
|---|---|
| Awarded | 06-10-2026; first job on 14-10-2026 |
| System | JUPITER Booster: 4× GH200 (H100 96 GB HBM3, 72-core Neoverse-V2 Grace) a node, whole nodes only |
| Spent so far | 474.9 GPU-hours (9.5%) by 23-10-2026 in JSC's own count (`jutil user cpuquota`, 72 core-hours a GPU-hour), 4,525.1 left (`tools/jupiter/budget.sh` gives it, with each day's from Slurm's accounting); planned to 15-12-2026, two months from activation (JuDoor shows 31-12-2026, kept as the margin): about 82 GPU-hours a day from 16-10, 20 node-hours, one node busy 85% of the time |
| Codes | EULLM Engine (Rust, llama.cpp), EULLM Forge (Python: PyTorch 2.9.1 of the site, Transformers 5.19, PEFT 0.21) |
| Data on JUPITER | Legislation (Normattiva, EUR-Lex) and public benchmarks; pseudonymised court decisions once JSC's data-protection addendum is signed, asked for on 15-10-2026 (§7) |

## 2. What the proposal committed to, and where each stands

| # | Deliverable | Leonardo baseline (A100 64 GB) | Status on JUPITER |
|---|---|---|---|
| 1 | The pipeline runs on aarch64 + GH200, unattended, evaluation included | runs end to end | Corpus, distillation (a 21-link chain, one epoch of 139M tokens), pair generation, instruction tuning, exam and judge run unattended, job after job (§5.3–5.9) |
| 2 | Distillation throughput per step, teacher in BF16 on one GPU | ~380 optimiser steps/h, teacher on 2 GPUs + student on a 3rd | **548–553 steps/h** with 2,048-token micro-batches, teacher alone on one GPU (§5.2) |
| 3 | Two runs per node (2 GPUs each) | 1 run per 3 GPUs | **Done**: two runs side by side at the same speed (§5.2, §5.3) |
| 4 | Instruction tuning time | 4B, 3,400 examples, 2 epochs: 46 min on one GPU | **24.4 min** for the 4B on 4,272 pairs, 2 epochs, one GH200: 0.34 s a pair against 0.81, 2.4× per GPU; the 8B 28 min; four runs side by side (§5.5) |
| 5 | A faster inference backend for synthetic data generation | 58 tok/s on 3 A100, 30B-A3B-Instruct in plain Transformers | **Done: 2,279 tok/s on one node (4 GH200), same model, EULLM Engine: 39× per node** (§5.4) |
| 6 | Quality on the fixed held-out set | e.g. the 5.11 → 5.92 perplexity regression it caught | A 217-question exam drawn from the law, graded by Qwen3-30B-A3B-Instruct-2507; open book, on pairs that never touch an exam article, Qwen3.5-9B 0.790 and Qwen3.5-4B 0.786 against 0.765 for Qwen3-4B-Instruct-2507 (§5.8) |
| 7 | Go/no-go on the Qwen3.5 family | — | **Go**: trains at near Qwen3 speed with flash-linear-attention and Triton 3.8.0, and its 4B and 9B are the best students on the exam (§5.6, §5.8) |
| 8 | GPU-hours per distillation epoch and per complete vertical | — | Legislation, 3 epochs: **2.98 GPU-hours an epoch a run** unpacked, **0.52 packed** (§5.3) |
| 9 | Energy use and carbon footprint | — | GPU energy per job: generation, instruction tuning, distillation (§6) |

## 3. Log

**25-10-2026**
* GRPO ported from Leonardo, to start from the distilled 9B with the
  false premises (§5.11): `submit_grpo_round.sh` runs a smoke test, the
  training links, the merge, five sets asked open book and their judge, in
  one submission. The policy trains on three GPUs and the judge, EULLM
  Engine serving Qwen3-30B-A3B-Instruct-2507, grades on the fourth; TRL
  1.16.0 is kept out of the venv (`setup_grpo_libs.sh`). The prompts are
  Leonardo's, without the articles of the tests drawn here
  (`filter_grpo_prompts.py`).
* Round g9-r1 queued from the distilled 9B with the premises: of the
  2,816 prompts of Leonardo's r2o, 2,184 kept (1,659 graded by the judge:
  1,144 on what an article provides, 515 practical; 525 by the verifiable
  rewards, 175 of them false premises); 228 dropped as on an article of
  our tests, 404 as repeated. Smoke test 2515386, links 2515387-2515388,
  the five sets 2515389, the start model on the held-out exam 2515390,
  judge 2515391.
* The false-premise round on the students continued on the rest of
  norms-v2 (q35r), the same 1,606 premise pairs and 14,209 open-book pairs
  as dqp, so only the distillation differs: tuning 2515537 (the held-out
  exam judged by 2515538), the LexIA traps and the practical exam
  2515539-2515542, their judge 2515543.

**26-10-2026**
* GRPO round g9-r1 (§5.11): 300 steps in 1 h 59 min of one node
  (2515387; about 24 s a step, the judge on EULLM Engine included). The
  practical exam rose from 67 to 79 of 145 and the held-out exam from 189
  to 198 of 217. The trap tests stayed level, and LexIA moved from 11 to 9.
* The rest of norms-v2 under the false premises (dqr): the 9B 13 LexIA
  traps against 11, the 4B 11 against 9, the practical exam level. But on
  the held-out exam the 9B fell from 188 to 168 open book without the note
  (judge 2515538), and from 189 to 166 asked as the product asks. The 4B
  is at 178-179. The continued distillation costs the held-out exam about
  20 questions.
* GRPO round g9r-r1, from the dqr 9B, same recipe as g9-r1: the held-out
  exam from 166 back to 198 (37:5, p<0.001 strict and lenient, the only
  difference shown of these rounds; judge-free 17:3), LexIA 15 against
  13 (2:0), the practical exam 70 against 67 (8:5), the trap tests level.
  GRPO repairs what the continued distillation cost on the held-out exam,
  and both rounds end at 198.
* Queued for the night: g9-r2 and g9r-r2, second rounds on each line,
  six prompts a step, 400 steps over the whole prompt set in another
  order (2533859-2533863, 2534077-2534087).

**27-10-2026**
* Second rounds. g9r-r2 is the candidate: LexIA 14, held-out exam 200,
  practical exam 81 + 16, trap test 384, rewritten 973. g9-r2 has
  12 / 200 / 77 + 20 / 384 / 964. Paired against dqp-d-q9b:
  - practical exam 21:7, p=0.013 strict, p=0.093 lenient;
  - held-out exam 14:3, p=0.013 strict, p=0.23 lenient;
  - LexIA 5:2;
  - rewritten trap test 16:8.
  The gain is significant strict and not lenient, on every set alike.
* The answers behind the practical exam, five each way. The wins are real:
  the second article the answer needs, cited and applied, where the start
  model cited one and was graded partial or wrong. Three of the five
  losses cite the wrong article, the neighbouring number or the same
  number of another code among the retrieved texts. Counted over the
  whole sets (`citation_check.py`, no judge), GRPO did not bring it:
  - practical exam: answers citing only articles outside the question's
    group went from 19 of 145 (dqp) to 15 in every GRPO round;
  - held-out exam: 1-3 of 217 for every model.
  Answers citing an article outside the group as well grew a little
  (43 to 46), all among the right ones (11 to 18), as the completeness
  gain predicts. The error is old and common: about half of every
  model's wrong practical answers cite an article outside the group (31
  of 57 for dqp, 28 of 48 for g9r-r2), against 13-22% of its right ones.
* The 4B's first round, g4-r1, from dqp-d-q4b with the second rounds'
  settings (2541382-2541387). The vertical of g9-r2 and g9r-r2, GGUF
  Q4_K_M through EULLM Engine (2541390, judge 2541391).
* Waiting for a node: a 6-hour link (2541383) asked for at 09:50 got
  one at 19:55; the vertical (4 h asked) waited 1 h 50 min, the short
  jobs none. The GRPO links are 3 hours from 28-10 (`GRPO_LINK_H=3`).

**28-10-2026**
* The vertical of the second rounds (2541390, 5 minutes of one node:
  conversion 66 s, Q4_K_M 21 s, each exam 83-93 s; judge 2541391), the
  held-out exam open book through EULLM Engine and llama-server. The
  same exam as dqp's vertical (191 open, 187 without the note).

  | Q4_K_M | engine, open | engine, no note | llama-server, open | llama-server, no note |
  |---|---:|---:|---:|---:|
  | g9-r2 | 197 | 197 | 198 | 195 |
  | g9r-r2 | 196 | 194 | 197 | 195 |

  GRPO's gain survives quantisation: 6 to 10 questions over dqp's Q4.
* The 4B's first round, g4-r1 (400 steps of six prompts in 3 h 52 min,
  2541383), against dqp-d-q4b:
  - held-out exam 197 + 2 against 186 + 13;
  - practical exam 68 + 22 against 63 + 22;
  - LexIA 9 against 9;
  - trap test 385 against 383, rewritten 970 against 956.
  It is the 9B's pattern: partials turned complete, the traps level.
* Queued, in 3-hour links: g4-r2 from g4-r1 (2566556-2566561), and
  g9r-r3 from g9r-r2 (2566890-2566897). g9r-r3 is the control the next
  reward must beat: a third round of the same recipe.
* The llama.cpp b11925 test (branch `chore/llama-cpp-b11925`): the first
  build labelled llama-b11925 was made from this branch's checkout, not
  the bump's: `EULLM_REPO`, exported by Forge's env.sh in the same shell,
  overrode the checkout `build_engine.sh` was run from (0.7.93 at
  2401fe4 instead of 0.7.95). To be built again with
  `env -u EULLM_REPO`. `tools/jupiter/sbatch_mtp_compare.slurm` is ready
  for the MTP part.

**29/30-10-2026**
* Third rounds, the same recipe again (seed 2). Judge's grades, correct
  (+ partial where it matters):

  | | LexIA | held-out | practical | trap test | rewritten |
  |---|---:|---:|---:|---:|---:|
  | g4-r1 | 9 | 197 | 68 + 22 | 385 | 970 |
  | g4-r2 | 12 | 199 | 74 + 20 | 387 | 965 |
  | g4-r3 | 10 | **202** | 75 + 16 | 387 | 966 |
  | g9r-r2 | 14 | 200 | **81** + 16 | 384 | **973** |
  | g9r-r3 | **16** | 199 | 75 + 20 | 383 | 968 |

  The 9B has stopped gaining on these prompts. Its third round wins two
  LexIA traps and loses six practical questions, both within what the
  judge moves on its own. The 4B still gains a little on the held-out
  exam, and g4-r3, at 202, is the best model on it of any size. The 4B
  is level with the 9B within 6 questions on the practical exam and 1 on
  the trap tests.
* The citation reward (`--citation-weight 0.3`, c3 rounds from r2 with
  r3's seed, so that only the reward differs):

  | | LexIA | held-out | practical | trap test | rewritten | only-off citations |
  |---|---:|---:|---:|---:|---:|---:|
  | g9r-r3 | 16 | 199 | 75 + 20 | 383 | 968 | 15 |
  | g9r-c3 | 15 | 201 | 79 + 18 | 383 | 968 | 14 |
  | g4-r3 | 10 | 202 | 75 + 16 | 387 | 966 | 20 |
  | g4-c3 | 12 | 200 | 74 + 17 | 386 | 947 | 18 |

  It changed next to nothing: the answers citing only articles outside
  the question's fell by one or two, and every difference is within what
  the judge moves alone, except the 4B's rewritten trap test, 19 lower.
  About 90% of the answers already cite the question's article, so the
  reward had little left to pay for. The wrong citations are a choice
  among the retrieved texts, which this reward does not see. Dropped.
  Since third rounds of the same recipe gave the 9B nothing either, GRPO
  on these prompts is done for the 9B. The candidates stay g9r-r2 (9B)
  and g4-r3 (4B), with g4-r4 running (2603577-2603582) and g4-r3's
  vertical (2603583-2603584).
* 30-10: login node jpbl-s01-01 without /e mounted (no home, no project);
  jpbl-s01-02 had it. No job was affected; the nodes stood idle from the
  end of the third rounds to the next submission.

**23-10-2026**
* The second node idle overnight: the submission script refused the
  distillation checkpoints given as "base+adapter" as missing models, which
  the job itself loads; fixed (`d51cee9`) and resubmitted in the morning.
* Qwen3.5 halfway through its distillation on norms-v2, through the
  open-book recipe (job 2466380, §5.8): the 9B distilled 189 of 217 with the
  note and 187 without, against 182 and 179 for its Base, level with
  Ministral-3-8B-Instruct; the 4B unchanged (186 and 177 against 187 and
  178).
* The Qwen3.5 chain at 8,500 (4B) and 9,100 (9B) of 15,625 steps after five
  8-hour links, about 1,700 a link for the 4B: three more links (2466376,
  2466377, 2466379), a link with nothing left ending in minutes.
* Spending: 474.9 GPU-hours by JSC's count (9.5%); `jutil` dates the period
  from 01-10, so a uniform pace would be at 24%, 185 node-hours short; 112.7
  GPU-hours in the last 24 hours with one node busy. Two nodes kept busy.
* The Qwen3.5 distillation on norms-v2 complete (§5.9): the 4B and the 9B
  at step 15,625 of 15,625 and merged, at the end of the seventh link
  (2431505); the three links added in the morning cancelled unstarted. The
  final pair queued through the open-book recipe (job 2480052, graded by
  2480053).
* The Qwen3.5 pair at the end of its epoch (§5.8): the 9B 188 and 187 of 217,
  as halfway, the 4B 189 and 183 (+2 and +5 over its Base); five models now
  within 189 open book, more than this exam separates.
* Our models on the private traps and practical exam, graded on JUPITER
  (§5.11): the distilled 9B first on the practical exam (66 of 145, +8 over
  its Base), last on the traps (3 of 36).
* The false-premise round queued (`submit_premise_chain.sh`): Leonardo's
  false-premise pairs generated here from the legislation (job 2488357),
  the distilled Qwen3.5 4B and 9B tuned on them with the open-book pairs
  (2488358, its exam judged by 2488359), then the traps and the practical
  exam (2488360-2488363) and their judge (2488364).
* The false-premise round (§5.11): the distilled 9B from 3 traps of 36 to
  11 with the practical exam at 67 and the held-out one at 188/188, the
  candidate so far; the vertical of the final pair, Q4_K_M costing the 9B
  3 questions.
* The two students continued on the rest of norms-v2, the 142,538 training
  chunks (~81M tokens) the 250,000 sample left out, the same 3,000 held out
  (`build_norms_v2_rest.sh`): links 2480199-2480203 of 8 hours, about 30
  hours for the 9B's 8,900 steps; whether the whole corpus moves the 9B
  further than the sample did.

**21-10-2026**
* Job 2427647: JUPITER's part of the three sites' comparison (c13-sites,
  `docs/sites-comparison.md`), the engine built from `main` of 20-10. With
  CUDA graphs on, a server at 32 requests for 90 minutes kept its VRAM flat
  (0.00 GiB an hour, as with them off), where LUMI's grew 1.3-2.5 GiB an hour
  with HIP graphs until they died; speed of three models at 1-32 requests
  (§5.1). `tools/jupiter/sites.sh` builds, sets up and submits it unattended.
* Queued: Leonardo's open-book recipe (one epoch on the 14,209 open-book
  pairs, lr 1e-4, 4,096-token sequences; instruct bases keep their chat rows,
  Ministral our system prompt in place of its "Le Chat" one) on Qwen3.5-4B
  and 9B Base, Ministral-3-8B-Instruct-2512 and Qwen3-4B-Instruct-2507, the
  exam also open book without the note on missing articles, as the product
  asks: links 2429166-2429167, then the judge 2429168
  (`submit_sft_chain.sh`, `3dd7a08`).
* Leonardo's open-book recipe graded (job 2429168): Ministral-3-8B-Instruct
  0.887 open book and the same without the note, Qwen3.5-4B 0.880, the
  trained Qwen3-4B-Instruct 0.869, against 0.765 for the reference (§5.8).
  The second two-hour link finished every run and every exam; the 6-hour
  link behind it found nothing left and ended in 2 minutes.
* The complete vertical (§5.8): Qwen3.5-4B and Ministral-3-8B-Instruct to
  GGUF Q4_K_M, examined through EULLM Engine at their bf16 scores (Ministral
  189 of 217 with the note, 187 without; jobs 2447162, 2449393), in minutes
  after training.
* Qwen3.5 distillation on norms-v2 started (job 2431499), seven 8-hour
  links: 8-hour jobs get a node as soon as 2-hour ones on JUPITER (median
  wait 0 minutes, 5 jobs), so links are as long as the work needs.
* Distilled against Base under the open-book recipe (job 2455477, §5.8):
  the Qwen3 models distilled on norms-v2 ahead in every comparison, the
  8B by 9 questions without the note (183 against 174 of 217).
* JSC's Data Protection Team answered on the pseudonymised decisions: the
  data fall under the GDPR, and it asks for its checklist with the DPO (§7).
* Spending behind pace: 347.9 GPU-hours by Slurm's count on 21-10 (7.0%),
  against 9.4% for a uniform pace from 14-10 to 31-12, 30 node-hours short;
  the rest by 31-12 needs 65 a day, the last 24 hours spent 16 (`budget.sh`
  now prints the pace table). A
  second chain beside the first: the Qwen3.5 distillation on norms-v2, the
  35B-A3B teaching the 4B and the 9B, one epoch unpacked (deliverable 7's
  GPU-hours an epoch against Qwen3's on the same corpus).

**20-10-2026**
* Mistral Small 4 converted from FP8 to BF16 on a login node (238.8 GB,
  `dequantize_fp8.py`) and examined as a candidate teacher on two private
  question sets, open book, answers kept for Leonardo's judge (§5.10).
* Faults fixed on the way: FP8 weights needing kernels from the Hub,
  experts left in FP8 by dequantizing on loading, a snapshot the Hub client
  refused, a GPU out of memory at 48 questions a batch (§8).
* A maintenance reservation held four jobs whose time limits were wider
  than needed; resubmitted as two, they started at once. All six answer
  sets done by the evening, ~14 GPU-hours for the six runs (~9 more for
  the failed attempts before them).

**19-10-2026**
* JUPITER unavailable most of the day (34.6 GPU-hours spent).
* The open-book pairs graded: every model up 7.8–15.5 points, Qwen3.5-4B
  186 of 217 against 159 for the reference (job 2413383, §5.8).
* Judges stopped at their one-hour limit: the grader took 3,495 s to load
  from `$PROJECT`. Graders are now copied into node memory like the teacher
  (`stage_model.sh`, `3746ea8`); the next judge took 18 minutes.
* `budget.sh` prints JSC's own figure for the allocation besides Slurm's.

**18-10-2026**
* The instruction pairs were found to overlap the exam (105 of its 153
  articles); traced, removed, and every comparison with the reference rerun
  on the clean pairs (§5.8).
* norms-v2 distillation finished its epoch (7,224 steps, 4B and 8B), merged;
  held-out loss down 26% and 24%. Instruction-tuned and examined: no change
  open book (§5.8, §5.9).
* 19,141 pairs generated from the EU acts (job 2390452) lowered the open-book
  score when added (§5.8). The pairs were 44–59% questions answered from
  memory; the exam's own task is open book.
* 14,209 open-book pairs generated by EULLM Engine from the Italian codes,
  the exam's articles left out (job 2403329, §5.4). Queued for the night:
  instruction tuning on them alone and mixed with the clean pairs, four
  models each, then the judge.

**17-10-2026**
* norms-v2 built on a login node: the codes with 17,529 EU acts in Italian,
  392,538 training chunks; its packed distillation chain started (§5.9).
* Held-out exam graded by a judge model, closed and open book (job 2371747).
* Two faults in the chain fixed: a checkpoint truncated by a time limit
  (atomic save, `18f0611`) and a 50–70-minute teacher load (teacher copied
  into node memory, `9d093b7`).
* 101.6 GPU-hours in the day, the first above the 82 a day the plan needs.

**16-10-2026, night**
* Qwen3.5 distillation on legislation, 4B and 9B (jobs 2349819–2350236), and
  instruction tuning of the Qwen3.5 and Qwen3 students on 9,223 pairs, every
  model on the exam closed and open book.

**14-10-2026**
* Login-node probe of the environment (`forge/scripts/jupiter/probe_env.sh`):
  aarch64, `booster` hands out whole nodes (a job pays 4 GPU-hours an hour),
  modules GCC 14.3, CUDA 13, Python 3.13.5, PyTorch 2.9.1.
* EULLM Engine built for sm_90 on the login node. Two build failures fixed in
  the code: the CUDA static libraries were looked for only under
  `/usr/local/cuda` (the site's toolkit is a module), and the site's GCC does
  not produce position-independent code by default (the binary is now linked
  `-no-pie`).
* Job 2326280: compute nodes have no internet and no `git`; models and data
  are staged from a login node.
* Job 2326571: first engine comparison on GH200, EULLM Engine against
  llama-server at the same llama.cpp commit (§5.1).
* Jobs 2326997, 2326998: model-load investigation; GPU probe for Forge.
* Jobs 2327213, 2327214: loads four at once; second round of the runtime
  comparison (40 points). Job 2327297: Forge probe, stopped at its first
  import (venv path order, fixed).

**16-10-2026**
* JSC's data-protection assessment form for the pseudonymised court
  decisions: a first submission wrongly declared none, withdrawn and
  replaced (§7).
* EU acts in force in Italian (17,563 regulations and directives) being
  fetched outside JUPITER for a larger legislation corpus; the processors
  (`prepare_eurlex.py`, `prepare_legislation.py --all-acts`) are in the
  repository.
* Venv: flash-linear-attention 0.5.2 for Qwen3.5's linear attention (on the
  module's Triton 3.5.1); Qwen3.5-9B-Base downloaded; the held-out exam drawn
  from the legislation (217 questions). Queued: every model on the exam,
  instruction tuning on 9,223 pairs, the Qwen3.5 probe with the fast
  kernels, then the Qwen3.5 distillation (§5.7).

**15-10-2026, night**
* Job 2338844: third link; both legislation runs complete and merged.
  Job 2339283: the same runs packed, in 47 minutes (§5.3).
* Job 2340154: instruction tuning, four runs on one node, then the ten seed
  questions (§5.5). Job 2340155: 4,951 more pairs (§5.4).
* Job 2340254: the Qwen3.5 pair on GH200 (§5.6).

**15-10-2026**
* Job 2339281: 4,272 instruction pairs generated from the legislation corpus
  by EULLM Engine in 10 minutes of one node, 2,279 tok/s (§5.4).
* Job 2334585: first Forge numbers on GH200 (§5.2).
* Engine fix published: a model on GPFS stays memory-mapped (§5.1).
* Legislation corpus copied from Leonardo and split by article (7,064 chunks
  to train on, 234 held out).
* Jobs 2337122–2337123: the distillation failed at its first forward step
  (teacher inputs on the wrong GPU), fixed in `distill.py`. Job 2337859: the
  chain runs (§5.3).
* Sequence packing added to `distill.py`, and the EULLM Engine backend to the
  synthetic-data generator, both tested; JUPITER jobs written for each.

## 4. Environment on JUPITER

* **Forge** runs on the site's PyTorch module (torch 2.9.1, built for sm_90),
  with a virtual environment on top for Transformers 5.19.0, PEFT 0.21.2,
  Accelerate 1.15.0 and Datasets 5.1.0 — never a second torch from pip.
* **Models** in the project storage: Qwen3-30B-A3B-Base (teacher, 61 GB in
  BF16), Qwen3-4B-Base and Qwen3-8B-Base (students), and GGUF models for the
  engine. All Apache 2.0.
* **Job shape**: one node, two runs side by side, each on a GPU pair and its
  two Grace sockets (GPUs 0–1 with cores 0–143, 2–3 with 144–287), in
  two-hour links that resume from their newest checkpoint.
* The scripts: `tools/jupiter/` (engine), `forge/scripts/jupiter/` (Forge).

## 5. Results

### 5.1 EULLM Engine on GH200 (deliverable 5)

Against llama-server at the same llama.cpp commit, Q4_K_M, one GH200 a point,
tokens/s (EULLM Engine / llama-server; jobs 2326571 and 2327214):

| model | 1 request | 16 requests | 32 requests | 16 clients, 400 graded gsm8k |
|---|---|---|---|---|
| Qwen3-8B | 229 / 226 | 1,434 / 1,412 | | 1,156 / 1,066 |
| Qwen3-14B | 142 / 142 | 1,066 / 1,044 | | 783 / 738 |
| Qwen3-32B | 70 / 70 | 589 / 588 | 875 / 778 | 439 / 421 |
| Qwen3.6-27B (hybrid) | 75 / 74 | 513 / 394 | 700 / 392 | 432 / 343 |
| Qwen3.6-35B-A3B (MoE) | 218 / 216 | 888 / 723 | | 646 / 576 |

Equal on dense models one request at a time; ahead with many requests on the
hybrid and MoE models (up to 79% with 32 on Qwen3.6-27B), at the same
accuracy on the graded workload. One GH200 writes 888 tok/s with a
35B-A3B MoE, where Leonardo's synthetic-data generation ran at 58–64 tok/s;
the like-for-like run (same model, Qwen3-30B-A3B-Instruct-2507) is prepared
(§5.4).

**Model loading.** The engine first took 107–184 s to load where
llama-server seemed to take 2–12 s. Part of that was the measurement (the
runner did not count llama-server's load); the rest was real: the engine
read models into memory on network file systems, a rule measured right on
LUMI's Lustre, and on JUPITER's GPFS that is the slow way (job 2326997, one
16.8 GB model):

| | read into memory | memory-mapped |
|---|---:|---:|
| EULLM Engine | 53 s | 10 s |
| llama-server | 48 s | 4 s |

Four servers loading one 22 GB file at once: 34 s read in, 16 s mapped
(job 2327213). The engine now keeps models on GPFS mapped (commit
`ca0d19d`); Lustre keeps the rule.

**The three sites' comparison (job 2427647, 21-10-2026).** The same points
on LUMI-G, JUPITER and Leonardo, one device each (`docs/sites-comparison.md`).
On LUMI every server under continuous batching grew 1.3-2.5 GiB of VRAM an
hour with llama.cpp's HIP graphs on, until it died; on the GH200, Qwen3-32B
at 32 requests for 90 minutes held 50.8 GiB from start to end with CUDA
graphs on and off (0.00 GiB an hour, no server died, 34,425 and 34,771
requests, the same accuracy on gsm8k and ARC). CUDA builds keep the graphs.
Tokens/s on one GH200 (a GPU-hour writes 3,600 times as many):

| model | 1 request | 4 | 16 | 32 |
|---|---:|---:|---:|---:|
| Qwen3-8B | 226 | 576 | 1,430 | 1,480 |
| Qwen3-32B | 70 | 199 | 588 | 794 |
| Qwen3.6-35B-A3B | 214 | 401 | 736 | 778 |

Equal to 14-10's `next10` build at 1 and 16 requests on the dense models;
lower for Qwen3-32B at 32 (794 against 875) and Qwen3.6-35B-A3B at 16 (736
against 888), with the node shared by the 90-minute points this time: a
question for the engine, measured again before it is called a regression.

### 5.2 The distillation step on GH200 (deliverables 2 and 3)

Job 2334585, the Leonardo recipe (Qwen3-30B-A3B-Base teacher in BF16,
Qwen3-4B-Base student with LoRA r128 and gradient checkpointing, 2,048-token
micro-batches, 16 per optimiser step), two runs side by side on one node:

| | JUPITER (GH200) | Leonardo (A100 64 GB) |
|---|---|---|
| GPUs per run | 2 (teacher alone on one, in BF16) | 3 (teacher split over two) |
| One micro-batch | 0.41 s | — |
| Optimiser steps an hour, per run | **548–553** | ~380 |
| Steps per GPU-hour | ~276 | ~127 (**2.2×**) |
| Teacher memory | 57 GiB of 95 | — |

Loading the 61 GB teacher takes ~3.5 minutes per job (two processes reading
it at once). The teacher generating with plain Transformers in BF16 on one
GH200: 13.9 tok/s alone, 113.5 with 8 sequences, 346.9 with 32.

### 5.3 First distillation on legislation

Two arms on one node, 4B and 8B students, the Base teacher, three epochs over
the legislation corpus (1,324 optimiser steps), in two-hour links.

* First attempt (jobs 2337122–2337123, 13 minutes of node time): both runs
  stopped at the first forward step. The teacher, alone on one GPU, gets no
  device hooks from Accelerate, and the batch stayed on the student's GPU.
  Fixed in `distill.py` (commit `ce822dc`) with a test.
* Jobs 2337859, 2337861, 2338844: both runs train, unattended, link after
  link. At 1 h 39 min the 4B was at step 560 and the 8B at step 500
  (evaluation at step 400: loss 0.989, KL 0.824, CE 1.376); the second link
  ended with both at step 1,180 of 1,324, and the third resumes from the
  checkpoint of step 1,100.
* **Throughput lesson.** The legislation chunks are short (~210 tokens), and
  one chunk a micro-batch ran at 1.5–1.6 micro-batches a second, ~340
  tokens/s against ~5,000 with 2,048-token micro-batches: the fixed cost of
  each forward was the whole cost. Sequence packing (several chunks in one
  2,048-token sequence, each attending only to itself; commit `ebc56c4`)
  was added and the same runs repeated packed (job 2339283).

| | unpacked (2337859, 2337861, 2338844) | packed (2339283) |
|---|---|---|
| Node time, both runs, 3 epochs | 4 h 28 min | **47 min** |
| GPU-hours an epoch, one run | 2.98 | **0.52** |
| Optimiser steps | 1,324 | 206 |
| 4B, held-out loss / KL / CE | 1.021 / 0.854 / 1.412 (step 1,200) | 1.013 / 0.888 / 1.304 (step 200) |
| 8B, held-out loss / KL / CE | 0.938 / 0.762 / 1.347 (step 1,200) | 0.923 / 0.790 / 1.232 (step 200) |

Packed reaches the same held-out loss in a sixth of the node time: KL to
the teacher slightly higher, cross-entropy on the text lower. Packing is
the default for short-document corpora from now on, for the models it is
correct for (not Qwen3.5, §5.6).

### 5.4 Synthetic instruction data with EULLM Engine (deliverable 5)

Job 2339281: one EULLM Engine server per GH200, 32 requests each,
Qwen3-30B-A3B-Instruct-2507 (Q8_0 GGUF), the generator Leonardo ran in plain
Transformers, with the same prompts, sampling and quality filters, on the
training split of the legislation corpus:

| | JUPITER, EULLM Engine | Leonardo, Transformers |
|---|---|---|
| GPUs | 4 GH200 (one node) | 3 A100 64 GB |
| Generation speed | **2,279 tok/s** | 58 tok/s (pilot, job 58559305) |
| Per GPU | ~570 tok/s | ~19 tok/s |
| Pairs | 4,272 accepted of 4,384 (97%) in 8.4 min of generation, 10 min of node | 6,000 estimated at 7–8 node-hours |
| Energy (GPUs) | 0.20 kWh, ~5,700 generated tokens per Wh | — |

39 times the throughput per node, 29 times per GPU. The 112 rejected pairs
were turned down by the filters: 98 referred to "the text" instead of
answering, 11 were about a single case, 3 were not valid JSON. The weights
differ in precision (Q8_0 against BF16); the quality of the pairs is
measured downstream, in the instruction tuning.

Job 2340155 asked the same passages again with three more seeds: 4,951 new
pairs in 10 min 52 s of node time (0.22 kWh), 9,223 in all. Fewer than three
times 4,272, because a short passage asked the same task gives the same
question, which the generator skips as already done; more pairs need more
text, not more seeds.

Two more runs on 18-10-2026, the same four servers:

| job | pairs | from | accepted | node time | GPU energy |
|---|---|---|---:|---:|---:|
| 2390452 | instruction pairs | 244,773 EU-act chunks | 19,141 of 20,000 (96%) | 47 min | 1.17 kWh (1,469 W mean) |
| 2403329 | open-book pairs (`FORGE_GEN_KIND=openbook`) | the Italian codes, the exam's articles left out | 14,209 of 14,374 (99%) | 19 min | 0.40 kWh |

An open-book pair is the exam's task: a question, the articles retrieval
finds for it (the right one among others), and an answer from them that
cites the article; 15% ask about an article that does not exist and are
answered as such (Leonardo's `generate_openbook_pairs.py`, now with the
engine as teacher, `f3557cb`).

### 5.5 Instruction tuning (deliverable 4) and the first held-out check (6)

Job 2340154: Leonardo's stage 3 (LoRA r32, 2 epochs, the identity examples
mixed in) on the 4,272 pairs, four runs side by side on one node, one GH200
each; then each model merged and asked the ten legal-it seed questions
(keyword coverage), with Qwen3-4B-Instruct-2507 as the reference.

| run | training on one GH200 | keyword coverage | ended its turn |
|---|---|---|---|
| 4B distilled on legislation | 24.4 min | 0.385 | 10/10 |
| 4B Base, not distilled | 24.5 min | 0.410 | 9/10 |
| 8B distilled on legislation | 28.3 min | 0.447 | 10/10 |
| 8B Base, not distilled | 27.2 min | 0.455 | 10/10 |
| Qwen3-4B-Instruct-2507 (reference) | — | 0.448 | 3/10 |

Leonardo's figure is 46 minutes for 3,400 pairs on one A100: 0.81 s a pair,
here 0.34 s, **2.4 times per GPU**, and four runs at once on a node.

On the ten questions no model is ahead of another: the spread is what ten
questions give by chance, and distillation on 1.5M tokens of legislation
shows nothing there. Ten questions cannot tell, so the exam drawn from the
law (217 questions on article contents, deadlines and articles that do not
exist; reported in totals only) is now sat by every model (§5.7).

### 5.6 The Qwen3.5 pair on GH200 (deliverable 7)

Job 2340254, Qwen3.5-35B-A3B-Base teaching Qwen3.5-4B-Base (Apache 2.0,
same 248,320-token vocabulary), two pairs side by side on one node:

| | Qwen3.5 (2340254) | Qwen3 (2334585) |
|---|---|---|
| Teacher load / memory | 69 s / 64.6 GiB | ~200 s / 57 GiB |
| 2,048-token micro-batch | 1.79 s | 0.41 s |
| Optimiser steps an hour at 16 | 125 | 548–553 |
| Teacher generating, 1 / 8 / 32 sequences | 11.4 / 92.7 / 369.6 tok/s | 13.9 / 113.5 / 346.9 tok/s |

Transformers loads the multimodal checkpoints as text-only models, and
generation runs as fast as Qwen3's. Training was four times slower because
the linear-attention layers (three in four) ran Transformers' reference
PyTorch path: flash-linear-attention was not installed. It is now (0.5.2,
on the module's Triton). Packing is not used for Qwen3.5: its
linear-attention layers carry a state along the sequence that the
per-document mask does not reset.

### 5.7 Running on 16-10-2026

* Every model of §5.5 on the 217-question exam.
* Instruction tuning on the 9,223 pairs: the 4B distilled unpacked and
  packed, the 4B Base, the 8B packed (packed against unpacked, distilled
  against not, and more pairs).
* The Qwen3.5 probe again with the fast kernels, then the Qwen3.5
  distillation on legislation, 4B and 9B students, in two-hour links.

### 5.8 The held-out exam, graded (17-10-2026, deliverable 6)

Every model of §5.5–5.7 sat the 217-question exam twice: closed book, and
open book, each question with the articles BM25 finds for it in the
legislation in front. Qwen3-30B-A3B-Instruct-2507 graded every answer
against the text of the article (correct, partial, wrong; rubric v3, as on
Leonardo; score = correct + half the partial, over 217), four graders on
one node (job 2371747, 34 minutes, 0.44 kWh). Totals only: the questions
stay held out.

| model (instruction-tuned on the same pairs) | open book | correct |
|---|---:|---:|
| Qwen3.5-9B distilled on legislation | **0.811** | 175 |
| Qwen3.5-9B Base | 0.804 | 172 |
| Qwen3.5-4B Base | 0.781 | 168 |
| Qwen3-8B distilled, packed (9,223 pairs) | 0.767 | 165 |
| Qwen3-4B-Instruct-2507 (reference, as released) | 0.765 | 159 |
| Qwen3.5-4B distilled on legislation | 0.765 | 164 |
| Qwen3-8B distilled (4,272 pairs) | 0.737 | 158 |
| Qwen3-4B distilled, packed (9,223 pairs) | 0.730 | 157 |
| Qwen3-4B Base (4,272 / 9,223 pairs) | 0.717 / 0.714 | 153 / 152 |
| Qwen3-8B Base (4,272 pairs) | 0.712 | 154 |
| Qwen3-4B distilled (9,223 / 4,272 pairs) | 0.703 / 0.694 | 147 / 149 |

**The instruction pairs overlap the exam.** The exam was drawn from every
Normattiva article and the pairs from passages of the training split, with
no article left out of either. Traced back to their passages
(`pairs_exam_overlap.py`, 18-10-2026): 251 of the 9,223 pairs ask about an
exam article, and 105 of the exam's 153 articles have pairs about them. The
reference was trained on none of them, so every figure of ours above may be
inflated against it. Rerun on the 8,972 pairs that touch no exam article
(jobs 2390511 and 2390513, 18-10-2026), four Base models side by side:

| model, instruction-tuned on the clean pairs | open book | correct | training, one GH200 |
|---|---:|---:|---:|
| Qwen3.5-9B Base | **0.790** | 171 | 47.7 min |
| Qwen3.5-4B Base | **0.786** | 168 | 41.8 min |
| Qwen3-4B-Instruct-2507 (reference) | 0.765 | 159 | — |
| Qwen3-8B Base | 0.747 | 159 | 59.6 min |
| Qwen3-4B Base | 0.684 | 146 | 48.3 min |

Every model moves by at most a few questions against its run on the
overlapping pairs: the overlap inflated little. Both Qwen3.5 models stay
ahead of the reference (+9 and +12 correct answers of 217) on pairs that
never saw an exam article; this is the comparison that stands.

Two more questions on the same clean pairs and exam (jobs 2393734–2393736,
2398718–2398719, 18-10-2026):

| question | model | without | with |
|---|---|---:|---:|
| Does distillation on norms-v2 (139M tokens, one epoch) help? | Qwen3-4B | 0.684 | 0.689 |
| | Qwen3-8B | 0.747 | 0.744 |
| Do 19,141 more pairs from EU acts help (28,113 in all)? | Qwen3.5-9B | 0.790 | 0.776 |
| | Qwen3.5-4B | 0.786 | 0.747 |
| | Qwen3-4B | 0.684 | 0.664 |

Distillation on a corpus a hundred times larger still moves nothing open
book. Three times the pairs, two in three of them about EU acts, lower
every score on an exam about Italian codes (by 3 to 9 correct answers):
pairs about the text the model will be asked about count, more pairs about
other text do not.

**Open-book pairs (19-10-2026).** The 14,209 open-book pairs (§5.4), alone
(`ob`) and with the 8,972 clean instruction pairs (`mix`), two epochs, the
same four models, the same exam (jobs 2403330–2403558, graded by 2413383):

| model | clean pairs | open-book pairs | both |
|---|---:|---:|---:|
| Qwen3.5-4B Base | 0.786 | 0.864 | **0.878** |
| Qwen3.5-9B Base | 0.790 | **0.876** | 0.820 |
| Qwen3-8B Base | 0.747 | **0.866** | 0.836 |
| Qwen3-4B Base | 0.684 | 0.839 | **0.848** |
| Qwen3-4B-Instruct-2507 (reference) | 0.765 | | |

Pairs in the exam's own form raise every model by 7.8 to 15.5 points, and
the best, Qwen3.5-4B with both sets, answers 186 of 217 right against 159
for the reference. Whether the clean instruction pairs add to them is mixed
(up for the 4B models, down for the 9B and the 8B), as Leonardo found when it
trained on open-book pairs alone. The judge loaded its grader from node
memory: 18 minutes for 20 answer sets.

**Leonardo's open-book recipe (21-10-2026).** The same 14,209 open-book
pairs, one epoch instead of two, as Leonardo trained its published models
(lr 1e-4, 4,096-token sequences, one pair at a time, 16 accumulated;
instruct bases keep their chat rows, Ministral our system prompt in place of
its "Le Chat" one), and the exam also asked without the note on missing
articles, as the product asks (jobs 2429166-2429167, graded by 2429168):

| model, one epoch on the open-book pairs | open book | correct | without the note | correct |
|---|---:|---:|---:|---:|
| Ministral-3-8B-Instruct-2512 | **0.887** | **189** | **0.887** | **189** |
| Qwen3.5-4B Base | 0.880 | 187 | 0.841 | 178 |
| Qwen3-4B-Instruct-2507 | 0.869 | 184 | 0.869 | 184 |
| Qwen3.5-9B Base | 0.862 | 182 | 0.848 | 179 |
| Qwen3-4B-Instruct-2507, not trained (reference) | 0.765 | 159 | 0.765 | 159 |

One epoch does as well as two (Qwen3.5-4B 0.880 against 0.864, the 9B
0.862 against 0.876), in half the training. Ministral-3-8B-Instruct, the
base of Leonardo's published `legal-it-8b`, is the best model on this exam
so far, 189 of 217 right, and loses nothing without the note; so does the
trained Qwen3-4B-Instruct (184, 25 more than untrained). The Base models
lose 3 to 9 answers without it: an instruct start reads missing articles
right without being told. The top four are within 7 questions of each
other, less than the exam separates reliably; their lead over the reference
(+23 to +30) is not. Closed book every model stays at 0.000-0.023.

**Distilled or not, under the open-book recipe (22-10-2026).** The
experiment Leonardo listed and never ran (`docs/legal-it-experiments.md`,
§6, item 3), on the law instead of the rulings: Qwen3 4B and 8B distilled
on norms-v2 (139M tokens, one epoch, §5.9) against the same Base models,
all four through the same recipe (one epoch on the open-book pairs), the
exam with and without the note (jobs 2455477, graded by 2455480):

| model | open book | correct | without the note | correct |
|---|---:|---:|---:|---:|
| Qwen3-8B distilled on norms-v2 | **0.873** | **184** | **0.866** | **183** |
| Qwen3-8B Base | 0.859 | 181 | 0.827 | 174 |
| Qwen3-4B distilled on norms-v2 | 0.855 | 178 | 0.832 | 173 |
| Qwen3-4B Base | 0.836 | 176 | 0.818 | 172 |
| Qwen3-4B-Instruct-2507 (reference) | 0.765 | 159 | 0.765 | 159 |

The distilled model is ahead in all four comparisons, by 1 to 3 questions
with the note and by 1 (4B) and 9 (8B) without it: the distilled 8B loses 1
answer when the note is taken away, its Base 7. With the note the margins
are within what the exam separates; without it, the 8B's is the largest
difference distillation has made on this exam. Whether it holds for
Qwen3.5 is the same comparison on the 4B and the 9B, once their
distillation on norms-v2 ends.

**Qwen3.5 halfway (23-10-2026).** The same comparison on the Qwen3.5
students of the norms-v2 chain at the checkpoints of 22-10 (4B at step
8,500, 9B at 9,100, of the epoch's 15,625), each merged into its Base by the
job and put through the same recipe (job 2466380, graded by 2466381),
against the Base models of the recipe above:

| model | open book | correct | without the note | correct |
|---|---:|---:|---:|---:|
| Qwen3.5-9B distilled on norms-v2, step 9,100 | **0.892** | **189** | **0.882** | **187** |
| Qwen3.5-9B Base | 0.862 | 182 | 0.848 | 179 |
| Qwen3.5-4B distilled on norms-v2, step 8,500 | 0.878 | 186 | 0.836 | 177 |
| Qwen3.5-4B Base | 0.880 | 187 | 0.841 | 178 |

Distillation moves the 9B by 7 questions with the note and 8 without it,
the largest gain it has made on this exam, and the 9B ties
Ministral-3-8B-Instruct, the best so far (189 and 187); the 4B stays where
its Base is (1 question less in both). As with Qwen3, the larger student
gains and the smaller one hardly does.

**Qwen3.5 at the end of the epoch (24-10-2026).** The same recipe on the two
students at step 15,625, merged by the distillation itself (job 2480052,
graded by 2480053):

| model | open book | correct | without the note | correct |
|---|---:|---:|---:|---:|
| Qwen3.5-9B distilled, whole epoch | 0.882 | 188 | 0.878 | 187 |
| Qwen3.5-9B distilled, step 9,100 | 0.892 | 189 | 0.882 | 187 |
| Qwen3.5-9B Base | 0.862 | 182 | 0.848 | 179 |
| Qwen3.5-4B distilled, whole epoch | **0.892** | **189** | **0.864** | **183** |
| Qwen3.5-4B distilled, step 8,500 | 0.878 | 186 | 0.836 | 177 |
| Qwen3.5-4B Base | 0.880 | 187 | 0.841 | 178 |
| Ministral-3-8B-Instruct | 0.887 | 189 | 0.887 | 189 |

The 9B keeps the gain it had halfway (+6 and +8 over its Base) and adds
nothing in the second half; the 4B, level with its Base halfway, ends 2 and
5 questions ahead. The held-out loss kept falling over the second half (the
9B from 0.814 at step 6,000 to 0.786 at 9,500 and on), so the students go on
learning the law while the exam stays where it was: five models are now
within 189 of 217 open book, and the differences between them, 1 to 3
questions, are less than this exam separates. Whatever more distillation
adds, this exam no longer measures it.

**The complete vertical (22-10-2026, deliverables 1 and 8).** The two best
models of the recipe above, merged, made into what ships and examined as it
ships (`sbatch_vertical.slurm`, jobs 2447162 and 2449393, graded by 2447164
and 2449394): converted to
GGUF F16 by llama.cpp's converter at the commit EULLM Engine is built on,
quantized to Q4_K_M, and asked the same exam through EULLM Engine
(`legal_eval.py --runtime eullm`, the prompt sent raw, 16 questions at once
in continuous batching) and through llama-server at once:

| model, open-book recipe | open book | correct | without the note | correct |
|---|---:|---:|---:|---:|
| Ministral-3-8B-Instruct, bf16, Transformers | 0.887 | 189 | 0.887 | 189 |
| Ministral-3-8B-Instruct, Q4_K_M, EULLM Engine | **0.889** | **189** | 0.882 | 187 |
| Ministral-3-8B-Instruct, Q4_K_M, llama-server | 0.885 | 188 | 0.887 | 188 |
| Qwen3.5-4B, bf16, Transformers | 0.880 | 187 | 0.841 | 178 |
| Qwen3.5-4B, Q4_K_M, EULLM Engine | 0.887 | 189 | 0.841 | 178 |
| Qwen3.5-4B, Q4_K_M, llama-server | 0.878 | 186 | 0.850 | 180 |

| stage (seconds) | Qwen3.5-4B | Ministral-3-8B |
|---|---:|---:|
| instruction tuning, one GH200 | 8,029 | (two links, the same job) |
| GGUF F16 | 78 | 96 |
| Q4_K_M | 13 | 41 |
| the 217-question exam through EULLM Engine | 66 | |

Quantization to Q4_K_M and the engine cost nothing the exam can see: the
answer sets of each model are within 3 questions of each other, in both
directions, and Ministral without the note keeps its lead (187-189).
From merged weights to a graded GGUF served by EULLM Engine takes about two
and a half minutes of a node; the vertical is its instruction tuning. The
first try for Ministral-3-8B-Instruct stopped at the conversion, which
imports `sentencepiece` before it falls back to the HF tokenizer (`03821b8`);
its rerun took 5.5 minutes of a node.

Why, as far as the data say. Distillation did learn the law: the held-out
loss fell from 0.914 to 0.674 for the 4B and from 0.827 to 0.625 for the 8B
over the epoch (evaluations at steps 200 and 7,200). Open book, though, the
article is in the prompt, so knowing it adds little; closed book, the
exam's questions (deadlines, the content of a given article) are beyond what
any of the models recalls. And instruction tuning teaches a behaviour, not
facts: 44% of the clean pairs and 59% of the EU pairs are questions answered
from memory, with no text in the prompt (the generator rejects questions
that refer to a text), and none has the retrieved articles in front of the
answer. More of them teach the model to answer without reading. The next
pairs are the exam's own task (`generate_openbook_pairs.py`, Leonardo's
open-book pairs): a question, the articles retrieval finds for it, the
answer from them citing the article, and questions about articles that do
not exist, answered as such; now generated by EULLM Engine (`f3557cb`).

Closed book every model, the reference included, scores 0.000–0.023: none
knows the articles by heart. Open book, the two Qwen3.5-9B models score
ahead of the reference (+10 questions of 217), and still are without the
overlap (above). A
model distilled and the same model not distilled stay within a few
questions of each other in every pair: 1.5M tokens of legislation move
nothing measurable; the family and the size of the student do. Whether a
corpus a hundred times larger does is what norms-v2 asks (§5.9).

### 5.9 norms-v2: the codes with the EU acts in Italian (from 17-10-2026)

The Italian codes with 17,529 EU acts in force, in Italian, fetched from the
Publications Office (`prepare_eurlex.py`): 392,538 training chunks, ~220M
tokens. A fixed sample of 250,000 (~139M tokens) and 3,000 held out is the
corpus of a packed distillation chain, 4B and 8B students, one epoch, 7,224
optimiser steps, two-hour links.

* A link killed by its time limit while saving left a truncated
  `training_state.pt` (8B, checkpoint-1100), and every later link of the 8B
  failed at once reading it. The state is now written under a temporary name
  and renamed, and a truncated one is skipped for the one before
  (`18f0611`); the 8B resumed from checkpoint-1000 (job 2371817).
* Each link spent 50–70 minutes of its two hours loading the teacher (links
  2362188–2362191 and 2371817; 3.6 minutes for the first link, 12 for
  2371816), against 3.5 minutes on 14-10: the 61 GB read from `$PROJECT`
  twice, once per run, in many small reads at once, as fast as the shared
  file system allows at that hour. The four graders of job 2371747 loaded
  theirs quickly the same day. The teacher is now copied once per node into
  memory (`/dev/shm`, a few large reads) and both runs load it from there
  (`9d093b7`), from link 2375535: 49 s for the copy, the runs training
  after 4–5 minutes instead of an hour.
* Halfway (4B at step 3,400, 8B at 2,800, jobs 2381934–2381935), instruction
  tuned and examined open book: the 4B 0.712 against 0.714 for the 4B Base,
  the 8B 0.740 against 0.751 for the 8B Base. Open book the article is in
  front of the model, so the exam measures reading it, which distillation
  on the law is not expected to change; its effect is the held-out loss.
* Qwen3.5-35B-A3B-Base teaching Qwen3.5-4B-Base and 9B-Base side by side on
  one node, unpacked (§5.6), one epoch of the 250,000 chunks (15,625 steps
  each): seven links, 2431499-2431505, from 21-10 16:37 to 23-10 21:59 UTC,
  six of 7 h 51 min and the last of 6 h 12 min, 53.3 node-hours, 213 GPU-hours
  for the two students (deliverable 7).

### 5.10 A candidate teacher: Mistral Small 4 (from 19-10-2026)

Mistral Small 4 (`mistralai/Mistral-Small-4-119B-2603`, Apache 2.0, a 119B
mixture of experts that can reason before answering), examined as Leonardo
examines its candidates: `legal_eval.py` open book, retrieval hybrid (BM25,
Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B) over the Italian codes, no note
on missing articles, once answering directly (400 new tokens) and once
reasoning first (`reasoning_effort` high, 4,000). Two question sets that are
not public: 36 trap questions and 145 practical ones. The questions and the
answers stay on the project storage, readable by the user alone, and travel
to Leonardo only over SSH; Leonardo's judge grades them with the other
candidates. The log carries counts and times only (`--quiet`).

The checkpoint is FP8. Transformers runs it through kernels it fetches from
the Hugging Face Hub, out of a compute node's reach (jobs 2414120, 2414122),
and its dequantize-on-load left the fused experts in FP8 (jobs
2418651–2418654); the model was converted once to BF16 on a login node
(`dequantize_fp8.py`, 238.8 GB) and is read from `$PROJECT` over the
node's four GPUs.

| set, mode | job | time | ended its turn | abstained | cites articles not retrieved | GPU energy |
|---|---|---:|---:|---:|---:|---:|
| traps, direct | 2419006 | 1,615 s | 15/36 | 10/36 | 7/36 | 0.28 kWh |
| traps, reasoning | 2419007 | 2,463 s | 34/36 | 10/36 | 6/36 | 0.52 kWh |
| practical, direct | 2422086 | 922 s | 45/145 | 3/145 | 12/145 | 0.22 kWh |
| practical, reasoning | 2422087 | 4,782 s | 140/145 | 2/145 | 17/145 | 1.36 kWh |
| traps, direct, 1,000 tokens | 2424181 | 868 s | 34/36 | 9/36 | 6/36 | 0.19 kWh |
| practical, direct, 1,000 tokens | 2424182 | 1,922 s | 130/145 | 3/145 | 14/145 | 0.49 kWh |

Answering directly, the model ran out of its 400 tokens in 21 of 36 and
100 of 145 answers: it writes long. With 1,000 tokens 34 of 36 and 130 of
145 end their turn, in about twice the time; reasoning first, 34 of 36 and
140 of 145, in three to five times the time of the 400-token run. The six
answer sets (543 answers) go to Leonardo's judge; the 400-token ones are
kept for comparison with the candidates examined there at that length.

**Graded on Leonardo** (its judge and rubric; the files stayed where they
are). On the 36 trap questions, without the note: 10 right answering
directly, 11 with 1,000 tokens, 12 reasoning first, against 16 for
Qwen3.6-27B and 14 for the published `legal-it-8b` (16 with our system
prompt). Citations of articles not among the retrieved texts were judged
wrong almost every time (4-5 of 6-7 on the traps, 8-12 of 12-17 on the
practical questions), a real fault for a RAG model. On the practical exam,
provisionally, 55-60 right of 145 against 60-61 for the published 8B.
Mistral Small 4 is not a better teacher in any mode, and the teacher stays
as it is: no more JUPITER time on it (`docs/legal-it-experiments.md`, §9).

### 5.11 Our models on the traps and the practical exam (24-10-2026)

The two private sets of §5.10, now asked of our own models in the same way
(`sbatch_teacher_eval.slurm`, jobs 2486201-2486208, with Ministral's system
prompt for Ministral) and graded on JUPITER by the same script and grader as
Leonardo's (`sbatch_judge.slurm` with `JUDGE_ANSWERS_DIR`, jobs 2486113 and
2486209). Mistral Small 4's answers of 20-10, graded again here, come out
within one question of Leonardo's grades (traps 10, 10 and 13 of 36 against
10, 11 and 12; practical 60, 55 and 60 of 145), so the two judges agree.
Correct answers:

| model | traps (36) | practical (145) | practical, score |
|---|---:|---:|---:|
| Qwen3.5-9B distilled, open-book recipe | 3 | **66** | **0.528** |
| Qwen3.5-4B distilled, open-book recipe | 8 | 62 | 0.500 |
| Qwen3.5-9B Base, open-book recipe | 6 | 58 | 0.459 |
| Ministral-3-8B-Instruct, open-book recipe | 11 | 65 | 0.500 |
| Mistral Small 4, direct (§5.10) | 10 | 60 | 0.417 |
| Qwen3.6-27B, on Leonardo | 16 | | |
| `legal-it-8b` published, on Leonardo | 14 | | |

On the practical exam, harder than the held-out one, distillation moves the
9B by 8 questions (58 to 66), ahead of every model asked, the 119B teacher
candidate included. On the traps it does the opposite: the 9B falls from 6
to 3 of 36, and no model of the open-book recipe reaches the published
`legal-it-8b`, whose training on Leonardo included false premises to refuse.
The recipe here teaches answering from the retrieved text, never refusing a
premise; before this is concluded, answers are read both ways (a judge has
failed right answers before, `forge/CLAUDE.md`).

**The false-premise round (24-10-2026).** Leonardo's false-premise pairs
(`make_premise_pairs.py`: a wrong deadline or penalty, an article that does
not exist, a repealed one taken for one in force, answered by correcting the
premise in three checked steps), 1,606 generated here from the legislation
alone with the exam's, the practical exam's and the traps' articles left
out (job 2488357), added to the open-book pairs; the distilled Qwen3.5 4B
and 9B tuned on both (2488358, 3 h 17 min), then all three exams
(`submit_premise_chain.sh`):

| model | traps (36) | practical (145) | held-out, open book | without the note |
|---|---:|---:|---:|---:|
| Qwen3.5-9B distilled, + premises | **11** | **67** | 188 | **188** |
| Qwen3.5-9B distilled | 3 | 66 | 188 | 187 |
| Qwen3.5-4B distilled, + premises | 9 | 63 | 184 | 181 |
| Qwen3.5-4B distilled | 8 | 62 | 189 | 183 |
| Ministral-3-8B-Instruct, open-book recipe | 11 | 65 | 189 | 189 |
| `legal-it-8b` published, on Leonardo | 14 | | | |

The 9B goes from 3 traps to 11, level with Ministral, and loses nothing
elsewhere: the best practical score so far and 188 of 217 with and without
the note. The 4B gains one trap and loses 5 and 2 questions on the held-out
exam. The distilled 9B with the premises is the candidate; the published
8B's 14 traps came with reinforcement learning on top, which this round
does not have.

The final distilled pair through the vertical (GGUF Q4_K_M, job 2493893,
graded by 2493894), against their bf16 scores of 188 and 187 (9B) and 189
and 183 (4B): the 9B 185 and 184 through EULLM Engine, 184 and 184 through
llama-server; the 4B 183 and 179, and 183 and 178. Quantization costs the
9B 3 questions, within what the judge alone moves; the 4B up to 6, more
than the Qwen3.5-4B before distillation lost (none).

The candidate, the distilled 9B with the premises, through the same vertical
(job 2496202, graded by 2496203): 191 and 187 of 217 through EULLM Engine,
186 and 187 through llama-server, against 188 and 188 in bf16. As it would
ship, Q4_K_M served by EULLM Engine, it keeps its score.

**A larger trap test (25-10-2026).** The 36 LexIA traps are too few to
steer by (the judge alone moves one or two) and must stay a test. 390
questions of the same kinds were drawn from the legislation on articles no
exam and no premise pair trained on uses (`make_trap_test.py`): a wrong
deadline, penalty or procedibility, an article that does not exist (207),
a repealed one, a ruling cited for a deadline, each with true twins that
must not be refused. Correct answers, by the judge (of 390):

| model | judge | judge-free check |
|---|---:|---:|
| Qwen3.5-4B distilled, + premises | 383 | 99.5% |
| Qwen3.5-9B distilled, + premises | 381 | 99.5% |
| Ministral-3-8B-Instruct | 284 | 27.4% |
| Qwen3.5-9B distilled | 241 | 29.0% |
| Qwen3.5-9B Base | 206 | 30.0% |

The judge-free check (`score_trap_test.py`, the GRPO rewards' check) knows
only the phrasings of the three-step format the premise pairs teach, and
fails right answers written otherwise: between formats the judge decides.
The models trained on the premises pass 98% and refuse almost no true
premise, but the test asks in the very templates they were trained on,
while LexIA, written by people, stays at 11 of 36: it measures the
templates. The same test rewritten by a teacher in three styles (a case
told by a client, a short colloquial question, a professional taking the
fact for granted; `paraphrase_trap_test.py`), every number of the question
kept, is the next measure.

Rewritten (job 2512957, 1,050 of 1,170 rewrites kept, every number of the
question in place) and asked of the same five models (2512959-2512964,
judge 2512966):

| model | templated (of 390) | rewritten (of 1,050) |
|---|---:|---:|
| Qwen3.5-9B distilled, + premises | 97.7% | **91.9%** |
| Qwen3.5-4B distilled, + premises | 98.2% | 91.0% |
| Ministral-3-8B-Instruct | 72.8% | 62.5% |
| Qwen3.5-9B distilled | 61.8% | 52.2% |
| Qwen3.5-9B Base | 52.8% | 45.9% |

Other words cost every model 6 to 10 points, and the models trained on the
premises keep their lead of about 40 over the same model without them. So
the phrasing does not explain the 11 of 36 on LexIA: its traps are, in
good part, of kinds the generated premises never ask (a principle
attributed to a ruling, a question outside the corpus, an article that
retrieval cannot find), as Leonardo found.

**GRPO on JUPITER (prepared 25-10-2026).** On Leonardo the best 8B, r2l,
came from the same false premises followed by GRPO on practical questions
and on premises: 87 of 145 on the practical exam against 60 for the
published model, and 14 LexIA traps. The same recipe is ported here for
the distilled Qwen3.5-9B with the premises (`submit_grpo_round.sh`):

* `grpo_train.py` unchanged; LoRA rank 32, eight answers a prompt, learning
  rate 1e-5, as in Leonardo's rounds.
* One node: the policy on GPUs 0-2, one process each. The judge is on
  GPU 3. It grades the practical questions against their article with
  rubric v3, and is Qwen3-30B-A3B-Instruct-2507 Q8_0 served by EULLM
  Engine. Leonardo served the same judge with llama-server. The verifiable
  rewards grade the premises.
* The prompts are Leonardo's. They leave out Leonardo's exams but not the
  trap test drawn here, so `filter_grpo_prompts.py` drops every prompt on
  an article of that test, the private sets or the held-out exam, as
  `make_trap_test.py` does.
* Before the training, a smoke test of three steps on 48 prompts. The
  training waits for it to succeed.
* The round ends with five sets asked open book without the note: the
  trap test, its rewrites, the LexIA traps, the practical exam and the
  held-out exam. The model it starts from is asked the sets it has no
  answers for, and the judge then grades both models.

First round, g9-r1 (26-10-2026), on 2,184 of r2o's prompts, 300 steps in
two hours. Judge's grades, correct + partial:

| set | dqp 9B (start) | g9-r1 |
|---|---:|---:|
| practical exam (145) | 67 + 21 | **79 + 16** |
| held-out exam, open book, no note (217) | 189 + 9 | **198 + 3** |
| LexIA traps (36) | 11 | 9 |
| trap test (390) | 381 | 384 |
| trap test rewritten (1,050) | 965 | 964 |

Paired (`compare_graded.py`, questions only g9-r1 got right : only the
start did):

* Practical exam: 17:5, p=0.017 strict. Counting "partial" as right it is
  12:5, p=0.14.
* Held-out exam: 12:3, p=0.035 strict; 6:3, p=0.51 lenient. The judge-free
  check is 3:1.
* LexIA: 0:2, p=0.50.
* Trap test: 3:0. Its rewritten version: 13:14.

By the rule we keep, a difference is real only if it holds both strict and
lenient, so none is shown yet. What GRPO did is mostly turn partial
answers into complete ones (21 to 16 partials on the practical exam, 9 to
3 on the held-out exam). On Leonardo r2h's gain was 29:6 strict, p<0.001.
The second round, g9r-r1 (from the 9B that the rest of norms-v2 had
helped on LexIA; jobs 2529733-2529738), uses the same prompts.

The training log says why the gain is thin. A step had three prompts of
eight answers each. In most steps two of the three groups, or all three,
had every answer graded the same (zero-std 0.67-1.00), and the mean
reward stayed between 0.6 and 1.0. Only about a third of the groups
taught anything. The 300 steps also saw 900 prompts, 41% of the 2,184.
The next round therefore takes six prompts a step (`--grad-accum 2`, the
same memory) and another order of the prompts (`--seed`).

## 6. Energy

Slurm keeps no energy for jobs on JUPITER Booster (`ConsumedEnergy` empty), so
every Forge job samples the GPUs' power with `nvidia-smi` (every 1–5 s) and
sums it (`forge/scripts/jupiter/energy.py`). Idle, a node's four GPUs draw
~390 W together (job 2327297). The carbon footprint will be derived from the
figures below with the site's energy mix.

| job | what | GPU energy |
|---|---|---|
| 2339281 | 4,272 instruction pairs, EULLM Engine | 0.20 kWh (1,359 W mean) |
| 2340155 | 4,951 more pairs | 0.22 kWh (1,357 W mean) |
| 2340154 | four instruction-tuning runs and their evaluation | 1.21 kWh in 1.43 h (844 W mean: the evaluations leave the GPUs mostly idle) |
| 2340254 | Qwen3.5 probe | 29 Wh training loops, 15 Wh generating |
| 2371747 | the judge, 30 answer sets of 217 questions | 0.44 kWh in 0.57 h (765 W mean) |
| 2390452 | 19,141 instruction pairs from the EU acts, EULLM Engine | 1.17 kWh in 0.79 h (1,469 W mean) |
| 2403329 | 14,209 open-book pairs, EULLM Engine | 0.40 kWh in 0.31 h (1,269 W mean) |
| 2419006, 2419007 | Mistral Small 4 on 36 questions, direct and reasoning | 0.28 and 0.52 kWh |
| 2422086 | Mistral Small 4 on 145 questions, direct | 0.22 kWh in 0.26 h (856 W mean) |
| 2422087 | Mistral Small 4 on 145 questions, reasoning | 1.36 kWh in 1.33 h (1,021 W mean) |
| 2424181, 2424182 | Mistral Small 4, direct, 1,000 tokens, 36 and 145 questions | 0.19 kWh in 0.24 h, 0.49 kWh in 0.53 h |

## 7. Data governance

* **What the proposal declared.** The project handles personal data:
  pseudonymised Italian court decisions (Corte di Cassazione 2021–2026,
  Consiglio di Stato 2017–2024), as the call's form required us to state,
  and we committed to a data processing agreement with the production site
  before any of it is uploaded. The Principal Investigator's Agreement
  (14-10-2026) is JSC's standard form, which excludes personal data without
  such an agreement.
* **The agreement is being sought.** On 15-10-2026 JSC was asked by e-mail
  for it; JSC answered with its Data Protection Team's assessment form, the
  step before the addendum to its terms of use. On 16-10-2026 the form was
  first submitted by mistake declaring no personal data; that submission is
  withdrawn by e-mail to the Data Protection Team and replaced by one for
  the pseudonymised decisions.
* **22-10-2026: JSC's answer.** The data as described fall under the GDPR;
  JSC's Data Protection Team asks for its checklist (data, purpose, who
  processes and accesses them, storage, legal basis, retention, data
  minimisation, technical and organisational measures), to assess them
  with its DPO, and for any prior approval of ours or consent form.
* **Until the addendum is signed** JUPITER holds legislation (Normattiva,
  EUR-Lex: public), public benchmarks, synthetic pairs generated from them
  and ruling summaries that carry nothing of the case, and no court
  decision is uploaded. The teacher on JUPITER is accordingly the Base
  model, without the continued-pre-training adapter trained on the
  decisions on Leonardo.
* **The measures declared, which bind the processing once allowed:** names
  of parties, lawyers and magistrates replaced; tax codes, VAT numbers,
  IBANs, addresses, phone numbers, e-mail addresses and dates of birth
  removed; surnames in case citations and links to the original documents
  removed; a document still holding a structured identifier rejected by an
  automatic check; generated examples about a single case, or still holding
  a placeholder, discarded; the data only on the project's storage, moved
  only over SSH, never in a repository or a third-party service, and deleted
  at the end of the project; decisions of 2025–2026 used for evaluation
  only, never for training.

## 8. Issues found and fixed

| when | issue | fix |
|---|---|---|
| 14-10 | Engine build could not find CUDA's static libraries in the site's toolkit | build looks in the toolkit `CUDA_PATH` names |
| 14-10 | Link error: site GCC does not make position-independent code | `-no-pie` for the engine binary |
| 14-10 | Benchmark runner counted models as installed from the catalogue listing | reads only the installed-models table |
| 14-10 | Load times compared unlike things (llama-server's load not counted) | load test timed from process start (`sbatch_load.slurm`) |
| 15-10 | Engine read models into memory on GPFS, 5× slower than mapping | GPFS models stay mapped (`ca0d19d`) |
| 15-10 | Forge venv: the module's older packages shadowed the venv's in jobs | venv first on the path in every job, imports checked before using a GPU |
| 15-10 | Teacher alone on one GPU received its inputs on the student's GPU | inputs moved to the teacher's device (`ce822dc`) |
| 15-10 | Short documents left the GPUs idle (~340 tokens/s) | sequence packing (`ebc56c4`): the same runs in a sixth of the time |
| 16-10 | Qwen3.5's linear attention ran on Transformers' reference PyTorch path, 4× slower than Qwen3 | flash-linear-attention 0.5.2 in the venv |
| 16-10 | flash-linear-attention 0.5.2 refuses the module's Triton 3.5.1 on Hopper (wrong gradients up to 3.7.0) | Triton 3.8.0 in a directory of its own for the Qwen3.5 jobs (`9f04bd6`) |
| 17-10 | A link killed while saving left a truncated checkpoint; every later link failed on it | atomic save, truncated state skipped on resume (`18f0611`) |
| 17-10 | Loading the teacher took 50–70 minutes of a two-hour link | teacher copied once per node into memory (`9d093b7`) |
| 18-10 | 105 of the exam's 153 articles had instruction pairs about them | overlap traced (`d746b11`); instruction tuning rerun on the 8,972 pairs without it: scores within a few questions |
| 19-10 | Judges stopped at their one-hour limit, the grader loading for 3,495 s from `$PROJECT` | grader copied into node memory, two-hour limit (`3746ea8`): 18 minutes |
| 19-10 | Mistral Small 4's FP8 weights need kernels from the Hub, which compute nodes cannot reach | — |
| 20-10 | Dequantizing on loading left the fused experts in FP8 | converted once to BF16 on a login node (`dequantize_fp8.py`, `1059ee8`, `2a7c5fc`) |
| 20-10 | The Hub client refused a snapshot without the consolidated files | the snapshot directory found in the cache (`a08780b`) |
| 20-10 | GPU 0, also holding the retrieval models, out of memory at 48 questions a batch | 24 a batch, expandable CUDA segments (`6691261`) |
| 22-10 | The chain's submission refused distillation checkpoints given as "base+adapter", which the job loads; the second node idle overnight | the check follows the job (`d51cee9`) |
| 22-10 | llama.cpp's converter stopped on Ministral: it imports `sentencepiece` before looking for a tokenizer.model | `sentencepiece` 0.2.2 in the venv (`03821b8`) |
| 20-10 | sbatch failed once with "Connection reset by peer" and the script reported the job submitted | `sites.sh` tries again and says when nothing was submitted (`7eca038`) |

## 9. Resource use

| job | what | node time |
|---|---|---|
| 2326280 | engine comparison, first try (models missing) | 2 min 27 s |
| 2326571 | engine comparison, 22 points + conversations | 17 min 23 s |
| 2326997 | model-load test, six loads | 10 min 9 s |
| 2326998 | GPU probe for Forge | 1 min 15 s |
| 2327213 | model-load test, four servers at once | 2 min 0 s |
| 2327214, 2327215 | engine comparison, 40 points | 15 min 54 s |
| 2327297 | Forge probe, stopped at the first import | 1 min 22 s |
| 2334585 | Forge probe: distillation step, generation, energy | ~5 min 42 s |
| 2337122, 2337123 | distillation, stopped at the first forward | 13 min 16 s |
| **completed** | | **~69.5 min of node time ≈ 4.6 GPU-hours** |
| 2337859, 2337861 | distillation chain on legislation, links 1–2 | 3 h 43 min |
| 2339281 | synthetic instruction data, EULLM Engine | 10 min 2 s |
| 2338844 | distillation chain, link 3 | 44 min 52 s |
| 2339283 | distillation on legislation, packed | 47 min 12 s |
| 2340154 | instruction tuning, four runs, evaluation | 1 h 27 min 30 s |
| 2340155 | 4,951 more instruction pairs | 10 min 52 s |
| 2340254 | Qwen3.5 probe | 5 min 14 s |

From 16-10-2026 the jobs are too many to list one by one; `budget.sh` splits
Slurm's accounting by day (GPU-hours; a node pays 4 an hour):

| day | 14-10 | 15-10 | 16-10 | 17-10 | 18-10 | 19-10 |
|---|---:|---:|---:|---:|---:|---:|
| GPU-hours | 3.4 | 29.9 | 30.6 | 101.6 | 117.5 | 34.6 |

207.4 in all by the morning of 18-10-2026: distillation 162.2 (120.2 of it
norms-v2), instruction tuning 30.5, the judge 8.4, engine and probes the
rest. 317.4 by 20-10-2026 in JSC's count (`jutil user cpuquota`); 19-10 was
low because JUPITER was down most of the day.

## 10. Next

1. Whether distillation on the law helps under the open-book recipe: yes
   for the larger students (Qwen3-8B +9 without the note, Qwen3.5-9B +7 and
   +8 halfway), hardly for the 4B ones (§5.8). The Qwen3.5 pair at the end
   of its epoch is queued behind the distillation.
2. Reinforcement learning (GRPO), Leonardo's r2l recipe, on the distilled
   Qwen3.5-9B with the false premises (§5.11; deliverables 4, 6): ported on
   25-10, to run once Leonardo's prompts are copied here.
3. One complete vertical unattended, timed: the best Qwen3.5-4B, merged, to
   GGUF Q4_K_M, served by EULLM Engine on GH200 and examined through it
   (deliverables 1, 8).
4. The Qwen3.5 distillation on a sample of norms-v2, for deliverable 7's
   epoch figures; on this exam it is not expected to move the score.
5. The rest of Normattiva (the full collections, `prepare_legislation.py
   --all-acts`) for more open-book pairs; ruling summaries and court
   decisions once the data-protection addendum is signed (§7).
