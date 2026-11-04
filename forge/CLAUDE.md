# CLAUDE.md — EULLM Forge (Python)

Loaded automatically when Claude Code works with files under `forge/`.
Project-wide rules (Git, license, architecture) live in the repo root
`.claude/CLAUDE.md` and always apply too.

## Verticalizzazione Strategy

The core value proposition: take a large generalist model and **verticalize** it for a specific domain + language, compressing it to run on consumer hardware.

### Pipeline

```
Base model (14B–72B)
  → 1. Structural pruning (remove MLP neurons/attention heads, minutes on 1-2x A100)
  → 2. Knowledge distillation (teacher→student recovery, days on 2-8x A100)
  → 3. Identity fine-tuning (LoRA: domain corpus + branding, 1-2h on 1x A100)
       ...then MERGED into the weights — an adapter directory is not a model
  → 4. HF-level quantization (AWQ/GPTQ) — SKIPPED for GGUF targets
  → 5. GGUF export: convert to F16, then llama-quantize → Q4_K_M (minutes, CPU only)
Output: 7B Q4 model (~4.5GB) that runs on any laptop with 8GB RAM
```

### A distillation run does not produce a model at the end. It produces one continuously.

Stage 2 writes a `checkpoint-N` every `save_steps`, and **each one is already
a shippable model**. It is a complete LoRA adapter over the student base:
merge it, convert it, quantize it, and you have a GGUF that loads and
generates exactly like the run's final output. Less trained. Nothing else
about it is different.

This is easy to miss because `distill.py` merges the adapter only when the
whole run finishes, which makes the deliverable *look* gated on an epoch that
takes weeks of chained 24 h jobs on a busy cluster. It never was, and treating
it as though it were costs three things that matter:

* **Evidence.** The pipeline can be proven end to end on day three instead of
  week four — and a pipeline nobody has run to the end is a pipeline with
  unknown bugs in its last stage, which is where they are most expensive.
* **Feedback.** Quality can be measured against a real baseline while there is
  still budget left to act on what it says. A loss curve is not a model; only
  a model tells you whether the thing is any good.
* **Risk.** If an allocation expires, a node dies, or a chain breaks mid-run,
  what survives is a *model* rather than a directory of optimizer state.

`forge/scripts/export_checkpoint.py` packages any checkpoint;
`forge/scripts/leonardo/sbatch_export_gguf.slurm` does it on a cadence without
being asked, on the **serial partition** — merging is elementwise arithmetic
and GGUF conversion is CPU-only, so neither belongs inside a GPU job, where it
would spend A100 time on addition and stall the training loop while it ran.

Corollary worth stating because it inverts the usual instinct: **`save_steps`
is not only a crash-recovery knob.** It is also the sampling rate of the
deliverable, and the interval at which a run can be evaluated at all.

### Packaging without measuring is half a loop

The export job also **evaluates** what it packages: after each GGUF it runs
`perplexity_compare.sh` against a fixed base model on a fixed held-out corpus
and appends a row to `exports/perplexity.csv`. The quality curve then builds
itself alongside the run.

This was added because the alternative had already happened. On 20 September
three perplexity measurements were run by hand, twenty minutes each, and the
results existed only in a terminal scrollback — not a record, and not
something a report can cite. Worse, nothing was watching: a run that had
stopped improving, or started getting worse, would have said so only whenever
somebody next remembered to check, which on a four-week chain is a lot of
allocation spent on a hypothesis nobody is testing.

Three properties, each of which is the reason it is safe to leave running:

* **The base is measured once.** For a fixed (base, corpus, chunks, ctx) its
  perplexity is a constant, and it costs as much as the student's. It is
  cached under a key that includes the corpus size, so regenerating the corpus
  invalidates the entry instead of silently producing a plausible delta
  against a corpus the base was never measured on.
* **Evaluation never fails the export.** The GGUF is the deliverable; the
  number is a comment on it. A lost measurement is a missing row, a failed
  export is a missing model.
* **The inputs are checked before the expensive part**, not after — same
  lesson as the quantizer pre-flight, which was added after a ten-minute merge
  died at conversion for want of a binary.

Build the two inputs once per project: the base GGUF via `quantize_to_gguf.sh`
on the untouched student base, and the corpus via `make_ppl_corpus.py` from
`val.jsonl`. Interpreting what comes out is a separate discipline — a held-out
in-domain corpus answers "did this help on this domain", not "is the model
correct", and the two get conflated exactly when the number is flattering.

**Two ordering rules, both found as real bugs in July 2026 and both easy to
reintroduce:**

1. **Identity comes before quantization, and its adapter must be merged.**
   `fine_tune_identity` returns a LoRA *adapter* path, not a model.
   `pipeline.py` used to assign it to a local variable, log it, and never wire
   it into `current_model_path` — so stage 5 exported the pre-LoRA weights and
   `eullm forge --identity "…"` produced a GGUF with no identity, after paying
   for the training. Any change to `run_pipeline` must keep the exported path
   descending from the identity stage; `test_pipeline.py` asserts exactly that.
2. **AWQ/GPTQ is not on the GGUF path.** llama.cpp's `convert_hf_to_gguf.py`
   reads fp16/bf16 safetensors and cannot process `qweight`/`qzeros`/`scales`
   tensors, so "AWQ then GGUF" fails at the last stage after the whole
   pipeline has run — and it is a redundant second quantization anyway, since
   `llama-quantize` is what produces Q4_K_M. The shipped profiles set
   `quantization.method: none`; `_validate_stage_combination` rejects the
   contradictory combination up front rather than letting it fail late.

### Demo Models (Phase 1)

| Model | Domain | Source | Target | Languages |
|-------|--------|--------|--------|----------|
| `eullm/legal-it-4b` | Italian law | Qwen3-30B-A3B-Base | 4B Q4 | IT, EN |
| `eullm/medical-de-7b` | German medicine | Qwen3-14B | 7B Q4 | DE, EN |
| `eullm/finance-fr-7b` | French finance | Qwen3-14B | 7B Q4 | FR, EN |

## Compute Infrastructure

### Leonardo: the shape of a job that starts. Check every submission against this.

Every rule here was learnt by losing hours to its opposite. Before giving the
user an `sbatch`, a chain or an `scontrol update`, check it against all of
them, and look for what the scripts' headers already measured before
proposing a new request shape.

**What the Fast Lane's queue does (AIFAC_F02_1400, `fastlane_probe.sh`, 3-11-2026):**
the shapes of rules 1-2 were Playground measurements of September, and on the
Fast Lane's first day they no longer held. 3-GPU jobs started in 13-20 s at
**every** wall time up to 24 h, `boost_qos_lprod` too; a whole node for 2 h in
7 min; 4 nodes for 2 h in 29 min; two 4-node jobs ran side by side (32 GPUs)
98 s after submission; whole nodes for 8-24 h, 2, 4 and 8 nodes all started
after 105-114 min. No limit of our own on GPUs, jobs or wall time; the
partition caps a job at 24 h and 64 nodes. One day's reading, so the rules
below are the starting point and the probe is run again when the queue
seems to change (`docs/cineca/fastlane-plan.md`, "Il primo giorno").

1. **Long links for 3-GPU work, short only where they pay.** A 3-GPU job of
   up to 24 h starts at once: chain 24 h links (`submit_chain.sh ... -t
   24:00:00`, the script's own stop time raised to match, e.g.
   `law_rewrites.py --stop-after 85500`) instead of 2 h ones, which lose
   ~14 min of start-up (20 %) each. Whole nodes and multi-node jobs wait
   ~2 h for any length: worth it for work that needs the shape (the sharded
   30B MoE, the multi-node study), not as a habit. The script headers still
   say 2 h and `tests/test_leonardo_job_shape.py` still holds them to it, so
   the longer time goes on the command line; the test is revised with the
   first script that needs a whole node by design. On the Playground (from
   2026-09-21) 24 h and whole-node jobs waited for days; if the probe ever
   reads like that again, rules 1-2 go back to 2 h and 3 GPUs.
2. **Three GPUs, 24 cores, 340 GB — all three numbers — for a part of a
   node.** Four GPUs or 32 cores make the job exclusive: ask for that only
   when the whole node is meant (rule 1), never by accident through the
   cores or the memory. One-GPU jobs: 8 cores, 120 GB.
3. **Changing a queued job:** `scontrol update` takes memory in MB
   (`MinMemoryNode=348160`, not `340G`), and one invalid field fails the
   whole update silently for the others. Read the result back with
   `scontrol show job`; if a field cannot be changed, cancel and resubmit with
   explicit `--gres/--cpus-per-task/--mem/-t` (dependents first, so nothing
   behind them starts).
4. **The serial partition is small:** `lrd_all_serial` allows 10 queued and
   2 running jobs per user. Keep it for packaging only; evaluation and
   anything that can, go on `boost_usr_prod` with one GPU.
5. **Never install into the shared venv** (`$WORK/eullm_venv`), not even
   from a script: no `pip install --upgrade`, never torch or transformers.
   Extra libraries go to a `--target` directory on PYTHONPATH for the jobs
   that need them (setup_grpo_libs.sh). The known-good state is
   `$WORK/eullm_venv-known-good-2026-10-01.txt`; restoring is
   `pip install -r` of it, not guessing versions.
6. **Space:** `$WORK` is 1 TB for everything we produce. A merged 8B is
   17 GB, a 14B 28 GB, plus the same again while converting to GGUF. Check
   `df -h $WORK` before adding models. **Downloaded base models go to the
   fast scratch** (`/leonardo_scratch/fast/<project>/models`), not to
   `$WORK`: they can be downloaded again, `$WORK` reached 96% on 2026-10-10
   with two downloads. Already on `$WORK`, they are moved with
   `move_to_scratch.sh`, which copies then swaps for a symlink and so stops
   no job. The scratch is purged: never put there what only we can produce.
   It has a 1 TB quota of its own, full on 2026-10-24 with merged GRPO
   rounds: check it too (`df -h`, never `cindata`, hours old).
   See the runbook's "Storage".
7. **Every submission is idempotent and gated:** it checks the inputs exist
   and that the same chain is not already queued, and stops at the first
   failed check, so running it twice cannot duplicate jobs.

8. **The GPU queue is never empty, and nothing in it is filler.** Every
   message that ends a piece of work states what GPU work is running or
   queued next, and if there is none, proposes the next *useful* GPU job
   in that same message — the user should never be the one to notice the
   allocation is idle. On 2026-10-04 a day went by at 1.8 node-hours with
   ~530 left for 29 days, and it was the user who noticed. The aim is not
   to spend the hours; running out early with better models is fine, burning
   them on runs that answer nothing is not. `status.sh` flags an empty GPU
   queue with `[!!]`.

Rules 1 and 2 are also enforced by the build: `tests/test_leonardo_job_shape.py`
fails CI for any `boost_usr_prod` script over 2 h, over three GPUs, or over
8 cores and 123 GB per GPU. Written down, the rules were still broken by the
next new script; checked, they cannot be. The whole-node scripts of the
finished phases are listed in that test with their reason.

- **EuroHPC Leonardo Booster (CINECA) — Fast Lane EHPC-AIF-2026FL01-1400**: 30,000 GPU-hours (7,500 node hours), 02/11/2026 → 02/02/2027, no extension; plan in `docs/cineca/fastlane-plan.md`. It follows the Playground EHPC-AIF-2026PG01-1147 (1,250 node hours, 02/09/2026 → 02/11/2026). Nodes have 4x A100 **64 GB** (not 96 GB — single-GPU memory budgets do not apply there), max walltime 24 h, no internet on compute nodes. Use the `leonardo/` training configs and `forge/scripts/leonardo/`; runbook in `docs/leonardo-runbook.md`.
- **EuroHPC JUPITER Booster (JSC) — awarded EHPC-AIF-2026PG01-1434**: 5,000 GPU-hours, two months, dates set by JSC. Nodes have 4x GH200: H100 **96 GB** HBM3 with a 72-core **aarch64** Grace each — every wheel must exist for aarch64, and the 30B-A3B teacher fits one GPU in BF16 (one run = 2 GPUs, two runs per node). The probe ran on 14-10-2026: the site's PyTorch module (torch 2.9.1, sm_90) with a venv on top for the rest (`forge/scripts/jupiter/setup_venv.sh`, `env.sh`), never a second torch from pip; nodes are exclusive, so a job is a whole node with two runs side by side. Plan, probe results and deliverables in `docs/jupiter/allocation-plan.md`.
- **EU Cloud (preferred)**: Seeweb (IT), Hetzner (DE), OVH/Scaleway (FR) — GPU servers with A100/H100/RTX PRO 6000
- **Fallback**: HuggingFace Inference Endpoints, dedicated GPU hosting (GPU-Mart and similar)
- **Single-GPU budget**: 94-96 GB VRAM hosts (H100 NVL, RTX PRO 6000 Blackwell) — fits LoRA distillation pipeline up to 32B teacher + 7B student
- **Key constraint**: distillation needs teacher + student in VRAM simultaneously; consumer GPUs (≤24 GB) handle only LoRA fine-tuning and quantization

## A change for Leonardo is done when it is ON Leonardo, not when it is merged

The user merges every PR and runs every command on Leonardo; nothing reaches
the cluster by itself. A fix merged on GitHub but not pulled there changes
nothing — and on 2026-09-25 that cost real runs: stage-3 jobs were submitted
before the chat-token fix had been pulled and had to be held and re-released,
and the user had to ask, more than once, why the pull was never mentioned.

So whenever a change that affects Leonardo is pushed, the to-do given to the
user always carries BOTH steps, together, in the same message:

1. merge the PR on GitHub;
2. the exact pull on the exact checkout(s) that run it, with a one-line check
   that the change arrived (e.g. `grep -c "<new function>" <file>` → `1`).

Which checkout runs what — find it rather than assume it:
`squeue --me -h -o "%j %o" | sort -u` prints each job's script path.
As of 2026-09-25: `$WORK/eullm` (main) runs export, generation, queue stats and
stage 3; the distillation chains run from pinned trees (`$WORK/eullm-v12` for
split and r32, `$WORK/eullm-8b` for 8B), updated file by file with
`git -C <tree> checkout origin/main -- <file>`, never a blanket pull.

Two traps worth stating each time they apply:
* `sbatch` stores a copy of the batch SCRIPT at submission. A pull changes the
  Python a queued job will import, but not its `.slurm` — a job queued before
  a `.slurm` change runs the old one.
* Never tell the user to submit jobs that depend on a change until the check
  above has printed the expected value.

## Automation is tested with the thing it waits for, and checked every morning

On 2026-09-27 the open-book generator wrote an EMPTY `.done` marker and
`submit_when_ready.sh` counts only non-empty files, so stage 3 waited all
night behind a watcher that said "not yet" every twenty minutes. Each half
had its own passing test; nobody had run one after the other. Rules:

* **A producer and its consumer are tested together.** Anything that waits
  for a file (watchers, `--need`, dependent jobs) gets a test that feeds it
  the file the real producer writes — see `tests/test_submit_when_ready.py`.
  A stub of the producer's output does not count: the stub is exactly where
  the two drift apart.
* **Messages to the user carry only commands to run NOW.** Commands for later
  go in a later message. Anything conditional ("when X is done, run Y") is a
  watcher or a dependency, never the user's job to time; the user runs what
  is sent, in order, as soon as it arrives.
* **Every morning starts with `bash forge/scripts/leonardo/status.sh`**: what
  is running, what died or ended suspiciously fast, what waits on what, and
  a `[!!]` line for each thing that needs a human. If it finds a problem
  whose shape it did not know, teach it that shape.

## A number decides nothing until the answers behind it are read

Every grader we have -- the GRPO rewards, the exam's abstention check, the
judge -- has scored right answers wrong, and each time the number alone
pointed at the wrong conclusion. Between 2026-10-20 and 10-23: the deadline
reward missed "60 giorni, non 30" (0.17 instead of 0.40); the procedibility
reward missed every correction (0.05 instead of 0.69); the ruling reward
stopped at the full stop of "Cass." and a round trained on it; the teachers'
practical exams had died at load time and were compared for three days as
if measured; and the judge failed five right abstentions in the three-step
format on devbig without the note, so every candidate seemed to "lose" a
skill it had. Five of these six were found by reading answers. Rules:

* **Before a comparison decides anything** (a promotion, a recipe dropped,
  a "saturated" or "cannot be learnt"), read at least five questions each
  way: `compare_graded.py --show 5` prints them with both answers and the
  judge's reasons. A loss nobody read is not a loss yet.
* **A grader is checked in both directions**: that it passes the right
  answers the model actually writes, and that it fails wrong ones. A reward
  too lenient looks like saturation; one too strict looks like a skill the
  model lacks.
* **When the answer format changes, every grader is checked again** on
  answers in the new format (the three-step format arrived on 10-20; the
  judge was not checked on it until 10-23).
* **A run that "finished" is opened** before its numbers are used: an exam
  that died writes no answers, and a table built without them still prints.
* What a check overturns goes into `docs/legal-it-experiments.md` beside the
  conclusion it replaces.

## Base Models

Only fully permissive licenses:
- **Qwen 3** — Apache 2.0 (primary choice, best multilingual)
- **Mistral** — Apache 2.0 (European company)
- **DeepSeek** — MIT
- **GPT-OSS** — Apache 2.0
- **Falcon 3** — Apache 2.0
- **Gemma 4** — Apache 2.0 since its release (2026-03), Base models included
  (E2B, E4B, 12B, 26B-A4B MoE, 31B). Earlier Gemma (2, 3) stays out: the
  Gemma Terms of Use carry use restrictions. Gemma 4 is a multimodal
  wrapper (`Gemma4ForConditionalGeneration`, text under `text_config`) and
  needs transformers 5.5+ (the 12B 5.10+): check the venv can load it
  before planning a run on it, and never upgrade the shared venv for it.

**One exception, decided by the PI on 2026-11-04: Qwen3.8-Flash-Next**
(`Qwen/Qwen3.8-Flash-Next`, 180B, 10 of 512 experts active, native hashed
3-gram tables), under the **Qwen Community License 1.0**, not Apache 2.0.
The licence allows use, fine-tuning, derivative works, distribution and sale,
and does not forbid training other models on it or on its outputs. Its
conditions: the copyright and licence notice travels with every copy and
derivative; a licensee that runs a *Model as a Service* (third parties
reach the model through an API or a hosted endpoint) or an *AI Work
Assistant* business (coding or office productivity) needs a separate licence
from Qwen for commercial use, derivatives included; internal use is exempt
as long as nothing of the model, its outputs or its capabilities reaches a
third party. So: allowed for research, as a teacher, and as the base of
models sold on-premises; **before any of its derivatives, or a student it
taught, is offered through an API, the separate licence is obtained first.**
Every model card of a derivative says so. Every other Qwen model used so far
(Qwen3, 3.5, 3.6, 3.8-27B) is Apache 2.0; check each new one, the same family
does not mean the same licence.

Llama (Meta) is excluded from the default catalog due to "Built with Llama" branding requirement.
