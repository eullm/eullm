# Leonardo (CINECA) Runbook — legal-it-4b on EuroHPC

> Allocation: **EHPC-AIF-2026PG01-1147** — 40,000 local core hours =
> **1,250 node hours** on Leonardo Booster, 02/09/2026 → 02/11/2026.
> Strategy background: [`legal-it-4b-strategy.md`](legal-it-4b-strategy.md).
> All the scripts referenced here live in `forge/scripts/leonardo/`.

## The machine, in the terms that matter here

| Fact | Consequence |
|------|-------------|
| Booster node = 4x A100 **64 GB** (32 cores, ~490 GB RAM) | No single GPU fits the 78-96 GB single-GPU budgets → Phase 1 uses DeepSpeed ZeRO-3 across 4 GPUs, Phase 2 shards the frozen teacher via accelerate. Dedicated Leonardo YAMLs live in `forge/training/configs/leonardo/`. |
| Accounting: 1 GPU-hour = ¼ node hour | Smoke tests request 1 GPU, not a node. Phase 3 runs on the serial partition and costs no GPU budget. |
| Max walltime 24 h (`boost_usr_prod`) | Multi-day phases run as chains of jobs (`submit_chain.sh`); every launcher auto-resumes from the latest checkpoint. |
| Compute nodes have **no internet** | Models are prefetched to `$WORK` from a login node; jobs run with `HF_HUB_OFFLINE=1` (env.sh forces it inside jobs). llama.cpp is cloned during setup. |
| Budget linearization: ~625 node hours/month, unused budget reported to EuroHPC | Start Phase 1 immediately; keep the queue fed. Check consumption with `saldo -b`. |
| `$HOME` 50 GB, `$WORK` 1 TB (project), `$SCRATCH` purged | Everything (venv, HF cache, dataset, checkpoints) lives on `$WORK`. env.sh encodes the layout. |

## Node-hour budget

| Item | Node hours |
|------|-----------:|
| Smoke + wiring validation | ~1 |
| Phase 1 — continued PT 30B-A3B (ZeRO-3, ≤ 2 x 24 h chain) | ~25-50 |
| Phase 2 — distillation (7 x 24 h chain) | ~120-170 |
| Phase 3 — GGUF (serial partition) | 0 |
| Retries / headroom | ~50 |
| **Total legal-it-4b** | **~250** |

Out of 1,250 available — leaves ample budget for a second epoch,
ablations, or the medical-de / finance-fr runs if their corpora are
ready in time.

## 0. One-time setup (login node)

```bash
# In ~/.bashrc on Leonardo (account name from `saldo -b`):
export EULLM_ACCOUNT=<project_account>

git clone https://github.com/eullm/eullm.git "$WORK/eullm"
cd "$WORK/eullm"

# Downloads ~80 GB of models into $WORK/hf_home — run it in tmux.
# For the dataset either export HF_TOKEN + HF_DATASET (private HF repo),
# or rsync train.jsonl/val.jsonl to $WORK/datasets/legal_it/ first.
bash forge/scripts/leonardo/setup.sh
```

The final step of setup.sh is a login-node pre-flight
(`check_training_env.py --cpu-only --multi-gpu --offline`): it proves
the venv, DeepSpeed, the dataset, and the offline HF cache are all in
place *before* any node-hours are spent.

## 1. Smoke test (1 GPU, debug QOS, ~15 min, ~0.1 node hours)

```bash
source "$WORK/eullm/forge/scripts/leonardo/env.sh"
cd "$EULLM_RUN_DIR"
bash "$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \
    "$EULLM_REPO/forge/scripts/leonardo/sbatch_smoke.slurm"
tail -f logs/eullm-smoke-*.out
```

Green means: offline model load, dataset registration, training steps,
checkpoint written. Optionally re-submit once to watch it resume from
`checkpoint-50` — that same mechanism is what the 24 h chains rely on.

## 2. Phase 1 — continued pre-training (4 GPUs, chain of 2)

```bash
bash "$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \
    "$EULLM_REPO/forge/scripts/leonardo/sbatch_phase1.slurm" 2
```

Output: LoRA adapter at
`$EULLM_RUN_DIR/checkpoints/qwen3_30b_a3b_legal_it_continued_pt/`.
If the epoch finishes inside job 1, job 2 wakes up, finds the final
checkpoint, re-runs the last partial step and exits quickly — a few
wasted node-minutes, not hours.

## 3. Phase 2 — distillation (4 GPUs, chain of 7)

```bash
bash "$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \
    "$EULLM_REPO/forge/scripts/leonardo/sbatch_phase2.slurm" 7
```

The sbatch script refuses to start if the Phase-1 adapter is missing.
Output: LoRA checkpoints plus — from the final save — a **merged**
full-weights student at
`.../checkpoints/qwen3_4b_legal_it_distilled/merged/`, which is what
Phase 3 consumes. Watch the `kl=` term in the logs: it should fall
steadily for the first few thousand steps.

## 4. Phase 3 — GGUF Q4_K_M (serial partition, no GPU budget)

```bash
bash "$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \
    "$EULLM_REPO/forge/scripts/leonardo/sbatch_quantize.slurm"
```

Output: `$EULLM_RUN_DIR/gguf/legal-it-4b/legal-it-4b-q4_k_m.gguf`
(~4.5 GB). Pull it home and smoke-test in the EULLM Engine:

```bash
rsync -av --progress \
    <user>@login.leonardo.cineca.it:"$WORK/eullm_runs/legal_it/gguf/legal-it-4b/" \
    ./gguf/
```

Move artifacts off Leonardo as they are produced — the allocation (and
the storage) ends on 02/11/2026, and CINECA recommends not leaving data
transfers to the last days.

## Monitoring

### Is anything running, and how far along?

```bash
squeue --me -o '%.10i %.18j %.9T %.10M %.10L %R'
```

`%M` is elapsed, `%L` is walltime left, `%R` is the node — or, for a pending
job, the reason. `(Dependency)` is a chain link waiting its turn, `(Priority)`
is the cluster being full, `(BeginTime)` is a job deliberately scheduled for
later.

For the training itself, read the log rather than the queue:

```bash
cd "$EULLM_RUN_DIR"
LOG="$(ls -t logs/*.out | head -1)"
tail -50000 "$LOG" | grep -v MatMul8bitLt | tail -10
```

**The `tail | grep -v` is not decoration.** With an 8-bit base, bitsandbytes
prints `MatMul8bitLt: inputs will be cast…` once per quantized matmul — 48
layers × 4 projections × every step. The Phase-1 log reached **2.1 GB in 24
hours**, and a plain `grep` over it sits there long enough to look hung. Two
sessions were lost to pressing Ctrl-C on a `grep` that was working fine.
`tail` reads only the end of the file and returns instantly at any size.

The line worth finding looks like:

```
{'loss': '1.269', 'grad_norm': '0.5634', 'learning_rate': '3.422e-06', 'epoch': '0.6164'}
```

`epoch` is the honest progress indicator. **After a chained restart it is also
the proof that the resume worked**: a job that silently began from scratch
shows `epoch` near zero, and there is no other signal that 24 hours just
evaporated.

### Follow a job to its end

```bash
bash "$EULLM_REPO/forge/scripts/leonardo/watch_job.sh" <jobid>
```

Waits for the log to appear if the job is still queued, filters while it runs,
stops when the job leaves the queue, and prints the `sacct` verdict plus the
lines that caused it. `tail -f` does none of that: when a job dies the tail
just sits there, which is how one failure went unnoticed for twenty minutes.

### Budget

```bash
bash "$EULLM_REPO/forge/scripts/leonardo/budget.sh"
bash "$EULLM_REPO/forge/scripts/leonardo/budget.sh" --reserve 200
```

**Do not decide anything on `saldo -b` alone.** It counts finished *and
billed* jobs, and the lag is long: on 2026-09-11 it reported 907 core-hours
against 1,675 actually spent — a job that had ended ten hours earlier was
neither still running nor yet billed. `budget.sh` sums `sacct` over the whole
allocation instead and shows `saldo` underneath as a cross-check.

### Where the calendar went

```bash
python3 "$EULLM_REPO/forge/scripts/leonardo/queue_stats.py" 2026-09-02 \
    --json "$WORK/eullm_runs/qstats/queue_stats_$(date +%Y%m%d_%H%M).json"
```

Node-hours are not this allocation's binding constraint; **calendar is** (see
[`leonardo-allocation-plan.md`](leonardo-allocation-plan.md)). One node kept
busy continuously for the whole two months consumes essentially the entire
grant, so every idle hour expires unrecoverably.

`queue_stats.py` splits the idle time by cause, because the two halves mean
opposite things in the Final Report:

| | |
|---|---|
| **idle, cluster full** | a job was queued and would not start. About the machine. |
| **idle, queue empty** | nothing was submitted, because nothing was ready. Ours. |

It also separates **queue wait** from **dependency wait**, which is where the
obvious arithmetic goes wrong: `Start - Submit` on a chained job counts the
time it spent waiting for its own predecessor. On the Phase-1 chain that gave
77 hours against 7 real ones. The clock here starts at the predecessor's end.

Every wait comes out dated, from and to, so the report can quote an interval
instead of a total — "the chain stalled from 2026-09-10 21:07 to 2026-09-11
04:13" is evidence; "7h06m of queue wait" is not.

`sacct` has a retention window, so these intervals stop being recoverable a
few weeks after the jobs run. That is what `--json` is for.

### Keeping the record without a scheduler

Leonardo permits no user `crontab`, and `scrontab` is disabled cluster-wide.
The substitute is a job that re-submits itself:

```bash
sbatch "$EULLM_REPO/forge/scripts/leonardo/sbatch_queue_stats.slurm"
squeue --me -n eullm-queue-stats    # exactly one PENDING row, always
```

Three seconds of one core on the serial partition, once a day, until the
allocation ends. Its one failure mode is a re-submission that does not happen:
nothing announces it, the job simply stops existing. If that `squeue` line
comes back empty, submit it again.

Worth doing **in addition**, at the end of any real sbatch script:

```bash
python3 "$EULLM_REPO/forge/scripts/leonardo/queue_stats.py" 2026-09-02 \
    --json "$EULLM_RUN_DIR/qstats/queue_stats_$(date +%Y%m%d_%H%M).json" || true
```

The daily job gives regular cadence; this catches the handover between one
chained job and the next, which is precisely when the gaps open. The `|| true`
matters — an instrument must never fail the job it is measuring.

### Stopping things

```bash
scancel <jobid>                  # one job of a chain
scancel --me                     # everything (chain deps die with it)
```

## Running the pilot beside a frozen run

Queued jobs read their scripts from `$EULLM_REPO` **when they start**, not
when they were submitted. So while a chain is in flight, a `git pull` silently
changes what the not-yet-started links will execute.

That does not mean waiting. Clone a second checkout:

```bash
git clone https://github.com/eullm/eullm.git "$WORK/eullm-v11"
python -m venv "$WORK/v11_venv"
```

`$WORK/eullm` stays pinned for the chain; `$WORK/eullm-v11` tracks `main` and
the new work runs from there with `EULLM_REPO` and `EULLM_RUN_DIR` pointed at
it. 300 MB, and the freeze stops being a blocker.

**The separate venv is the part that actually matters.** Installing vLLM into
the venv a running chain uses would change its PyTorch, and the next link
would pick that up on restart — the most efficient way to lose four days of
compute. Read-only use of the training venv is fine; installing into it is not.

Jobs cannot collide on hardware: `boost_usr_prod` has `OverSubscribe=NO`, so
SLURM never places two jobs on the same node. The QoS allows 256 nodes and
1,000 submitted jobs per user, so parallelism is limited by budget and by
having work worth running, not by the scheduler.

## Troubleshooting

- **Job dies immediately, log mentions HF cache / offline** — a model
  is missing from `$WORK/hf_home`: re-run `prefetch_models.py` on a
  login node.
- **`sbatch: error: Invalid account`** — `EULLM_ACCOUNT` unset or wrong;
  the exact name is in `saldo -b`.
- **Phase-1 OOM per GPU** — drop `cutoff_len` 2048 → 1024 in the
  Leonardo YAML; ZeRO-3 already shards weights/optimizer, activations
  are the only real lever.
- **Phase-2 OOM on cuda:0** — lower `teacher_gib_on_student_gpu` to 4
  (pushes more teacher onto GPUs 1-3), then `cutoff_len` as above.
- **A chained job starts and exits green in minutes** — normal: the
  previous job already finished the phase; the chain drains cheaply.
- **Low priority / long queue waits** — monthly quota exceeded (budget
  linearization). Check `saldo -b`; jobs still run, just deprioritized.
- **`nano`, `watch`, `htop` die with a segmentation fault** — not a broken
  binary. The `python/3.11.7` module loads its own `ncurses/6.5` and puts it
  first on `LD_LIBRARY_PATH`; the system binaries were linked against the
  OS copy and crash on the mismatch. Run them with that variable stripped:
  ```bash
  alias nano='env -u LD_LIBRARY_PATH nano'
  alias watch='env -u LD_LIBRARY_PATH watch'
  ```
  This cost half an hour twice before the two crashes were recognised as the
  same one. To edit a file without an editor at all, `cat > file <<'EOF'`
  with the marker quoted writes it verbatim, `$VARIABLES` included.
- **A `grep` over a training log appears to hang** — it is not hung, the file
  is gigabytes of repeated bitsandbytes warnings. Use
  `tail -50000 "$LOG" | grep -v MatMul8bitLt` instead; see Monitoring above.
- **A chain link resumed but you cannot tell from what** — check `epoch` in
  the last training line, not the loss. A fresh start shows it near zero while
  the loss can look plausible either way.
- **`crontab` refused, `scrontab: fatal: scrontab is disabled`** — expected,
  neither is available. Use the self-resubmitting job pattern in
  `sbatch_queue_stats.slurm`.
