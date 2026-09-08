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

## Monitoring cheat-sheet

```bash
squeue --me                      # queue state of your chain
saldo -b                         # node-hour consumption vs monthly quota
tail -f "$EULLM_RUN_DIR"/logs/*.out
scancel <jobid>                  # kill one job of a chain
scancel --me                     # kill everything (chain deps die with it)
```

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
