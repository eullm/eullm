# Pre-registration — what we expect the v1.1 pilot to find

> Written **2026-09-09**, before any of it was measured, while Phase 1 of
> v1.0 is still running. Frozen on purpose: this file records predictions,
> not results. Results go in
> [`legal-it-4b-report-outline.md`](legal-it-4b-report-outline.md), and the
> comparison between the two is the point.
>
> Design being tested: [`../adr-001-offline-distillation.md`](../adr-001-offline-distillation.md)

## Why this file exists

A prediction written after the measurement is a rationalisation. The value
of stating one first is that it can be wrong in public, and being wrong in a
specific, numeric way is more informative than being right vaguely.

Three predictions were already wrong in the first two days of this project,
and each was only *visible* as an error because someone had said a number
out loud beforehand:

| predicted | measured |
|---|---|
| Phase 1 epoch in ~60 h | 81 h — the early throughput sample was too small |
| the eval every 500 steps is the throughput problem | ~1 min in 30, about 3% |
| halving `cutoff_len` and disabling `overlap_comm` cut the ZeRO-3 peak | peak went *up*, 61.25 → 61.84 GiB |

None of those would have registered as a lesson if no expectation had been
recorded. This file makes that deliberate rather than accidental.

Every prediction below carries a **falsification threshold** — the number at
which it counts as wrong — and a **confidence**, so that a wrong prediction
made with low confidence is not scored the same as a wrong prediction made
with high confidence.

## P1 — vLLM emits `prompt_logprobs` at K=64 on Qwen3-MoE

**Prediction:** it works, but is slower per token than generation-mode
logprobs, and **memory is the binding constraint rather than compute** — the
logprob buffer at K=64 over long sequences, not the model.

**Falsified if:** the call fails or returns fewer than K values per position;
or throughput is within 20% of generation mode (which would mean the path is
not doing the extra work we expect and something else is wrong).

**Confidence: low-medium.** This is the assumption the whole of design A
rests on, which is exactly why it is tested first. If it fails, the provider
interface is built around a different backend and nothing else in the plan
changes.

## P2 — teacher topology: 2 × TP=2 beats TP=4

**Prediction:** two independent replicas on two GPUs each beat one teacher
across four, on aggregate tokens/s, by **more than 15%**.

**Reasoning:** MoE all-to-all traffic grows with the number of ranks in a way
dense tensor-parallel traffic does not, and the teacher pass is embarrassingly
parallel across documents, so replication avoids the communication entirely.

**Falsified if:** TP=4 is equal or faster, or the gap is under 15%.

**Confidence: low.** The ADR explicitly says not to assume this, and the
prediction is recorded here precisely so that assuming it cannot happen
silently. A 30 GB model per card at TP=2 also leaves less room for batching,
which could cancel the communication saving.

## P3 — K=32 is enough; K=64 is insurance, K=128 is waste

**Prediction:** on Italian case law, **K=32 captures ≥99% of the probability
mass** at T=1, and the KL against the full vocabulary is below 0.01 nats.

**Reasoning:** legal Italian is a narrow, highly conventional register.
Next-token distributions in formulaic text are far more peaked than in general
prose, and the top-32 should be doing almost all the work.

**Falsified if:** retained mass at K=32 is below 99%, or the truncation KL
exceeds 0.01 nats.

**Confidence: medium.** This one is worth stating loudly because it runs
against the instinct to pick the largest K that fits. If it holds, the cache
is ~143 GB instead of ~277 GB and everything downstream gets cheaper.

## P4 — online 2+2 wins at N=1; break-even at N≈2-3

**Prediction:** for a single student run, **design B (online split) finishes
in fewer node-hours end to end** than design A (teacher pass + cached
student). The break-even is at **two to three student runs** per teacher
version.

**Reasoning:** the teacher is only 3.3 B active parameters, so its inline
forward is cheap; design A pays a full corpus pass up front plus cache I/O on
every student epoch. Design A only recovers that across repeats.

**Falsified if:** design A wins at N=1, or the break-even is above N=5.

**Confidence: low-medium.** This is the prediction most likely to be wrong,
because the cost of `prompt_logprobs` (P1) and of reading a 150-270 GB cache
are both unmeasured.

## P5 — the paired short runs are indistinguishable

**Prediction:** with the same seed, the same starting checkpoint and the same
teacher precision on both sides, the loss curves of the offline and online
paths **overlap within run-to-run noise**, and validation perplexity at step
2,000 differs by **less than 1%**.

**Reasoning:** the two designs optimise the same objective against the same
teacher. They should be the same computation arranged differently.

**Falsified if:** perplexity differs by more than 1%, or the curves separate
beyond the spread of two same-path runs with different seeds.

**Confidence: high** — and this is the prediction whose failure matters most.
A divergence here is not an interesting property of the pipelines; it is a
bug, most likely in the `logits[t] → target[t+1]` alignment. **Establish the
same-path noise baseline first**, or there is nothing to compare "within
noise" against.

## P6 — the int8 teacher of v1.0 was close to harmless

**Prediction:** against bf16, the int8 teacher shows **top-1 agreement above
99%** and **mean KL below 0.01 nats** over 200 validation samples.

**Reasoning:** `LLM.int8()` decomposes outlier features into bf16 precisely
because that is where naive 8-bit quantization of transformers breaks.

**Falsified if:** top-1 agreement below 99%, or mean KL above 0.01 nats.

**Confidence: medium.** Note the asymmetry in what the outcomes mean. If it
holds, the v1.0 teacher was fine and the move to bf16 buys tidiness rather
than quality. If it fails, v1.0's Phase-1 adapter was trained against a
meaningfully distorted model and the baseline needs an asterisk — which would
be an important finding about the whole quantize-to-fit approach, not just
about this run.

## What is deliberately not predicted

**The quality of a v1.0 student trained through the old Phase 2**, because we
have decided not to run it (ADR Part 11). Predicting an outcome nobody will
measure is not a prediction, and writing one here would quietly create the
impression that the comparison happened.

**Phase 2 wall-clock for the full corpus.** It depends on P1, P2 and P4, all
unmeasured. A number invented now would be the same kind of guess as the
60-hour Phase 1 estimate, and would be quoted later as though it had been
reasoned.

## Scoring

When the pilot reports, each prediction gets **held / falsified / untested**,
with the measured value beside the threshold, in the report's results section.
No prediction is edited here after the fact. If one turns out to have been
badly posed — unfalsifiable, or measuring the wrong thing — that is recorded
as a defect of the prediction, which is itself a result worth having.
