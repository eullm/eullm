# Verticalizing an LLM on a fixed EuroHPC budget — report outline

> Status: outline · Started 2026-09-08 · Target: Zenodo (DOI → ORCID),
> plus a shorter engineering post on the ZeRO-3/MoE result.

This is the skeleton of the write-up for the `eullm/legal-it-4b` run, and —
more urgently — the list of measurements that **only exist while the run is
happening**. Throughput, per-phase node-hours, loss curves and memory peaks
cannot be reconstructed afterwards from a finished checkpoint. Capture them
as they appear; the prose can wait.

## What the contribution is, and what it is not

"We fine-tuned a model on Italian legal text" is not a contribution; there
are many such papers. What is under-documented, and what this report is
actually about, is the part usually left out:

1. **The budget determines the architecture.** A 1,250 node-hour allocation
   with a 24 h walltime cap and no network on compute nodes is why the
   teacher is a MoE rather than the dense model the plan started with. Papers
   report the configuration they ended with; the reasoning that forced it is
   the reusable part.
2. **A MoE teacher under ZeRO-3 has a memory profile nobody writes down.**
   Reproducible failure, exact numbers, verified fix (§4).
3. **GDPR-aware corpus construction from a public case-law archive**, stated
   honestly as pseudonymisation rather than anonymisation, including a
   redaction bug that reported success while leaking (§3).
4. **The failures, with numbers.** Model ids that did not exist, a launcher
   pointing at a stale checkpoint path, a pre-flight that verified the wrong
   tokenizer, an OOM at the first forward. Each cost measurable time, and
   each is the kind of thing the next team hits.

## Structure

1. **Introduction** — sovereign / EU-compliant LLMs, why verticalization
   rather than a general model, why Italian case law as the first domain.
2. **The corpus** — italgiure SentenzeWeb, size, split, chunking. The
   pseudonymisation pipeline: what is redacted, what is deliberately kept
   (public officials, R.G. numbers, company names) and the legal reasoning
   for the distinction. Why the output remains personal data under GDPR
   Art. 4(5).
3. **A redaction bug that reported success** — the `\b`-anchored codice
   fiscale pattern, the four OCR shapes it silently missed, 59 codes reaching
   the training text after a "clean" run over 5.7 M redactions, and the
   gate that now blocks a launch on a dirty corpus. Generalisable lesson: a
   clean report from a redaction tool is evidence about its patterns, not
   about the corpus.
4. **Teacher and student selection under a compute ceiling** — the model
   survey (what Qwen actually publishes as Base, why Qwen3.5/3.8, Mistral
   and Gemma were excluded), and the MoE decision: 3.3 B active of 30.5 B
   total on a forward-only path.
5. **The ZeRO-3 memory result** — the core engineering finding. ZeRO-3 keeps
   parameters partitioned at rest but materialises a whole layer on every
   GPU; a Qwen3-30B-A3B layer is ~625 M parameters against ~25 M for a dense
   layer of the same hidden size, so dense-tuned limits fill 63.4 GiB of an
   A100 64 GB twenty seconds into the first forward. With the derivation and
   the corrected configuration.
6. **Operating inside a batch allocation** — 24 h walltime chaining,
   resume-from-checkpoint, no network on compute nodes (prefetch discipline),
   and observability: why the job now reports on itself, and what the four
   heartbeat numbers distinguish.
7. **Results** — the table below.
8. **Reproducibility** — configs, scripts and commit hashes; what a reader
   needs to repeat this on their own allocation.
9. **Limitations** — single domain, single language pair, one student size,
   no human evaluation of legal correctness, corpus not publishable.

## Measurements to capture DURING the run

Fill these in as they occur. An empty cell after the run is a number lost.

### Phase 0 — corpus

| Quantity | Value |
|---|---|
| Source slices (years, sections) | Cassazione snciv + snpen 2021-2026 + codici + Costituzione |
| Chunks, train / val | 1,127,316 / 11,387 (99/1, seed 42) |
| Tokens (approx) | ~700 M |
| On-disk size, train / val | 2900 MiB / 29 MiB |
| Pseudonymisation counts by category | (from `metadata.anonymization`, aggregated) |
| Codici fiscali found by the round-6 sweep | 59 (57 train / 2 val) |

### Phase 1 — continued pre-training

| Quantity | Value |
|---|---|
| Trainable params / total / percent | 106,954,752 / 30,639,077,376 / 0.3491 % |
| Time to first training step (cold cache) | |
| Time to first training step (warm cache) | |
| Seconds per optimizer step | |
| Peak VRAM per GPU (steady state) | |
| Peak host RSS per rank | |
| Steps for one epoch | |
| Wall-clock and node-hours for one epoch | |
| Loss at step 0 / 1k / 10k / final | |
| Val perplexity at each eval | |
| Number of chained jobs actually needed | |

### Phase 2 — distillation

| Quantity | Value |
|---|---|
| KL term at start / after 1k / final | |
| Effective alpha schedule realised | |
| Seconds per step | |
| Peak VRAM (teacher + student) | |
| Node-hours | |

### Phase 3 — quantization and export

| Quantity | Value |
|---|---|
| BF16 student size | |
| Q4_K_M GGUF size | |
| Perplexity before / after quantization | |
| Tokens/s on the target consumer GPU | |

### Budget

| Quantity | Value |
|---|---|
| Node-hours allocated | 1,250 |
| Node-hours spent, by phase | |
| Node-hours lost to failed runs | 0.67 (job 56760964, ZeRO-3 OOM) |
| Queue wait time, total | |

## Predictions, scored

The pilot's predictions were written down before it ran, in
[`v11-pilot-preregistration.md`](v11-pilot-preregistration.md). The results
section scores each one held / falsified / untested with the measured value
beside its threshold.

This is not decoration. Three predictions were already wrong in the first two
days, and each registered as a lesson only because a number had been stated
beforehand — the 60-hour Phase 1 estimate against 81 measured, the eval blamed
for a throughput loss it was not causing, and two ZeRO-3 levers that raised
the memory peak they were meant to lower. "We expected X and measured Y" is a
result; "we measured Y" is a data point.

### Scored so far

| | predicted | measured | |
|---|---|---|---|
| **P6** KL | < 0.01 nats | **0.0061** | held |
| **P6** top-1 agreement | > 99 % | **95.65 %** | **falsified** |

*(2026-09-11, 20 val documents / 10,186 positions, Qwen3-30B-A3B-Base at
int8 against bf16, no adapter. 9 minutes on one node.)*

**The two halves disagree, and the disagreement is the finding.** Top-5
agreement is 99.95 % and the KL is 0.006 nats, so where the argmax flips, the
first two candidates were near-tied: the change is in which of two almost
equal probabilities wins, not in the distribution. Distillation optimises a
divergence against the teacher's probabilities, not agreement on its argmax,
and 0.006 nats against a student loss around 1.27 is about 0.5 % of the
signal.

So the question P6 was asked to answer — *did quantizing the teacher to make
it fit distort the target?* — is answered **no**, and v1.0's Phase-1 adapter
needs no asterisk.

But the prediction was posed badly, and that is recorded rather than quietly
repaired. Top-1 agreement was chosen because it is the half a reader can
interpret without information theory; it measures the stability of an argmax,
not the fidelity of a distribution, and on formulaic Italian legal text — full
of positions with two near-equiprobable continuations — the argmax is unstable
even between two copies of the same model. The threshold that should have been
pre-registered is KL alone, with top-1 reported as context. Per the
pre-registration's own scoring rule, a badly posed prediction is itself a
result.

## Incidents log

One line each, with the cost. This is the section that makes the report
worth more than a tidy methods description.

| Date | Incident | Cost | Fix |
|---|---|---|---|
| 2026-09-08 | Configured teacher/student ids (`Qwen3-32B-Base`, `Qwen3-7B-Base`) do not exist on the Hub | prefetch aborted, replanning | model survey, real ids |
| 2026-09-08 | `RE_CF` anchored with `\b` missed 59 codici fiscali while reporting a clean run | corpus rewrite before launch | shape-only match; sweep gates the launch |
| 2026-09-08 | Phase 2/3 launchers pointed at pre-rename checkpoint paths | would have failed a phase each | repointed |
| 2026-09-08 | Pre-flight verified the smoke model's tokenizer, not the job's | false green | `--tokenizer-model` |
| 2026-09-08 | ZeRO-3 dense-tuned limits OOM'd at first forward on the MoE teacher | 0.67 node-hours | `ds_zero3_moe.json` |
| 2026-09-08 | 20+ minutes of silent job, diagnosed by hand from `/proc` | ~30 min of operator time | job heartbeat |

## Administrative

- **The EuroHPC Final Report is obligatory**, not optional: the PI submits it
  within three months of the allocation completing, on the EuroHPC JU
  template, to EuroHPC Peer-Review, and failure to submit can disqualify
  future proposals from any member of the research group. This allocation
  ends 02/11/2026, so it is due by **02/02/2027**. This outline is its
  skeleton as well as Zenodo's — write once, submit twice. Allocation
  strategy and the backlog that fills it:
  [`../leonardo-allocation-plan.md`](../leonardo-allocation-plan.md).
- Still to confirm with CINECA or the EuroHPC portal, not extractable from
  the published call PDFs: the exact Final Report template, and the required
  acknowledgement wording.
- Acknowledgement of the allocation (EHPC-AIF-2026PG01-1147) is required in
  any publication.
- Corpus stays private: pseudonymised, and `sentence_id` / `source_id`
  re-identify each ruling in a public archive. The report describes the
  pipeline; it does not ship the data.
