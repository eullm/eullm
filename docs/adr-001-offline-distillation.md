# ADR-001 — Freeze legal-it v1.0, then move distillation offline

> Status: **accepted, 2026-09-08** · Supersedes nothing · Affects Phase 1 and
> Phase 2 of the verticalization pipeline from v1.1 onward.

## Decision, in one line

Let the running job produce `legal-it v1.0` untouched as a reproducible
baseline, and design the *next* pipeline so the teacher and the student never
occupy VRAM at the same time — which removes the constraint that forced every
architectural compromise made on 2026-09-08.

## Part 1 — v1.0 is frozen

The job now running is the baseline. It is not modified for any reason short
of an OOM, a NaN or divergence in the loss, or a real error. Not for a
throughput improvement, not for a better memory layout, not for anything
learned after it started.

This is a deliberate stop. Between the first submission and this run, four
jobs failed and the configuration was changed five times, each change
justified by the failure before it. That is the right way to find a working
configuration and the wrong way to obtain a *result*: a run that is adjusted
while it executes cannot be compared with anything, including itself.

### What must be captured, because it cannot be recovered afterwards

* Git commit and the complete resolved YAML
* Versions: PyTorch, Transformers, DeepSpeed, bitsandbytes, peft, CUDA driver
* Checkpoints, and the loss curve
* Peak VRAM per GPU — **allocated, not reserved** (see the note below)
* GPU utilisation and throughput (seconds per optimizer step)
* Node-hours consumed, and the failed attempts that preceded it
* The corpus revision, including that it went through the round-6 PII sweep

`forge/scripts/leonardo/capture_provenance.sh` collects these into the
checkpoint directory. Run it while the job is alive: the pre-flight block and
the run banner are in the job log, and a log is not an artifact.

> **Reserved is not allocated.** `nvidia-smi` reports what the caching
> allocator has reserved from the driver, and with `expandable_segments` that
> pool grows toward the card's capacity and is not returned. A run showing
> 63,188 MiB of 64,946 MiB is not necessarily 1.7 GB from an OOM; it may have
> gigabytes of free space inside its own pool. The number that means anything
> is `torch.cuda.max_memory_allocated`, which the memprobe prints at exit.
> Any headroom criterion stated in reserved terms measures the allocator, not
> the model.

## Part 2 — the constraint we should have attacked

Every problem of 2026-09-08 descends from one requirement: in online
distillation the teacher and the student must be resident together. That is
what makes 61 GB of frozen teacher a problem worth sharding, what made ZeRO-3
look necessary, what made ZeRO-3's all-gather explode on a 128-expert MoE,
and what ultimately forced the teacher to 8-bit so it would fit beside
everything else.

Quantizing the teacher was a way around the constraint. Removing the
constraint is better, and it is not difficult.

### Offline top-K logit distillation

**Phase 2A — teacher pass.** Load the teacher in BF16, spread across the four
A100s, and run the corpus once. For every position write: the token id, the
top-K token ids, their logits or probabilities, any normalisation metadata,
and the sample/document id for traceability. No student is loaded. No
gradients exist.

**Phase 2B — student training.** Unload the teacher entirely. Train the
student against the cached distributions with the same
`α · KL + (1−α) · CE` objective. The teacher is not in VRAM at all.

What this buys, beyond fitting:

* **The teacher returns to BF16.** Not as a concession — there is simply
  nothing left to share memory with. The quantization gates added today stop
  being load-bearing for Phase 2 and become a v1.0 property.
* **The teacher pass is reusable.** It is computed once and consumed by every
  subsequent student: a 4 B and a 7 B, a different LoRA rank, a repeated run
  after a bug. Today, changing the student means re-running the teacher.
* **Both halves suit a 24 h batch queue.** Each is separately resumable — the
  teacher pass by corpus offset, the student by checkpoint — where an online
  run must checkpoint two models in lockstep.
* **Tensor and expert parallelism become optional.** With no student
  competing for memory, a BF16 teacher fits comfortably across four cards
  under a plain `device_map`. Native Qwen3-MoE expert parallelism is worth
  measuring for the teacher pass, but the offline split means it is a
  throughput question, not a feasibility one.

### The cost, stated up front

Storage. At K=64, each position needs 64 ids (int32) and 64 logits (fp16),
or 384 bytes; across roughly 700 M tokens that is **~268 GB**. K=32 halves it
to ~134 GB. The `$WORK` quota is 1 TB, so it fits — but this is the real price
of the design and it belongs in the comparison of Part 4, not in a footnote.

Worth measuring in the pilot: fp16 logits may be more precision than the KL
objective can use, and storing normalised log-probabilities as int16 would
halve the cache again.

### K is measured, not chosen

Pilot on one subset at **K=32** and **K=64**, and report:

* probability mass retained by the top-K truncation
* KL against the full distribution on that subset
* resulting student quality
* cache size on disk
* teacher-pass and student-training throughput

Pick K from those numbers. A truncation that discards meaningful mass makes
the student fit a distribution the teacher never had, which is the same class
of error the quantization gate exists to catch.

## Part 3 — tokenizer compatibility bounds the teacher choice

Logit-level KD requires teacher and student to share a vocabulary; the KL is
computed over vocabulary positions and two tokenizers make that meaningless.
Qwen3-30B-A3B → Qwen3-4B/7B is therefore the natural path, and this is why
the Qwen3.5/3.8 line was excluded earlier: vocab 248,320 against Qwen3's
151,936.

**Qwen3.8-27B must not replace the KL teacher in v1.x** until a genuine
cross-tokenizer strategy exists. It can be evaluated separately in roles that
operate on *text* rather than on logits, where the tokenizer difference is
irrelevant:

* sequence-level KD teacher (SeqKD)
* synthetic data generator
* reasoning / instruction teacher
* evaluator or judge

## Part 4 — the new pipeline must earn its place

Before a full run: a short end-to-end test showing no OOM, **at least 2–3 GB
of headroom per GPU measured as allocated**, scaling across the four cards,
teacher-pass throughput, logit cache size, student throughput, and quality
equivalent to online distillation.

And v1.0 is not discarded. It is the experimental baseline the replacement is
measured against, and the replacement must win quantitatively on at least one
of: quality, throughput, node-hours, VRAM, stability, or the ability to train
a larger student. "Better designed" is not a result.

## What this does not change

The corpus, the pseudonymisation pipeline and its gate, the model naming, the
Phase 3 GGUF path, and the licence constraints all stand. This ADR is about
where the teacher's parameters live while the student learns, and nothing
else.
