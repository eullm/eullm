# ADR-001 — Freeze legal-it v1.0, then stop co-hosting teacher and student

> Status: **accepted, 2026-09-08** (revised the same day, see *Revision*) ·
> Affects Phase 1 and Phase 2 of the verticalization pipeline from v1.1 on.

## Decision, in one line

Let the running job produce `legal-it v1.0` untouched as a reproducible
baseline, and design the *next* pipeline so the teacher and the student never
occupy VRAM at the same time — measuring two ways of achieving that rather
than picking one in advance.

## Revision — why this document no longer prescribes offline caching

The first version of this ADR chose offline top-K caching outright. That was
premature: it committed to an architecture before measuring, which is the
mistake that produced the ZeRO-3 configuration in the first place.

The correction matters because of the specific model in play. Qwen3-30B-A3B
activates **3.3 B of 30.5 B parameters per token**, so a teacher forward is
cheap. Spending ~270 GB of disk, an I/O pipeline and a whole new class of
correctness bugs to avoid a cheap forward may well be a bad trade. The two
candidates are therefore both built and both measured.

## Part 1 — v1.0 is frozen

The job now running is the baseline. It is not modified for any reason short
of an OOM, a NaN or divergence in the loss, or a real error. Not for a
throughput improvement, not for a better memory layout, not for anything
learned after it started.

This is a deliberate stop. Between the first submission and this run, four
jobs failed and the configuration changed five times, each change justified
by the failure before it. That is the right way to find a working
configuration and the wrong way to obtain a *result*: a run adjusted while it
executes cannot be compared with anything, including itself.

### What must be captured, because it cannot be recovered afterwards

Commit and resolved YAML; versions of PyTorch, Transformers, DeepSpeed,
bitsandbytes, peft and the CUDA driver; checkpoints and the loss curve; peak
VRAM per GPU **as allocated, not reserved**; GPU utilisation and seconds per
optimizer step; node-hours, including the failed attempts; and the corpus
revision, including that it went through the round-6 PII sweep.

`forge/scripts/leonardo/capture_provenance.sh` collects these into the
checkpoint directory. Run it while the job is alive: a job log is not an
artifact.

> **Reserved is not allocated.** `nvidia-smi` reports what the caching
> allocator has taken from the driver, and with `expandable_segments` that
> pool grows toward the card's capacity and is never returned. A run showing
> 63,188 MiB of 64,946 MiB is not necessarily 1.7 GB from an OOM; it may hold
> gigabytes of free space inside its own pool. `torch.cuda.max_memory_allocated`
> is the number that means something, and the memprobe prints it at exit. A
> headroom criterion stated in reserved terms measures the allocator, not the
> model.

## Part 2 — the constraint, and two ways to remove it

Every problem of 2026-09-08 descends from one requirement: in online
distillation the teacher and the student must be resident together. That is
what made 61 GB of frozen teacher worth sharding, what made ZeRO-3 look
necessary, what made its all-gather explode on a 128-expert MoE, and what
finally forced the teacher to 8-bit so it would fit beside everything else.

Quantizing was a way around the constraint. Both designs below remove it, and
**neither is chosen before the pilot measures them.**

### A — offline top-K logit cache

Teacher BF16 across the four cards runs the corpus once and writes top-K
logits to disk. The teacher is then unloaded and the student trains against
the cache with the whole node to itself.

### B — online split node, 2+2

Teacher BF16 with tensor parallelism across GPUs 0-1 (about 30.5 GB per card
before activations); student training on GPUs 2-3. No persistent cache.

### Which wins is a question about the teacher's stability, not only speed

The decision criterion that is easiest to overlook: **a logit cache is
invalidated by any change to the teacher.** Re-run Phase 1 and 270 GB become
waste. So:

* **Offline pays when the teacher is fixed and the student varies** — several
  student sizes, LoRA ranks, or repeated runs after a bug, all reading one
  teacher pass.
* **Online pays while the teacher is still moving**, which is where the
  project is now.

This is a choice per phase, not a permanent one, and the pilot must report a
**break-even in number of experiments**, not a single wall-clock comparison.
Written out, with *P* the teacher pass, *S* a student run reading the cache,
and *S⁺* a student run computing the teacher inline:

```
offline(N) = P + N·S
online(N)  = N·S⁺        where  S⁺ ≈ S + P_inline
```

Naively that makes offline win at N ≥ 2, which is exactly why the comparison
has to be run rather than reasoned: the naive form hides three costs. The
teacher pass may be *slower* per token than inline forwards, because
`prompt_logprobs` is not the path vLLM is optimised for. Reading a 270 GB
cache adds to every *S*. And N is not the number of experiments you plan —
**it is the number you get between teacher changes**, because a Phase-1 re-run
voids the cache entirely. Report the break-even N*, and state the assumed
teacher-change interval next to it.

## Part 3 — teacher inference for design A

First backend to try: **vLLM** — Apache-2.0, so it clears the project's
copyleft rule. Not as a replacement for the EULLM engine, but because this
phase needs direct HF/BF16 loading, Qwen3-MoE tensor parallelism, batched
teacher-forcing, and bulk extraction of per-position top-K.

Behind an interface, to avoid a structural dependency:

```
TeacherLogitProvider          # returns RAW LOGITS, not log-probabilities
└── VllmTeacherProvider       # first implementation
└── EullmTeacherProvider      # later, if the engine gains logit export
```

Raw logits, not log-probabilities, or the temperature story in Part 4 breaks.

> **Verify this before building around it.** Teacher-forcing needs vLLM's
> `prompt_logprobs`, which returns values for every *prompt* position — not
> the `logprobs` of generation. It exists, but at large K over long sequences
> it is memory-hungry and has a history of rough edges. The first thing the
> pilot must establish is whether vLLM emits K=64 prompt-logprobs on
> Qwen3-MoE at an acceptable cost. If it does not, the provider is built
> around a different backend and no time is lost on an interface shaped by a
> tool that cannot do the job.

### Teacher topology benchmark

On the same subset, compare **TP=4** (one teacher across four GPUs) against
**2 × TP=2** (two independent replicas, half the corpus each), reporting
tokens/s, documents/s, GPU utilisation, VRAM, wall time, node-hours and
communication overhead. Do not assume 2×TP2 wins: MoE all-to-all traffic
does not scale the way dense tensor-parallel traffic does.

## Part 4 — cache format for design A

**Per sequence**, not per token: sample/document id, the target token
sequence, the top-K token ids (uint32), the top-K **raw logits in bf16**, the
`logZ_T` normalisers as fp32 for T = 1, 2, 4, the sequence length and shard
offsets, the tokenizer fingerprint, and the teacher checkpoint, adapter, Git
commit and config.

Three things are deliberately **not** stored, all for the same reason — a
field that can be derived and is stored anyway is a field that can contradict
its own source:

* **Position per token.** It is the index within the sequence record.
* **Normalised probabilities.** They fix a temperature at write time; raw
  logits plus `logZ_T` do not.
* **Residual mass.** It is `1 − Σ_topK exp(logit_i − logZ)`, computable
  exactly from what is already there.

**bf16 for the logits, not fp16**, and the reason is not dynamic range. The
teacher computes in bf16, so its logits already carry exactly that precision;
storing them in bf16 is lossless with respect to the source, at identical
size. fp16's two extra mantissa bits would hold no information.

`logZ_T` at three temperatures is what makes a truncated top-K a usable
distribution: the softmax at temperature T over the *full* vocabulary cannot
be recovered from the T=1 normaliser. Three fp32 per position is ~12 bytes,
about 8 GB over the corpus, against re-running the teacher to change a
hyperparameter.

Sharded, never one monolithic file.

### Storage, with the arithmetic done

At 700 M positions, ids as uint32, logits as bf16, plus 12 bytes of `logZ_T`:

| K | bytes/position | corpus |
|---:|---:|---:|
| 16 | 108 | ~76 GB |
| 32 | 204 | ~143 GB |
| 64 | 396 | **~277 GB** |
| 128 | 780 | **~546 GB** |

The `$WORK` quota is 1 TB and also holds the HF cache (~69 GB), the corpus and
every checkpoint. **K=128 is therefore a pilot measurement on a subset, not a
full-corpus option** unless compression changes the picture.

And compression is the lever this plan is missing. Top-K logits compress well:
store the top-1 absolutely and the remaining K−1 as int8 deltas, and logit
storage falls by 2–4×. That puts K=64 near 150 GB and brings K=128 back into
range. Measure it in the pilot rather than assuming bf16 throughout.

## Part 5 — K is measured, not chosen

Pilot on one subset at **K = 16, 32, 64, 128**, and choose on all of:

* probability mass captured by the truncation
* truncation error / KL against the full vocabulary
* validation perplexity
* the fixed Legal-IT evaluation set
* **general-capability regression** — a domain-distilled model forgets, and a
  K small enough to look good on legal text can be quietly destroying
  everything else
* storage
* end-to-end node-hours

Then pick one K for the whole corpus.

A truncation that discards meaningful mass makes the student fit a
distribution the teacher never had — the same class of error the quantization
gate exists to catch.

## Part 6 — cache correctness tests are mandatory

Two different questions live here and must not be answered by one test,
because they have different causes and different fixes:

1. **Is the cache a faithful record of what the teacher said?** A bug in
   writing, sharding, alignment or reading.
2. **How much does truncating to K lose?** A property of the method, present
   even in a perfect cache.

### 6.1 — the equality test (faithfulness)

On a fixed batch, compare the **live teacher's top-K and `logZ_T`** against
the **cached** values, and require equality within numerical tolerance. This
is the test that catches a cache which is internally consistent and wrong.

> **Set the tolerance from a control, not from taste.** A GPU forward is not
> bit-exact across batch compositions — different batch shapes change
> reduction order in the matmuls and all-reduces. Run the teacher **twice
> live** on the same batch first, measure the spread, and derive the cache
> tolerance from it. Skip this and the test either fails at random or is so
> loose it distinguishes nothing. Whatever tolerance is chosen, record the
> live-vs-live baseline next to it, or the number is unfalsifiable.

### 6.2 — the truncation measurement (method)

Separately, and only once 6.1 passes: measure the error K introduces against
the full vocabulary. Reported as retained probability mass and KL, per Part 5.
Failing to separate the two means a genuine cache bug can be waved through as
"expected truncation error".

### 6.3 — the rest, all automated

The `logits[t] → target[t+1]` alignment; padding; packing boundaries; BOS/EOS
handling; sample boundaries; and tokenizer compatibility.

The alignment test earns its place first among these because the failure is
silent: an off-by-one does not raise the loss in any obvious way, it poisons
270 GB of cache, and it is discovered after the node-hours are spent.

**The tokenizer check must not stop at vocabulary size.** Two tokenizers can
agree on 151,936 and differ in merges or special-token ids. Compare a
fingerprint over the vocabulary mapping and the special-token ids.

## Part 7 — resume

The teacher pass is fully resumable. Every shard is written atomically and
recorded in a manifest with: document range, teacher commit, adapter or
checkpoint, tokenizer hash, K, dtype, temperature metadata, and completion
status. A job killed at the walltime cap restarts from the first incomplete
shard.

## Part 8 — the student, and where sharding *is* the right tool

* 4 B LoRA → DDP
* 7 B LoRA → DDP if memory allows
* full fine-tune of 4 B or 7 B → **do not assume DDP suffices.** A 4 B full
  fine-tune needs roughly 56 GB for weights, fp32 master copy and Adam state
  before activations. FSDP or ZeRO-2/3 is the correct tool here.

Worth stating plainly, because today's failure invites the wrong conclusion:
the lesson is not that ZeRO is bad. It is that **ZeRO is for trainable
parameters.** On a frozen base it bought nothing and cost everything; on a
fully fine-tuned student it is exactly right.

## Part 9 — EULLM's place

EULLM remains the runtime for the model Forge produces. Do **not** convert the
teacher to GGUF merely to use EULLM in Phase 2A — the pilot uses the HF/BF16
checkpoint directly.

Batched teacher-forcing with per-token top-K/raw-logit export and multi-GPU
teacher inference is a plausible EULLM feature, to be decided *after*
benchmarking against vLLM, not before.

## Part 10 — Qwen3.8

Not usable for token-level KL toward a Qwen3 student: the tokenizers are
incompatible (248,320 against 151,936), and KL over vocabulary positions is
meaningless across them. Evaluate it separately in roles that operate on
text, where the difference does not matter: sequence-level KD teacher,
synthetic-data generator, reasoning/instruction teacher, evaluator or judge.

## Part 11 — success criteria

The new pipeline replaces v1.0 only on a measured improvement in at least one
of quality, node-hours, throughput, VRAM headroom, stability, or the ability
to train a larger student — **without substantial regression in the others**.

v1.0 is not discarded. It is the baseline the replacement is measured
against, and "better designed" is not a result.

## What this does not change

The corpus, the pseudonymisation pipeline and its gate, the model naming, the
Phase 3 GGUF path and the licence constraints all stand. This ADR is about
where the teacher's parameters live while the student learns, and nothing
else.
