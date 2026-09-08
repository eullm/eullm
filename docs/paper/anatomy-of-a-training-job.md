# Anatomy of a training job

> Draft for the eullm site · started 2026-09-08 · companion to
> `legal-it-4b-report-outline.md`, which is the longer technical report.
>
> Written in English because the engineering result travels further that
> way; an Italian version for the legal-domain audience is worth doing
> separately rather than as a translation, since the interesting half is
> different for each.

Papers describe pipelines. Nobody describes what a single job on a
supercomputer actually does between `sbatch` and the first loss value, which
is a shame, because that is where the failures live. This is that account,
written from one real run of the `eullm/legal-it-4b` pipeline on the CINECA
Leonardo Booster, with the numbers we measured rather than the ones we
expected.

## The pipeline, in one paragraph

A generalist model is verticalized in five stages: build a domain corpus
(Phase 0), continue pre-training a large teacher on it (Phase 1), distil the
teacher into a small student (Phase 2), quantize the student to GGUF
(Phase 3), and optionally fine-tune an identity (Phase 4). Only Phases 1 and
2 need the supercomputer. The output runs on a laptop.

That asymmetry is the point, and it is worth stating numerically before
anything else. Producing the model peaks at roughly 250 GiB of system memory
and 65 GiB of aggregate GPU memory across four A100s, plus NVLink and a
parallel filesystem underneath. Running the result needs a 2.5 GB file and
8 GB of RAM. About a hundred to one. The barrier to sovereign AI is
concentrated in one place — you need the big machine once, to build — and
that is precisely what makes it a surmountable barrier rather than a
permanent moat.

## What each phase does

**Phase 0 — Corpus.** Fetch the case-law archive, pseudonymise it,
deduplicate, chunk. This is also where you decide what the model will *not*
learn: party names, fiscal codes, addresses. Costs no node-hours; runs on a
login node. Ours: 1,127,316 training chunks and 11,387 validation chunks,
about 700 M tokens.

**Phase 1 — Continued pre-training of the teacher.** Take a model that knows
general Italian and let it read 700 M tokens of legal Italian. The loss is
ordinary next-token cross-entropy — no tricks. The base weights are frozen;
a LoRA adapter of 106,954,752 trainable parameters (0.35 % of 30.6 B) absorbs
the difference between "Italian" and "the Italian of the Court of
Cassation". Output: a ~2 GB adapter.

**Phase 2 — Distillation.** The frozen teacher reads the corpus and emits,
for every token, its full probability distribution over the vocabulary. The
student learns to reproduce that distribution, not merely the correct answer:
`α · KL(student ‖ teacher) + (1−α) · CE(student, y)`, with α annealed from
0.9 to 0.5. This is the difference between learning the answer and learning
the reasoning over alternatives — a teacher that says "60 % *ricorso*, 25 %
*gravame*, 10 % *appello*" is transmitting a judgement about near-synonyms
that a 4 B model would never extract from raw text on its own.

**Phase 3 — Quantization and GGUF export.** Convert to F16, then
`llama-quantize` to Q4_K_M: weights drop from 16 bits to roughly 4.5 bits on
average, block-wise, keeping higher precision for the tensors that need it.
Runs on CPU, on the serial partition, and costs no GPU budget at all.

**Phase 4 — Identity.** A short LoRA so the model knows what it is. The rule
that matters: merge the adapter into the weights *before* quantizing, or you
export the pre-identity model after paying for the training. We have made
that mistake once.

## Anatomy of a single Phase 1 job

Eight sub-phases. Two of them account for nearly all the wall-clock, and one
of those two is where the job died on the first attempt.

### 1. Pre-flight (~1 min)

Check the GPUs are there, the dataset is on disk, the tokenizer is in the
local cache. Compute nodes have no outbound network, so anything not
pre-fetched is unreachable — a missing file discovered here costs a queue
slot, discovered later it costs four A100s for however long the job had
already run.

### 2. Dataset (2 min warm, ~4 min cold)

Byte-level BPE tokenization and packing into 2048-token blocks. Pure CPU
work: table lookups, string handling, branching. GPUs sit at 0 %, correctly —
this is not a GPU workload. It is, however, a workload that should not be
running on a node whose four A100s are being billed by the hour; it belongs
on the serial partition, cached to disk, and read back by the GPU job.

### 3. Loading the weights (~25 min warm, ~40 min cold)

26 safetensors shards, 61 GB, off a parallel filesystem. Every rank reads the
whole checkpoint and keeps its quarter, so the aggregate read is about
244 GB. System memory climbs to roughly 250 GiB across the four ranks while
GPU memory sits flat at its resting partition size. This is the longest phase
of the job and it produces no output whatsoever.

### 4. LoRA injection and engine construction (~30 s)

The adapter is attached and DeepSpeed builds its partition bookkeeping. This
is the first line in the log that tells you anything useful:

```
trainable params: 106,954,752 || all params: 30,639,077,376 || trainable%: 0.3491
```

Worth checking against arithmetic rather than eyeballing. LoRA adds
`r × (in + out)` per module; with r=128, hidden 2048, 32 heads of 128 and 4
KV heads, one layer is 786,432 + 327,680 + 327,680 + 786,432 = 2,228,224, and
48 layers give exactly 106,954,752. An exact match proves which modules were
targeted — in our case that the MoE router was *not* among them, which was
the intent.

### 5. The first forward — where it breaks

ZeRO-3 keeps parameters partitioned at rest but must materialise a whole
layer on **every** GPU to compute it. On a dense model that is a detail. On a
Mixture-of-Experts it is the whole story: one Qwen3-30B-A3B layer holds 128
experts × 3 matrices ≈ 625 M parameters, 1.25 GB in bf16, against roughly
25 M for a dense layer of the same hidden size. Fifty times larger.

Configured with limits tuned for dense models — a live-parameter cap of 1e9
with matching reuse distance and prefetch — several such layers become
resident at once. On our first attempt GPU memory went from 17.5 GB to full
in twenty-two seconds:

```
CUDACachingAllocator.cpp:528] memory mapping failed with OOM on device 1
while trying to map 20971520 bytes (free: 13107200, total: 68099571712)
```

12.5 MB free of 63.4 GiB, failing to map 20 MB. The node had 270 GB of system
memory unused at that moment, and it was worth nothing: host RAM and GPU HBM
are separate physical memories on an A100 SXM node, and an allocator that has
run out of device memory cannot borrow from the other one unless you have
explicitly configured offloading.

The fix is to cap the gathered working set — 1e8 live parameters with bounded
prefetch and reduce buckets — not to buy a bigger GPU.

### 6. Steady-state training

Forward, backward, optimizer step, repeat. This is the part everyone
imagines when they say "training", and on this job it is preceded by nearly
half an hour of everything else.

### 7. Checkpointing

Every 500 steps, keeping the last two. The checkpoint is what makes the next
sub-phase survivable.

### 8. The walltime cap

The partition caps jobs at 24 hours. A multi-day phase is therefore a chain
of jobs with `afterany` dependencies: when job *k* is killed at the cap, job
*k+1* starts and resumes from the latest checkpoint. Preemption is not an
exception to plan for, it is the normal control flow.

## The lesson that generalises

For thirty of the first thirty-five minutes, this job produces no output. Not
because anything is wrong — reading a checkpoint and building partitions are
genuinely slow — but because the libraries involved log on completion rather
than during, and because `tqdm` disables itself when stderr is not a TTY,
which under `sbatch` it never is.

We diagnosed the first run by hand: `srun` into the allocation from a login
node, read `/proc`, sample `top`, work out from context-switch counts and
resident-memory deltas whether four processes were computing or deadlocked.
That took about as long as the phase we were diagnosing, and it produced one
wrong conclusion along the way — `ps` reports `%CPU` averaged over a
process's lifetime, not instantaneously, which made an idle supervisor look
busy.

The fix was fifteen lines of shell: the job now writes one line a minute with
GPU utilisation, GPU memory, summed worker RSS and load average.

```
[hb] +01443s  gpu util 0,0,0,0 %  mem 16780,16780,16780,16780 MiB  rss 268.2 GiB  load 16.84
```

Those four numbers separate every state that mattered. GPU idle with RSS
climbing is a checkpoint load. GPU idle with RSS flat is engine
initialisation, or a hang. GPU busy is training. It is not sophisticated
observability; it is the minimum a long-running batch job owes whoever is
watching it, and almost none of them do it.

## Numbers

| | |
|---|---|
| Node | 4 × A100 64 GB, 32 cores, ~503 GiB RAM |
| Teacher | Qwen3-30B-A3B-Base — 30.5 B total, 3.3 B active |
| Student | Qwen3-4B-Base |
| Trainable (Phase 1) | 106,954,752 — 0.3491 % |
| Corpus | 1,127,316 / 11,387 chunks, ~700 M tokens |
| Time to first step, cold cache | ~40 min |
| Time to first step, warm cache | (measuring) |
| Peak system memory, all ranks | ~270 GiB |
| GPU memory, resting partition | 16.8 GiB per GPU |
| Node-hours lost to the ZeRO-3 OOM | 0.67 |

_Allocation EHPC-AIF-2026PG01-1147 on Leonardo Booster (CINECA), EuroHPC JU._
