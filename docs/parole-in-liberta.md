# Parole in libertà

> Ideas that are not decisions. Nothing here is committed to, scheduled, or
> promised — but each one was worth more than the conversation it appeared in,
> and losing it would be worse than writing it down.
>
> An entry earns its place by carrying enough detail to be picked up cold:
> what the idea is, what already exists, what is missing, what it would buy,
> and what would have to be true. An entry that cannot say those five things
> is a mood, not an idea.

---

## 1. The Engine as the teacher, and EULLM as a complete framework

**Date: 2026-09-12. Came out of four consecutive failed jobs trying to make
vLLM run on Leonardo.**

### The idea

EULLM already ships a single static binary that loads GGUF models and runs
inference on CUDA, Metal, Vulkan and CPU. The v1.1 distillation pipeline needs
something that, given a document, returns the teacher's top-K distribution for
every position — teacher forcing, not generation.

**llama.cpp can already do this.** `llama_get_logits_ith()` returns the full
logit vector for a position, and a `llama_batch` with the per-token logits
flag set gives every position of a sequence in one decode. The computation
exists; what is missing is a way to ask for it and a format to receive it in.

So: teach the Engine to export teacher distributions, and the distillation
pipeline stops depending on a Python inference stack entirely.

### Why this is more than a convenience

The strategic point, and the reason this is written down rather than left in a
terminal: today EULLM is **a runtime**. A very good drop-in for Ollama with an
audit trail, but a thing that *runs* models other people made.

An Engine that also produces the signal needed to *make* models is a different
category of product. Not "a runtime plus some scripts" — one binary that both
serves a model and supplies the teacher signal for compressing the next one.
Nobody in the Ollama-compatible space does this.

It also makes the compression pipeline portable in a way the Python path never
will be. The same binary runs on ROCm and Metal, so the same distillation
pipeline would run on LUMI's MI250X and on a laptop, not only where the CUDA
wheel matrix happens to line up.

### What already exists

* `llama_get_logits_ith()` — full logits per position.
* `llama_batch` with per-token logits flags — teacher forcing in one call.
* Qwen3-MoE support in llama.cpp.
* Multi-GPU layer splitting, so a 30 B teacher spreads across a node.
* The `TeacherLogitProvider` seam in ADR-001 Part 3, which already names
  `EullmTeacherProvider` as a later implementation.

### What is missing

* **An export path.** The Engine has no way to say "score this text and give
  me the top-K per position". A subcommand or an endpoint, plus a wire format.
* A decision on **where the output goes**: straight into the cache format in
  `forge/eullm_forge/teacher_cache/`, or a stream the Python side consumes.
* Batching. Scoring one document at a time would be far too slow for a corpus
  of 700 M positions.

### What would have to be true

* **The numerics have to be acceptable.** llama.cpp's kernels are not
  transformers' kernels, and a GGUF teacher is quantized where the current one
  is bf16 or int8-through-bitsandbytes. Measurable with exactly the method P6
  used: KL and top-k agreement against the bf16 reference, on the same
  validation sample. If a Q8 GGUF teacher lands near P6's 0.006 nats, the
  question is closed.
* **Throughput has to be competitive.** llama.cpp is fast at generation; bulk
  prompt scoring with logits for every position is a different access pattern
  and has to be measured, not assumed.
* Somebody has to want to maintain a second consumer of the logits path.

### Why not now

It is Rust work in `engine/`, and the pipeline needs to move this month on
the allocation we have. Commissioned on evidence rather than on frustration
with pip — and the evidence to gather first is in entry 2 below.

---

## 2. vLLM is an optimisation, not an enabler

**Same day, same cause. This one is nearly actionable and mostly needs
measuring.**

The v1.1 plan put vLLM at the centre of design A because of `prompt_logprobs`.
Four jobs failed getting it to run on Leonardo — CUDA 13 wheels against a 12.2
driver, then a stale `torchcodec`, then a venv with `nvidia-nccl` in two CUDA
versions at once.

Meanwhile P6 has been scoring documents with **plain transformers** all along,
and its timings say something the plan did not account for:

| | |
|---|---|
| 20 samples | 9 min |
| 200 samples | 10 min |

180 extra documents cost about a minute — the rest was loading. That is
~90,000 positions in ~60 s **with two models resident**. One model, roughly
3,000 positions/s, and at batch size 1.

Against a corpus of ~700 M positions:

```
700,000,000 / 3,000  ~=  65 hours on one node
```

Batching should cut that substantially, since P6 processes one sequence at a
time. So a full teacher pass with transformers is on the order of **one to
three days and 30-70 node-hours** — against 978 free at the time of writing.

vLLM might do it in a third of that. **Real, but not the difference between
possible and impossible**, which is how the plan had been treating it.

What this suggests: write the transformers `TeacherLogitProvider` first, since
it works today and needs nothing installed; measure positions/s properly with
batching; and let that number decide whether a faster backend is worth
anything. Both the vLLM provider and the Engine provider then become
optimisations chosen on evidence, which is what ADR-001's provider seam was
for.

---

## How to use this file

Add an entry when something is too good to lose and too early to schedule.
Take one out when it becomes an ADR, a backlog item, or a decision not to do
it — and when it does, say here what it became, so the file records what
happened to ideas rather than only that they were had.
