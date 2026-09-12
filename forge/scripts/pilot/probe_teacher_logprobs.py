#!/usr/bin/env python3
"""P1: can vLLM score a corpus with the teacher, and what does it cost?

Design A in ADR-001 caches the teacher's top-K distribution once and trains
any number of students against the cache. The whole design rests on one
unverified assumption: that vLLM will return top-K logprobs **for every
position of a prompt**, in one pass, on a Qwen3 MoE. If it will not, the
provider interface gets built around a different backend and the rest of the
plan is unaffected — which is exactly why this runs first, and why it is
cheap.

The pre-registration (docs/paper/v11-pilot-preregistration.md) calls P1
falsified if the call fails, if fewer than K values come back per position,
or if scoring is within 20% of generation-mode throughput — that last one
because it would mean the path is not doing the extra work we expect and
something else is wrong.

**"Throughput" is defined here as positions scored per second**, and the
definition matters more than the number. Generation mode emits K logprobs per
*generated* token, one token at a time; prompt scoring emits K logprobs per
*prompt* position, a whole sequence at a time. Tokens/s would compare two
different things. What the cache actually needs is "how long to score N
positions", so that is what both halves measure.

Two things this deliberately records rather than assumes:

  * **What comes back is logprobs, not logits.** ADR-001 Part 4 specifies a
    provider returning raw logits. vLLM normalises. The difference is one
    additive constant per position, and the cache format already stores a
    per-temperature normaliser for exactly this reason — but the assumption
    should be corrected in the ADR from a measurement, not from memory.
  * **How many entries per position actually arrive.** vLLM may include the
    prompt's own token beyond the top-K, so K+1 is a pass, not a failure.
    Fewer than K is the falsification.

    python probe_teacher_logprobs.py --model Qwen/Qwen3-30B-A3B-Base \\
        --data "$EULLM_DATA_DIR/val.jsonl" --samples 32 --top-k 64 \\
        --tensor-parallel-size 4 --report p1.json

Failure is an answer. A backend that cannot do this exits non-zero with the
reason written to the report, rather than raising through the sbatch.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# ── vLLM settings this cluster needs, set here rather than in a shell ────
#
# Every one of these was first discovered as a failed job and then typed by
# hand into whichever terminal was open. That does not survive: a variable
# exported in a shell is not versioned, reaches a batch job only if sbatch
# happens to export it, and differs silently between login nodes — all three
# of which cost jobs on this allocation. Setting them next to the code that
# needs them means they cannot be forgotten, and the reason travels with them.
#
# `setdefault`, not assignment: an explicit value from the caller still wins,
# so either of these can be flipped back to measure whether it still matters.
#
# Must precede `import vllm`, which reads them at import time — hence module
# level rather than inside main().
os.environ.setdefault(
    # FlashInfer JIT-compiles its sampling kernels on first use and looks for
    # nvcc at /usr/local/cuda/bin, which does not exist here: CUDA comes from
    # modules at a Spack path. The build fails with `ninja: build stopped`.
    # We also never sample — this probe reads prompt distributions — so the
    # sampler backend is pure cost.
    "VLLM_USE_FLASHINFER_SAMPLER", "0",
)
os.environ.setdefault(
    # fork() in a process that has already initialised CUDA is undefined
    # behaviour, and vLLM's default start method depends on the platform.
    "VLLM_WORKER_MULTIPROC_METHOD", "spawn",
)


def resolve_local_model(model: str) -> str:
    """A repo id becomes the local snapshot it was prefetched into.

    Compute nodes have no network, which is why models are prefetched to
    `$HF_HOME` from a login node. `HF_HUB_OFFLINE=1` is supposed to make that
    enough, and for transformers it is — but vLLM resolves a repo id against
    the Hub anyway, to list the repository's files, and on a node with no
    route out that fails before anything else can happen:

        Could not reach the Hub ([Errno 101] Network is unreachable)
        ERROR repo_utils.py:169 Error retrieving file list ...
        [FAIL] could not load the model: [Errno 101] Network is unreachable

    Handing it an absolute path skips that lookup entirely. Four evenings of
    failures were attributed to CUDA versions, a stale torchcodec and two
    NCCLs before this turned out to be underneath all of them — each real, and
    each hiding the next.

    A path that already exists is returned untouched, so this is safe to apply
    unconditionally.
    """
    if Path(model).exists():
        return model
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return model
    snapshots = Path(hf_home) / "hub" / f"models--{model.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return model

    # refs/main names the commit the prefetch landed on. Preferring it over
    # "whatever directory sorts last" means two jobs cannot silently score
    # against different revisions of the same model.
    ref = snapshots.parent / "refs" / "main"
    if ref.is_file():
        candidate = snapshots / ref.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return str(candidate)
    dirs = sorted((d for d in snapshots.iterdir() if d.is_dir()),
                  key=lambda d: d.stat().st_mtime)
    return str(dirs[-1]) if dirs else model


def load_samples(path: Path, n: int, seed: int, field: str) -> list[str]:
    """Reservoir-sample n texts without holding the whole corpus in memory."""
    rng = random.Random(seed)
    picked: list[str] = []
    seen = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get(field)
            except json.JSONDecodeError:
                continue
            if not isinstance(text, str) or len(text) < 200:
                continue
            seen += 1
            if len(picked) < n:
                picked.append(text)
            else:
                j = rng.randrange(seen)
                if j < n:
                    picked[j] = text
    return picked


def inspect_prompt_logprobs(outputs, top_k: int) -> dict:
    """What vLLM actually returned, as opposed to what we hoped.

    Position 0 carries no distribution — nothing precedes it — and vLLM
    returns None there. That is expected and not counted as a short position.
    """
    scored = 0
    short: list[int] = []
    widths: set[int] = set()
    sample_entry = None
    for out in outputs:
        plp = getattr(out, "prompt_logprobs", None)
        if not plp:
            continue
        for pos, entry in enumerate(plp):
            if entry is None:          # first position: no context
                continue
            widths.add(len(entry))
            if len(entry) < top_k:
                short.append(len(entry))
            if sample_entry is None:
                sample_entry = next(iter(entry.values()))
            scored += 1
    return {
        "positions_scored": scored,
        "entries_per_position": sorted(widths),
        "positions_below_k": len(short),
        # A Logprob object carries .logprob (normalised) — recording the
        # attribute names is how the ADR's "raw logits" claim gets corrected
        # from evidence.
        "value_attributes": sorted(
            a for a in dir(sample_entry) if not a.startswith("_")
        ) if sample_entry is not None else [],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--field", default="text")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--gen-tokens", type=int, default=64,
                   help="tokens to generate for the comparison half")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    # Resolved once, and both the banner and the report carry the path
    # actually used — "which revision did this score against" has to be
    # answerable from the artefact, not from the command line someone typed.
    model_path = resolve_local_model(args.model)
    report: dict = {"model": args.model, "model_path": model_path,
                    "top_k": args.top_k,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "seq_len": args.seq_len, "samples": args.samples}

    def finish(code: int) -> int:
        if args.report:
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\n[ok] report written to {args.report}")
        return code

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:                       # noqa: BLE001 - reported
        report["verdict"] = "vllm-unavailable"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAIL] cannot import vllm: {exc}", file=sys.stderr)
        return finish(2)

    texts = load_samples(args.data, args.samples, args.seed, args.field)
    if not texts:
        report["verdict"] = "no-samples"
        print(f"[FAIL] no usable samples in {args.data}", file=sys.stderr)
        return finish(2)
    print(f"[..] {len(texts)} samples, loading {model_path} "
          f"on TP={args.tensor_parallel_size}", flush=True)

    t0 = time.time()
    try:
        llm = LLM(model=model_path,
                  tensor_parallel_size=args.tensor_parallel_size,
                  max_model_len=args.seq_len,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  enforce_eager=False)
    except Exception as exc:                       # noqa: BLE001 - reported
        report["verdict"] = "load-failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAIL] could not load the model: {exc}", file=sys.stderr)
        return finish(2)
    report["load_seconds"] = round(time.time() - t0, 1)
    print(f"[ok] loaded in {report['load_seconds']}s", flush=True)

    # Tokenize and truncate HERE, rather than handing vLLM raw strings.
    #
    # The engine refuses a prompt longer than max_model_len, and a corpus
    # document is far longer than the window we score it in:
    #
    #   prompt_logprobs=64 rejected: This model's maximum context length is
    #   512 tokens. However ... your prompt contains at least 513 input tokens
    #
    # Passing token ids also removes an ambiguity that matters for the numbers
    # this probe reports: positions/s is only comparable across runs if the
    # count of positions is something we set rather than something the
    # tokenizer happened to produce.
    tokenizer = llm.get_tokenizer()

    def as_prompts(limit: int) -> list[dict]:
        limit = max(8, limit)
        return [
            {"prompt_token_ids":
                tokenizer(t, truncation=True, max_length=limit)["input_ids"]}
            for t in texts
        ]

    # One token of headroom for the output max_tokens asks for.
    score_prompts = as_prompts(args.seq_len - 1)
    report["prompt_tokens_total"] = sum(
        len(pr["prompt_token_ids"]) for pr in score_prompts)

    # ── the question: top-K for every prompt position, in one pass ────────
    t0 = time.time()
    try:
        scored = llm.generate(
            score_prompts,
            SamplingParams(max_tokens=1, temperature=0.0,
                           prompt_logprobs=args.top_k),
        )
    except Exception as exc:                       # noqa: BLE001 - reported
        report["verdict"] = "prompt-logprobs-unsupported"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAIL] prompt_logprobs={args.top_k} rejected: {exc}",
              file=sys.stderr)
        return finish(1)
    score_seconds = time.time() - t0

    shape = inspect_prompt_logprobs(scored, args.top_k)
    report["prompt_logprobs"] = shape
    report["score_seconds"] = round(score_seconds, 1)
    report["score_positions_per_second"] = round(
        shape["positions_scored"] / score_seconds, 1) if score_seconds else 0.0

    # ── the comparison: generation mode over the same prompts ─────────────
    t0 = time.time()
    # The generation half needs room for the tokens it will emit as well.
    generated = llm.generate(
        as_prompts(args.seq_len - args.gen_tokens - 1),
        SamplingParams(max_tokens=args.gen_tokens, temperature=0.0,
                       logprobs=args.top_k),
    )
    gen_seconds = time.time() - t0
    gen_positions = sum(len(o.outputs[0].token_ids) for o in generated)
    report["gen_seconds"] = round(gen_seconds, 1)
    report["gen_positions_per_second"] = round(
        gen_positions / gen_seconds, 1) if gen_seconds else 0.0

    fast = report["score_positions_per_second"]
    slow = report["gen_positions_per_second"]
    report["speedup_over_generation"] = round(fast / slow, 2) if slow else None

    # ── verdict against the pre-registered thresholds ─────────────────────
    reasons = []
    if shape["positions_scored"] == 0:
        reasons.append("no positions carried a distribution")
    if shape["positions_below_k"]:
        reasons.append(
            f"{shape['positions_below_k']} positions returned fewer than "
            f"{args.top_k} entries")
    report["verdict"] = "holds" if not reasons else "falsified"
    report["falsification_reasons"] = reasons

    print()
    print("=" * 62)
    print(f"  positions scored      {shape['positions_scored']:>12,}")
    print(f"  entries per position  {str(shape['entries_per_position']):>12}"
          f"   (K={args.top_k}; K+1 is fine, fewer is not)")
    print(f"  scoring               {report['score_positions_per_second']:>12,.1f} pos/s")
    print(f"  generation            {report['gen_positions_per_second']:>12,.1f} pos/s")
    print(f"  speedup               {str(report['speedup_over_generation']):>12}x")
    print(f"  value attributes      {', '.join(shape['value_attributes'][:6])}")
    print(f"  VERDICT               {report['verdict'].upper():>12}")
    for r in reasons:
        print(f"    - {r}")
    print("=" * 62)

    return finish(0 if report["verdict"] == "holds" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
