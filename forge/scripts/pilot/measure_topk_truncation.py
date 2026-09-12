#!/usr/bin/env python3
"""P3: how much of the teacher's distribution does top-K actually keep?

ADR-001 Part 5 says K is measured, not chosen, and this is the measurement.
It decides the size of the cache — K=32 is ~143 GB over the corpus against
~277 GB at K=64 — and the pre-registration predicts that on Italian case law,
a narrow and highly conventional register, K=32 already captures ≥99% of the
mass at T=1.

**No vLLM.** The full vocabulary distribution comes from a plain transformers
forward, which is the same stack Phase 1 has been running for days. That is
deliberate: P3 answers a question about the corpus and the teacher, not about
an inference backend, so it must not be blocked behind P1.

### The two thresholds in P3 are the same number

The prediction asks for retained mass ≥ 99% **and** truncation KL < 0.01
nats, as though they were independent checks. They are not. With `q` the
top-K distribution renormalised over retained mass `M`:

    KL(q || p) = sum_topK (p_i/M) log((p_i/M) / p_i) = -log M

So `M = 0.99` is exactly `KL = 0.01005`. Both are reported, because the mass
is the readable one and the KL is the one that composes with the training
objective — but a reader should know they are one measurement, not two, and
that agreement between them is arithmetic rather than evidence.

Reported per K and per temperature, because truncation bites harder as T
rises: a hotter softmax spreads mass into the tail the top-K does not hold.

    python measure_topk_truncation.py --model Qwen/Qwen3-30B-A3B-Base \\
        --data "$EULLM_DATA_DIR/val.jsonl" --samples 64 \\
        --k 16 32 64 128 --report p3.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

DEFAULT_KS = (16, 32, 64, 128)
DEFAULT_TEMPERATURES = (1.0, 2.0, 4.0)


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="Phase-1 LoRA adapter (optional)")
    p.add_argument("--bits", type=int, choices=(4, 8), default=None,
                   help="quantize the teacher; omit for bf16")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--field", default="text")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--temperatures", type=float, nargs="+",
                   default=list(DEFAULT_TEMPERATURES))
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    texts = load_samples(args.data, args.samples, args.seed, args.field)
    if not texts:
        print(f"[FAIL] no usable samples in {args.data}", file=sys.stderr)
        return 2
    print(f"[..] {len(texts)} samples, loading {args.model}", flush=True)

    kwargs: dict = {"dtype": torch.bfloat16, "device_map": "auto"}
    if args.bits:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.bits == 8,
            load_in_4bit=args.bits == 4,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("[ok] loaded", flush=True)

    ks = sorted(set(args.k))
    temps = list(args.temperatures)
    # Accumulate retained mass per (temperature, K) and count positions once.
    mass = {(t, k): 0.0 for t in temps for k in ks}
    positions = 0

    for i, text in enumerate(texts, start=1):
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=args.seq_len)
        ids = enc["input_ids"]
        if ids.shape[1] < 8:
            continue
        with torch.no_grad():
            logits = model(ids.to(model.device)).logits.float()
        # Drop the last position: it predicts a token outside the window.
        logits = logits[0, :-1, :]
        positions += logits.shape[0]

        for t in temps:
            probs = torch.softmax(logits / t, dim=-1)
            # One sort serves every K, and the largest K is a prefix of it.
            top = probs.topk(max(ks), dim=-1).values
            cumulative = top.cumsum(dim=-1)
            for k in ks:
                mass[(t, k)] += float(cumulative[:, k - 1].sum())

        if i % 8 == 0:
            print(f"  {i}/{len(texts)} samples, {positions} positions", flush=True)

    if positions == 0:
        print("[FAIL] no usable positions", file=sys.stderr)
        return 2

    import math

    rows = []
    for t in temps:
        for k in ks:
            m = mass[(t, k)] / positions
            rows.append({
                "temperature": t,
                "k": k,
                "retained_mass": m,
                # KL(truncated-renormalised || full), which reduces exactly to
                # -log M. Reported because it is the form that composes with
                # the training objective, not because it is new information.
                "truncation_kl_nats": -math.log(max(m, 1e-12)),
            })

    print()
    print("=" * 56)
    print(f"{'T':>5} {'K':>5} {'retained mass':>16} {'KL (nats)':>12}")
    for r in rows:
        print(f"{r['temperature']:>5.1f} {r['k']:>5} "
              f"{r['retained_mass']:>15.4%} {r['truncation_kl_nats']:>12.5f}")
    print("=" * 56)
    print(f"positions: {positions:,}")

    report = {
        "model": args.model,
        "adapter": args.adapter,
        "bits": args.bits,
        "samples": len(texts),
        "seq_len": args.seq_len,
        "positions": positions,
        "rows": rows,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[ok] report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
