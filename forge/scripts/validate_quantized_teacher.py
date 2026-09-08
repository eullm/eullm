#!/usr/bin/env python3
"""Gate Phase 2 on how much the quantized teacher's logits actually moved.

In distillation the teacher's output distribution IS the training target, so
a quantization error there is not a degradation of training — it is copied
into the student by construction. That makes "we quantized the teacher to fit
it in memory" a claim that has to be measured before Phase 2 runs, not a note
in a config header.

Both teachers are loaded at once — bf16 as the reference, quantized as the
candidate — and run over real samples from the validation corpus. For every
predicted position we compare the two distributions:

  * **KL(bf16 || quantized)** in nats, averaged over positions. Reads as "how
    much information is lost by believing the quantized model instead of the
    reference". Zero means identical.
  * **top-1 / top-5 agreement**: how often the quantized model's most likely
    token, and its top-5 set, match the reference. This is the part a reader
    can interpret without a background in information theory, and the part
    that predicts whether generations diverge.

Exits non-zero when any threshold is missed, so an sbatch script with
`set -e` refuses to start Phase 2 rather than distilling from a teacher
nobody checked.

    python forge/scripts/validate_quantized_teacher.py \\
        --model Qwen/Qwen3-30B-A3B-Base --bits 8 \\
        --adapter ./checkpoints/qwen3_30b_a3b_legal_it_continued_pt \\
        --data "$EULLM_DATA_DIR/val.jsonl" --samples 200

THE DEFAULT THRESHOLDS ARE A STARTING POINT, NOT A RESULT. They are set
where 8-bit weight quantization of a healthy model should land comfortably;
nobody has yet measured this model, which is the entire purpose of the
script. Record what the first run reports, then set the thresholds from
evidence and say so in the report.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


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


def build_model(model_id: str, bits: int | None, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict = {"dtype": torch.bfloat16, "device_map": "auto"}
    if bits in (4, 8):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=bits == 8,
            load_in_4bit=bits == 4,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


def compare(reference, candidate, tokenizer, texts: list[str], seq_len: int) -> dict:
    """Mean KL and top-k agreement between two models over the same inputs."""
    import torch
    import torch.nn.functional as F

    kl_total = 0.0
    top1_hits = top5_hits = positions = 0

    for i, text in enumerate(texts, start=1):
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=seq_len
        )
        ids = enc["input_ids"]
        if ids.shape[1] < 8:
            continue
        with torch.no_grad():
            ref_logits = reference(ids.to(reference.device)).logits.float()
            cand_logits = candidate(ids.to(candidate.device)).logits.float()
        cand_logits = cand_logits.to(ref_logits.device)

        # Drop the last position: it predicts a token outside the window.
        ref = ref_logits[0, :-1, :]
        cand = cand_logits[0, :-1, :]

        ref_logprobs = F.log_softmax(ref, dim=-1)
        cand_logprobs = F.log_softmax(cand, dim=-1)
        # KL(reference || candidate), summed over vocabulary, mean over
        # positions. `kl_div` expects (input=log q, target=p).
        kl = F.kl_div(
            cand_logprobs, ref_logprobs, log_target=True, reduction="none"
        ).sum(dim=-1)
        kl_total += float(kl.sum())

        ref_top5 = ref.topk(5, dim=-1).indices
        cand_top1 = cand.argmax(dim=-1)
        top1_hits += int((cand_top1 == ref_top5[:, 0]).sum())
        top5_hits += int((cand_top1.unsqueeze(-1) == ref_top5).any(dim=-1).sum())
        positions += ref.shape[0]

        if i % 25 == 0:
            print(f"  {i}/{len(texts)} samples, {positions} positions", flush=True)

    if positions == 0:
        raise SystemExit("no usable samples — check --data and --field")
    return {
        "positions": positions,
        "mean_kl_nats": kl_total / positions,
        "top1_agreement": top1_hits / positions,
        "top5_agreement": top5_hits / positions,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Teacher model id or path")
    p.add_argument("--bits", type=int, choices=(4, 8), required=True)
    p.add_argument("--adapter", default=None, help="Phase-1 LoRA adapter (optional)")
    p.add_argument("--data", type=Path, required=True, help="JSONL to sample from")
    p.add_argument("--field", default="text")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-kl", type=float, default=0.05, help="nats, mean per position")
    p.add_argument("--min-top1", type=float, default=0.95)
    p.add_argument("--min-top5", type=float, default=0.99)
    p.add_argument("--report", type=Path, default=None, help="Write metrics as JSON")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    texts = load_samples(args.data, args.samples, args.seed, args.field)
    print(f"[..] {len(texts)} samples from {args.data}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("[..] loading bf16 reference", flush=True)
    reference = build_model(args.model, None, args.adapter)
    print(f"[..] loading {args.bits}-bit candidate", flush=True)
    candidate = build_model(args.model, args.bits, args.adapter)

    metrics = compare(reference, candidate, tokenizer, texts, args.seq_len)
    metrics.update(
        model=args.model, bits=args.bits, adapter=args.adapter,
        samples=len(texts), seq_len=args.seq_len,
    )

    checks = [
        ("mean KL (nats)", metrics["mean_kl_nats"], args.max_kl, "<="),
        ("top-1 agreement", metrics["top1_agreement"], args.min_top1, ">="),
        ("top-5 agreement", metrics["top5_agreement"], args.min_top5, ">="),
    ]
    print()
    print("=" * 60)
    failed = []
    for name, value, threshold, op in checks:
        ok = value <= threshold if op == "<=" else value >= threshold
        print(f"  {name:>18}: {value:.5f}   ({op} {threshold})   {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)
    print(f"  {'positions':>18}: {metrics['positions']:,}")
    print("=" * 60)

    metrics["passed"] = not failed
    if args.report:
        args.report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[ok] metrics written to {args.report}")

    if failed:
        print(
            f"\n[!!] {args.bits}-bit teacher rejected on: {', '.join(failed)}.\n"
            "     In distillation the teacher's distribution is the target, so\n"
            "     this error would be trained into the student. Use a higher\n"
            "     precision, or change the thresholds deliberately and record\n"
            "     why in docs/legal-it-4b-strategy.md.",
            file=sys.stderr,
        )
        return 1
    print(f"\n[ok] {args.bits}-bit teacher is within thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
