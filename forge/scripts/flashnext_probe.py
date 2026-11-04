#!/usr/bin/env python3
"""Can Qwen3.8-Flash-Next run, and learn through its n-gram table alone, on one JUPITER node?

    python forge/scripts/flashnext_probe.py --model Qwen/Qwen3.8-Flash-Next \\
        --texts $FORGE_DATA_DIR/memory/codes/program.jsonl

The model (Qwen Community License 1.0; ~180B stored: a 125B MoE with 6B
active, a 51B n-gram embedding at layer 2, a 4B MTP head) is the legal-it
roadmap's phase 3, the n-gram memory, already built: if its table can be
trained on the law with the backbone frozen, E2 needs no module of ours.
Three measurements, printed:

1. Loading: transformers keeps the n-gram table (~95 GiB) off the GPUs
   (``_no_placement_params``) and looks it up where it lies; the rest,
   ~258 GB in BF16, is spread over the four GH200. Seconds and memory a GPU.
2. Answering: three public questions on the civil code, greedy, thinking
   off; the answers and the tokens a second.
3. Learning: everything frozen but the n-gram table, made to give sparse
   gradients (only the rows a batch touched), updated by plain SGD on the
   CPU; a few steps on one batch of legislation, the loss each step. A loss
   that falls says the table alone can carry what is trained; the seconds a
   step say what a run would cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

QUESTIONS = (
    "Che cosa prevede l'art. 2043 del codice civile?",
    "In che cosa consiste la risoluzione per inadempimento secondo il codice civile?",
    "Chi ha l'onere della prova nel processo civile, secondo l'art. 2697 c.c.?",
)


def gib(n: float) -> str:
    return f"{n / 2**30:.1f} GiB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--texts", type=Path, required=True, help="JSONL with a 'text' field")
    ap.add_argument("--gpu-gib", type=int, default=80, help="weights a GPU at most")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.5)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    n_gpu = torch.cuda.device_count()
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        max_memory={**{i: f"{args.gpu_gib}GiB" for i in range(n_gpu)}, "cpu": "400GiB"})
    model.eval()
    tables = [m for n, m in model.named_modules() if n.endswith("ple_embedding.ngram_embedding")]
    print(f"[probe] loaded in {time.monotonic() - t0:.0f} s on {n_gpu} GPUs", flush=True)
    for i in range(n_gpu):
        print(f"[probe]   GPU {i}: {gib(torch.cuda.memory_allocated(i))} allocated", flush=True)
    for t in tables:
        w = t.weight
        print(f"[probe]   n-gram table {tuple(w.shape)} {w.dtype} on {w.device}, "
              f"{gib(w.numel() * w.element_size())}", flush=True)
    if not tables:
        print("[!!] no n-gram table found", flush=True)
        return 1
    first = next(p.device for p in model.parameters() if p.device.type == "cuda")

    # 2. Answering.
    for q in QUESTIONS:
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(first)
        t1 = time.monotonic()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        new = out[0, ids["input_ids"].shape[1]:]
        dt = time.monotonic() - t1
        print(f"[probe] Q: {q}\n[probe] A ({len(new)} tokens, {len(new) / dt:.1f} tok/s): "
              f"{tok.decode(new, skip_special_tokens=True).strip()}\n", flush=True)

    # 3. Learning through the table alone.
    for p in model.parameters():
        p.requires_grad_(False)
    for t in tables:
        t.weight.requires_grad_(True)
        t.sparse = True                     # gradients on the rows looked up only
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    texts = []
    with args.texts.open(encoding="utf-8") as f:
        for line in f:
            text = (json.loads(line).get("text") or "").strip()
            if len(text) > 400:
                texts.append(text)
            if len(texts) >= 64:
                break
    enc = tok("\n\n".join(texts), add_special_tokens=False)["input_ids"]
    need = args.batch * args.seq
    if len(enc) < need:
        print(f"[!!] only {len(enc)} tokens of text for a batch of {need}", flush=True)
        return 1
    batch = torch.tensor(enc[:need]).view(args.batch, args.seq).to(first)
    for step in range(args.steps):
        t1 = time.monotonic()
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        rows = 0
        with torch.no_grad():
            for t in tables:
                g = t.weight.grad
                if g is None:
                    continue
                g = g.coalesce()
                rows += g.indices().shape[1]
                t.weight.index_add_(0, g.indices()[0], g.values().to(t.weight.dtype),
                                    alpha=-args.lr)
                t.weight.grad = None
        print(f"[probe] step {step + 1}: loss {loss.item():.4f}, {rows:,} table rows updated, "
              f"{time.monotonic() - t1:.1f} s", flush=True)
    for i in range(n_gpu):
        print(f"[probe]   GPU {i}: peak {gib(torch.cuda.max_memory_allocated(i))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
