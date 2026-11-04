#!/usr/bin/env python3
"""Train a hashed n-gram memory on a frozen model, or the LoRA control beside it.

Line A of the Fast Lane plan (`docs/cineca/fastlane-plan.md`), experiment
E2-mini of `docs/research/legal-memory-plan.md`: the model is frozen, a
memory (`eullm_forge.memory`) is grafted after one early layer and only the
memory learns, on the memory corpus of the articles (program rows and
teacher rewrites, one JSON object per line with a ``text`` field). The
control (``--mode lora``) trains a LoRA of about as many parameters on the
same tokens in the same order, so the closed-book suite can tell whether the
memory adds anything a LoRA does not.

Built for 2-hour links of a chain: ``--stop-after`` ends the link at a step
boundary, every save writes the trained part, its optimiser and the data
position, and a link resumes from the last save **with the data where it
stopped** (the rule of `docs/cineca/allocation-recovery.md`, point 5). A run
that has finished writes ``DONE`` and later links exit at once.

    python forge/scripts/ngram_train.py --model <merged model dir> \\
        --data $WORK/eullm_runs/memory/program.jsonl $WORK/eullm_runs/memory/rewrites*.jsonl \\
        --out $WORK/eullm_runs/lineA/a2-r2l-ngram --mode ngram --stop-after 6600

Output: ``<out>/memory`` (the memory file, `NGramMemory.load`) or
``<out>/adapter`` (the LoRA), ``<out>/train-log.jsonl`` (one line per logged
step), and ``<out>/ckpt`` while training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------------- data
def read_texts(paths: list[Path]) -> list[str]:
    """Every ``text`` of the files, in file order; half-written last lines skipped."""
    texts = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = (row.get("text") or "").strip()
                if text:
                    texts.append(text)
    return texts


def pack(token_lists, seq_len: int, eos: int):
    """Documents joined with ``eos`` and cut into rows of exactly ``seq_len``.

    An int32 array of shape (rows, seq_len): a hundred million tokens as
    Python lists would take gigabytes, as an array 400 MB. The tail shorter
    than a row is dropped.
    """
    import numpy as np

    parts = []
    for toks in token_lists:
        parts.append(np.asarray(toks, dtype=np.int32))
        parts.append(np.asarray([eos], dtype=np.int32))
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    n = len(flat) // seq_len
    return flat[:n * seq_len].reshape(n, seq_len)


def tokenized_rows(tok, paths: list[Path], seq_len: int, cache: Path):
    """The packed rows, tokenised once per run and kept in ``cache`` for the later links."""
    import numpy as np

    key = {"data": [str(p) for p in paths], "seq_len": seq_len,
           "sizes": [p.stat().st_size for p in paths]}
    meta = cache.with_suffix(".json")
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == key:
        return np.load(cache), None
    texts = read_texts(paths)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    ids = []
    for i in range(0, len(texts), 1000):
        ids.extend(tok(texts[i:i + 1000], add_special_tokens=False)["input_ids"])
    rows = pack(ids, seq_len, eos)
    tmp = cache.with_name(cache.name + ".tmp.npy")
    np.save(tmp, rows)
    os.replace(tmp, cache)
    meta.write_text(json.dumps(key))
    return rows, len(texts)


def epoch_order(n_rows: int, epoch: int, seed: int) -> list[int]:
    """A fixed permutation per epoch: a resumed link sees the rows a fresh run would."""
    order = list(range(n_rows))
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    return order


def batches(n_rows: int, batch: int, seed: int, start_step: int, epochs: int):
    """(step, epoch, row indices) from ``start_step`` on, the last partial batch dropped."""
    per_epoch = n_rows // batch
    if per_epoch == 0:
        raise SystemExit(f"[ngram] {n_rows} rows do not make one batch of {batch}")
    for step in range(start_step, per_epoch * epochs):
        epoch, k = divmod(step, per_epoch)
        order = epoch_order(n_rows, epoch, seed)
        yield step, epoch, order[k * batch:(k + 1) * batch]


def lora_rank_for(model, target_params: int, targets) -> int:
    """The LoRA rank whose adapters have about ``target_params`` parameters.

    ``targets`` as PEFT takes them: a list of module-name suffixes, or a
    regex over the full name (`eullm_forge.identity.lora_target_modules`).
    """
    import re

    import torch

    def hit(name: str) -> bool:
        if isinstance(targets, str):
            return re.fullmatch(targets, name) is not None
        return name.split(".")[-1] in targets

    per_rank = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and hit(name):
            per_rank += mod.in_features + mod.out_features
    if per_rank == 0:
        raise SystemExit("[ngram] no LoRA target found in the model")
    return max(1, round(target_params / per_rank))


# --------------------------------------------------------------------- model
def load_frozen(path: str):
    """The model in bf16, every weight frozen: on the one GPU, spread over several, or on CPU."""
    import torch

    from eullm_forge.identity import load_text_model

    kwargs = {"dtype": torch.bfloat16}
    if torch.cuda.device_count() > 1:
        kwargs["device_map"] = "auto"
    model = load_text_model(path, **kwargs)
    if torch.cuda.device_count() == 1:
        model = model.to("cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the frozen model (merged HF directory)")
    ap.add_argument("--data", nargs="+", type=Path, required=True, help="JSONL with a text field")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("ngram", "lora"), default="ngram")
    ap.add_argument("--layer", type=int, default=1, help="decoder layer the memory follows")
    ap.add_argument("--rows-per-head", type=int, default=262_144)
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--heads-per-ngram", type=int, default=8)
    ap.add_argument("--conv-kernel", type=int, default=4)
    ap.add_argument("--lora-params", type=int, default=0,
                    help="LoRA size to match (default: the n-gram memory these options build)")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr-table", type=float, default=1e-2, help="SparseAdam, the table")
    ap.add_argument("--lr", type=float, default=1e-3, help="AdamW: projections, gate, LoRA")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--log-steps", type=int, default=20)
    ap.add_argument("--stop-after", type=float, default=0, help="seconds; 0 = run to the end")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="for tests")
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    from eullm_forge.memory import NGramMemory, attach, decoder_layers, memory_config_for

    t0 = time.monotonic()
    out = a.out
    if (out / "DONE").exists():
        print(f"[ngram] {out} is finished (DONE): nothing to do", flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    rows, n_texts = tokenized_rows(tok, a.data, a.seq_len, out / f"rows-{a.seq_len}.npy")
    n_tokens = len(rows) * a.seq_len
    where = f"{n_texts:,} texts" if n_texts is not None else "rows from the cache"
    print(f"[ngram] {where}, {len(rows):,} rows of {a.seq_len} "
          f"({n_tokens / 1e6:.1f}M tokens), batch {a.batch}, {a.epochs} epoch(s)", flush=True)

    model = load_frozen(a.model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    dev = next(model.parameters()).device

    mem_cfg = memory_config_for(model, layer=a.layer, rows_per_head=a.rows_per_head,
                                embed_dim=a.embed_dim, heads_per_ngram=a.heads_per_ngram,
                                conv_kernel=a.conv_kernel, seed=a.seed)
    ckpt = out / "ckpt"
    state = {"step": 0}
    if (ckpt / "state.json").exists():
        state = json.loads((ckpt / "state.json").read_text())

    if a.mode == "ngram":
        memory = (NGramMemory.load(ckpt / "memory") if (ckpt / "memory").exists()
                  else NGramMemory(mem_cfg))
        # beside the layer it follows, which device_map may have put on another GPU
        memory = memory.to(next(decoder_layers(model)[memory.config.layer].parameters()).device)
        attach(model, memory)
        dense = [p for n, p in memory.named_parameters() if not n.startswith("table.")]
        opts = [torch.optim.SparseAdam([memory.table.weight], lr=a.lr_table),
                torch.optim.AdamW(dense, lr=a.lr, weight_decay=0.0)]
        trainable = memory.n_params()
    else:
        from peft import LoraConfig, PeftModel, get_peft_model

        from eullm_forge.identity import lora_target_modules

        target = a.lora_params or NGramMemory(mem_cfg).n_params()
        if (ckpt / "adapter").exists():
            model = PeftModel.from_pretrained(model, ckpt / "adapter", is_trainable=True)
        else:
            targets = lora_target_modules(model)
            rank = lora_rank_for(model, target, targets)
            model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank,
                                                     target_modules=targets,
                                                     lora_dropout=0.0, task_type="CAUSAL_LM"))
        params = [p for p in model.parameters() if p.requires_grad]
        opts = [torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)]
        trainable = sum(p.numel() for p in params)
    for i, opt in enumerate(opts):
        f = ckpt / f"optim{i}.pt"
        if f.exists():
            opt.load_state_dict(torch.load(f, map_location="cpu"))  # moved to each param's device
    print(f"[ngram] mode {a.mode}, {trainable / 1e6:.1f}M trainable parameters; "
          f"resuming at step {state['step']}" if state["step"] else
          f"[ngram] mode {a.mode}, {trainable / 1e6:.1f}M trainable parameters; fresh start",
          flush=True)

    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in opts]

    def save(step: int) -> None:
        tmp = out / "ckpt.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        if a.mode == "ngram":
            memory.save(tmp / "memory")
        else:
            model.save_pretrained(tmp / "adapter")
        for i, opt in enumerate(opts):
            torch.save(opt.state_dict(), tmp / f"optim{i}.pt")
        (tmp / "state.json").write_text(json.dumps({"step": step}))
        shutil.rmtree(ckpt, ignore_errors=True)
        os.replace(tmp, ckpt)
        print(f"[ngram] saved at step {step}", flush=True)

    model.train()
    log = open(out / "train-log.jsonl", "a", encoding="utf-8")
    done = state["step"]  # steps completed so far
    total = (len(rows) // a.batch) * a.epochs
    seen, t_log = 0, time.monotonic()
    for step, epoch, idx in batches(len(rows), a.batch, a.seed, state["step"], a.epochs):
        if a.max_steps and step >= a.max_steps:
            break
        if a.stop_after and time.monotonic() - t0 > a.stop_after:
            save(step)
            log.close()
            print(f"[ngram] stop-after reached at step {step} of {total}", flush=True)
            return 0
        warm = min(1.0, (step + 1) / max(1, a.warmup))
        for opt, lrs in zip(opts, base_lrs):
            for g, lr in zip(opt.param_groups, lrs):
                g["lr"] = lr * warm
        batch = torch.from_numpy(rows[idx]).long().to(dev)
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        done = step + 1
        seen += batch.numel()
        if (step + 1) % a.log_steps == 0:
            dt = time.monotonic() - t_log
            rec = {"step": step + 1, "epoch": epoch, "loss": round(loss.item(), 4),
                   "tok_s": round(seen / dt), "of": total}
            if a.mode == "ngram":
                rec["scale"] = round(memory.scale.item(), 5)
            if torch.cuda.is_available():
                rec["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            print("[ngram] " + json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            seen, t_log = 0, time.monotonic()
        if (step + 1) % a.save_steps == 0:
            save(step + 1)
    log.close()
    final = done
    if final < total:  # stopped by --max-steps: a later run carries on from here
        save(final)
        return 0
    if a.mode == "ngram":
        memory.save(out / "memory")
    else:
        model.save_pretrained(out / "adapter")
    (out / "DONE").write_text(json.dumps({"steps": final, "tokens": final * a.batch * a.seq_len}))
    print(f"[ngram] finished: {final} steps, written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
