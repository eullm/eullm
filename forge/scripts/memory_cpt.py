#!/usr/bin/env python3
"""E1-mini: the articles trained into the backbone, continued pretraining through LoRA.

    # once, one process: tokenize, mix and pack every epoch, to disk
    python forge/scripts/memory_cpt.py prepare --model <merged> \\
        --corpus memory/codes/program.jsonl memory/codes/rewrites*.jsonl \\
        --replay openbook/pairs.jsonl premise/sft-steps.jsonl \\
        --general memory/general-it.jsonl --out runs/memory/e1-g9
    # then on the node's GPUs
    accelerate launch --num_processes 4 forge/scripts/memory_cpt.py train \\
        --model <merged> --out runs/memory/e1-g9

The legal-it roadmap, phase 2 (``docs/research/legal-it-roadmap.md``): LoRA on
the memory corpus of the nine codes, a replay of the post-training in every
batch (20-30% of the tokens) and 20% of general Italian, 3-5 epochs. The
corpus and the general text are packed into blocks of ``--block`` tokens,
every token trained on; a replay pair is trained as in post-training, its
prompt masked (`eullm_forge.identity.build_sft_features`), so the model keeps
answering as it was taught. The mix is `eullm_forge.datasets.memory_mix`.

Measured afterwards on the closed-book suite (sbatch_product_eval.slurm
PE_CLOSED=1) against the backbone: it goes on only with +15 points on what a
model knows rather than denies (real articles recognised, number and heading),
and no significant loss open book or on the traps.

``train --max-steps 30`` is a smoke test: it trains apart, saves nothing,
and prints the seconds a step and what the whole run will take.

Resumable: ``train`` continues from the last checkpoint in ``--out``, and
stops at once when ``--out/adapter`` exists. Prints counts only, never a text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.memory_mix import (  # noqa: E402
    corpus_texts,
    epoch_examples,
    general_texts,
    replay_pairs,
    tokenize_texts,
)


def prepare(args) -> int:
    from datasets import Dataset
    from transformers import AutoTokenizer

    from eullm_forge.identity import build_sft_features, ensure_chat_template

    data = args.out / "data"
    if (data / "dataset_info.json").is_file():
        print(f"[memory-cpt] {data} prepared already", flush=True)
        return 0
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(args.model)
    ensure_chat_template(tok)
    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit("[memory-cpt] the tokenizer has no end-of-text token")

    def encode(batch):
        return tok(batch, add_special_tokens=False)["input_ids"]

    corpus = tokenize_texts(corpus_texts(args.corpus), encode, args.block)
    general = (tokenize_texts(general_texts(args.general), encode, args.block)
               if args.general else [])
    pairs = replay_pairs(args.replay) if args.replay else []
    replay = build_sft_features(pairs, tok, max_length=args.replay_max_length) if pairs else []
    if not corpus:
        raise SystemExit("[memory-cpt] no corpus text")
    print(f"[memory-cpt] corpus {len(corpus):,} texts, "
          f"{sum(map(len, corpus)) / 1e6:.1f} M tokens; replay {len(replay):,} pairs; "
          f"general {len(general):,} texts, {sum(map(len, general)) / 1e6:.1f} M tokens "
          f"({time.monotonic() - t0:.0f} s)", flush=True)

    rows, totals = [], {}
    for epoch in range(args.epochs):
        ex, counts = epoch_examples(
            corpus, replay, [len(r["input_ids"]) for r in replay], general,
            replay_share=args.replay_share if replay else 0.0,
            general_share=args.general_share if general else 0.0,
            eos_id=eos, block=args.block, seed=args.seed, epoch=epoch)
        rows += ex
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        tokens = counts["corpus"] + counts["general"] + counts["replay"]
        print(f"[memory-cpt] epoch {epoch + 1}: {len(ex):,} examples, {tokens / 1e6:.1f} M tokens "
              f"(corpus {counts['corpus'] / tokens:.0%}, replay {counts['replay'] / tokens:.0%}, "
              f"general {counts['general'] / tokens:.0%})", flush=True)
    data.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(data))
    (args.out / "mix.json").write_text(json.dumps(
        {"epochs": args.epochs, "block": args.block, "seed": args.seed,
         "replay_share": args.replay_share, "general_share": args.general_share,
         "corpus": [str(p) for p in args.corpus], "replay": [str(p) for p in args.replay],
         "general": [str(p) for p in args.general], "tokens": totals}, indent=2) + "\n")
    print(f"[memory-cpt] {len(rows):,} examples -> {data} ({time.monotonic() - t0:.0f} s)",
          flush=True)
    return 0


def train(args) -> int:
    smoke = args.max_steps > 0
    if not smoke and (args.out / "adapter" / "adapter_config.json").is_file():
        print(f"[memory-cpt] {args.out / 'adapter'} exists: trained already", flush=True)
        return 0
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    from eullm_forge.identity import (
        collate_sft,
        ensure_chat_template,
        load_text_model,
        lora_target_modules,
    )

    data = load_from_disk(str(args.out / "data"))
    tok = AutoTokenizer.from_pretrained(args.model)
    ensure_chat_template(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    cuda = torch.cuda.is_available()
    model = load_text_model(args.model, dtype=torch.bfloat16 if cuda else torch.float32)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
        target_modules=lora_target_modules(model), task_type="CAUSAL_LM"))
    model.enable_input_require_grads()

    config = TrainingArguments(
        # A smoke test writes apart and saves nothing: the run proper starts clean.
        output_dir=str(args.out / "smoke" if smoke else args.out),
        num_train_epochs=1,                 # the epochs are in the prepared data
        max_steps=args.max_steps if smoke else -1,
        save_strategy="no" if smoke else "steps",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.02,                  # a float below 1: a share of the steps
        bf16=cuda,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        use_cpu=not cuda,
    )
    trainer = Trainer(model=model, args=config, train_dataset=data,
                      data_collator=lambda b: collate_sft(b, pad))
    last = get_last_checkpoint(str(args.out)) if args.out.is_dir() and not smoke else None
    if trainer.is_world_process_zero():
        print(f"[memory-cpt] {len(data):,} examples, LoRA r{args.rank}, lr {args.lr}, "
              f"{trainer.args.world_size} GPUs x {args.grad_accum} accumulated"
              + (f", resuming from {last}" if last else "")
              + (f", SMOKE TEST of {args.max_steps} steps" if smoke else ""), flush=True)
    result = trainer.train(resume_from_checkpoint=last)
    if smoke:
        if trainer.is_world_process_zero():
            secs = result.metrics.get("train_runtime", 0.0)
            per_step = args.grad_accum * trainer.args.world_size     # examples a step
            print(f"[memory-cpt] smoke test: {args.max_steps} steps in {secs:.0f} s, "
                  f"{secs / args.max_steps:.1f} s a step, loss {result.training_loss:.3f}; "
                  f"the whole run ~{secs / args.max_steps * len(data) / per_step / 3600:.1f} h",
                  flush=True)
        return 0
    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(str(args.out / "adapter"))
        tok.save_pretrained(str(args.out / "adapter"))
        print(f"[memory-cpt] adapter {args.out / 'adapter'}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="tokenize, mix and pack every epoch")
    p.add_argument("--model", required=True, help="the backbone (its tokenizer)")
    p.add_argument("--corpus", nargs="+", type=Path, required=True,
                   help="memory corpus: program.jsonl and the rewrites")
    p.add_argument("--replay", nargs="*", type=Path, default=[],
                   help="post-training pairs (instruction, output)")
    p.add_argument("--general", nargs="*", type=Path, default=[],
                   help="general Italian text (prefetch_general_it.py)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--replay-share", type=float, default=0.25)
    p.add_argument("--general-share", type=float, default=0.20)
    p.add_argument("--block", type=int, default=2048, help="tokens a packed block")
    p.add_argument("--replay-max-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train", help="LoRA on the prepared data (under accelerate)")
    t.add_argument("--model", required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--rank", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--grad-accum", type=int, default=8)
    t.add_argument("--save-steps", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-steps", type=int, default=0,
                   help="a smoke test: this many steps, nothing saved, the time printed")
    args = ap.parse_args(argv)
    return prepare(args) if args.cmd == "prepare" else train(args)


if __name__ == "__main__":
    sys.exit(main())
