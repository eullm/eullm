#!/usr/bin/env python3
"""Phase 2 — Knowledge distillation Qwen3-32B-legal-it → Qwen3-7B-Base.

A frozen teacher (the Phase-1 LoRA-tuned 32B) generates a probability
distribution over each next token; the student (7B) is trained to
imitate that distribution via KL divergence (soft labels), plus a
fraction of the standard cross-entropy on the ground-truth tokens
(hard labels).

Loss = α · KL(student || teacher) · T² + (1-α) · CE(student, y)

Training is single-GPU (no FSDP / DeepSpeed) targeting one ~96 GB
device. Memory layout (BF16 throughout):

    Teacher 32B (frozen, no grad):     ~64 GB
    Student 7B + grad + 8-bit Adam:    ~28 GB
    Activations (seq 2048, both nets): ~6 GB
    Headroom:                          ~variable

If the 32B teacher does not fit, pass --teacher-load-in-8bit (bitsandbytes
NF4/INT8) to drop the teacher to ~16 GB at the cost of slightly noisier
logits.

The script is checkpoint-resumable: pass --resume-from <dir> or just
re-run with the same --output-dir and the latest checkpoint inside it
will be picked up automatically.

Usage:
    python forge/scripts/distill.py \\
        --config forge/training/configs/distill_qwen3_32b_to_7b.yaml

Or with explicit args (overrides YAML if both are present):
    python forge/scripts/distill.py \\
        --teacher-model Qwen/Qwen3-32B-Base \\
        --teacher-adapter ~/checkpoints/qwen3_32b_legal_it_continued_pt \\
        --student-model Qwen/Qwen3-7B-Base \\
        --dataset-dir ~/datasets/legal_it \\
        --output-dir ~/checkpoints/qwen3_7b_legal_it_distilled \\
        ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DistillConfig:
    # Models
    teacher_model: str = "Qwen/Qwen3-32B-Base"
    teacher_adapter: Optional[str] = None  # Phase-1 LoRA adapter dir
    student_model: str = "Qwen/Qwen3-7B-Base"
    teacher_load_in_8bit: bool = False     # bitsandbytes 8-bit teacher
    teacher_load_in_4bit: bool = False     # bitsandbytes 4-bit teacher (NF4)

    # Data
    dataset_dir: str = "~/datasets/legal_it"
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    cutoff_len: int = 2048
    max_train_samples: Optional[int] = None  # None = all

    # Training
    output_dir: str = "~/checkpoints/qwen3_7b_legal_it_distilled"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    max_steps: int = -1                    # -1 = unlimited (use epochs)
    warmup_steps: int = 1000
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 20

    # Distillation
    kl_temperature: float = 2.0
    kl_alpha: float = 0.7                  # weight of soft (KL) loss
    # ce_alpha is implicitly (1 - kl_alpha)

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    # Resume
    resume_from: Optional[str] = None  # path to a checkpoint dir; auto if None


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _parse_args() -> DistillConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config path (CLI args override YAML).")
    # Allow every DistillConfig field as a CLI arg.
    for f in DistillConfig.__dataclass_fields__.values():
        flag = "--" + f.name.replace("_", "-")
        kw = {"default": None}
        if f.type is bool:
            kw["action"] = argparse.BooleanOptionalAction
        elif f.type is int:
            kw["type"] = int
        elif f.type is float:
            kw["type"] = float
        else:
            kw["type"] = str
        parser.add_argument(flag, **kw)
    raw = parser.parse_args()

    cfg_dict: dict = {}
    if raw.config:
        with open(raw.config) as f:
            cfg_dict = yaml.safe_load(f) or {}
    for f in DistillConfig.__dataclass_fields__.values():
        v = getattr(raw, f.name)
        if v is not None:
            cfg_dict[f.name] = v

    cfg = DistillConfig(**cfg_dict)
    cfg.dataset_dir = os.path.expanduser(cfg.dataset_dir)
    cfg.output_dir = os.path.expanduser(cfg.output_dir)
    if cfg.teacher_adapter:
        cfg.teacher_adapter = os.path.expanduser(cfg.teacher_adapter)
    if cfg.resume_from:
        cfg.resume_from = os.path.expanduser(cfg.resume_from)
    return cfg


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _quantization_config(load_in_8bit: bool, load_in_4bit: bool):
    if not (load_in_8bit or load_in_4bit):
        return None
    from transformers import BitsAndBytesConfig
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_teacher(cfg: DistillConfig, dtype: torch.dtype, device: str):
    print(f"[teacher] loading {cfg.teacher_model} "
          f"(8bit={cfg.teacher_load_in_8bit}, 4bit={cfg.teacher_load_in_4bit})",
          file=sys.stderr)
    quant = _quantization_config(cfg.teacher_load_in_8bit,
                                 cfg.teacher_load_in_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model,
        torch_dtype=dtype,
        device_map={"": device} if quant is None else "auto",
        quantization_config=quant,
    )
    if cfg.teacher_adapter:
        print(f"[teacher] applying adapter {cfg.teacher_adapter}",
              file=sys.stderr)
        model = PeftModel.from_pretrained(model, cfg.teacher_adapter)
        model = model.merge_and_unload()    # collapse LoRA into base weights
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_student(cfg: DistillConfig, dtype: torch.dtype, device: str):
    print(f"[student] loading {cfg.student_model}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.student_model,
        torch_dtype=dtype,
        device_map={"": device},
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    kl_alpha: float,
    kl_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the distillation loss.

    Args:
        student_logits: (B, T, V) student logits.
        teacher_logits: (B, T, V) teacher logits, detached.
        labels: (B, T) ground-truth next-token ids; -100 = ignore.
        kl_alpha: weight of the soft (KL) term, complement is CE weight.
        kl_temperature: softmax temperature for both teacher and student
            logits before the KL.

    Returns:
        ``(total_loss, stats_dict)``.
    """
    # Shift for next-token: predict token t given tokens [0..t-1].
    s_logits = student_logits[..., :-1, :].contiguous()
    t_logits = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Mask padding tokens (label == -100) out of both losses.
    valid_mask = (shift_labels != -100)

    # --- Soft (KL) ---
    T = kl_temperature
    s_log_probs = F.log_softmax(s_logits / T, dim=-1)
    t_probs = F.softmax(t_logits / T, dim=-1)
    kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)  # (B, T-1)
    kl = (kl * valid_mask).sum() / valid_mask.sum().clamp_min(1)
    kl = kl * (T * T)   # scale per Hinton et al. 2015

    # --- Hard (CE) ---
    ce = F.cross_entropy(
        s_logits.view(-1, s_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

    total = kl_alpha * kl + (1.0 - kl_alpha) * ce
    return total, {"loss": total.item(), "kl": kl.item(), "ce": ce.item()}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_dataloaders(cfg: DistillConfig, tokenizer):
    data_files = {
        "train": str(Path(cfg.dataset_dir) / cfg.train_file),
        "validation": str(Path(cfg.dataset_dir) / cfg.val_file),
    }
    raw = load_dataset("json", data_files=data_files)
    if cfg.max_train_samples:
        raw["train"] = raw["train"].select(range(cfg.max_train_samples))

    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.cutoff_len,
            padding=False,
            return_attention_mask=True,
        )

    cols = raw["train"].column_names
    tokenized = raw.map(
        _tok,
        batched=True,
        remove_columns=cols,
        num_proc=4,
        desc="tokenising",
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(
        tokenized["train"], batch_size=cfg.per_device_train_batch_size,
        shuffle=True, collate_fn=collator, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        tokenized["validation"], batch_size=cfg.per_device_train_batch_size,
        shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.is_dir():
        return None
    candidates = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1])
        if p.name.split("-")[-1].isdigit() else -1,
    )
    return candidates[-1] if candidates else None


def save_checkpoint(
    student, optimizer, scheduler, scaler, step: int, output_dir: Path,
    cfg: DistillConfig,
) -> Path:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] saving to {ckpt_dir}", file=sys.stderr)
    student.save_pretrained(ckpt_dir, safe_serialization=True)
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": cfg.__dict__,
    }
    torch.save(state, ckpt_dir / "training_state.pt")
    return ckpt_dir


def load_checkpoint(ckpt_dir: Path, optimizer, scheduler, scaler) -> int:
    print(f"[resume] loading state from {ckpt_dir}", file=sys.stderr)
    state = torch.load(ckpt_dir / "training_state.pt", map_location="cpu",
                       weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return state["step"]


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def train(cfg: DistillConfig) -> None:
    torch.manual_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = "cuda:0"
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- tokenizer (shared between teacher and student) ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- data ---
    train_loader, val_loader = build_dataloaders(cfg, tokenizer)
    total_micro_steps = (
        len(train_loader) * cfg.num_train_epochs
        if cfg.max_steps < 0
        else cfg.max_steps * cfg.gradient_accumulation_steps
    )
    total_optim_steps = total_micro_steps // cfg.gradient_accumulation_steps

    # --- models ---
    teacher = load_teacher(cfg, dtype, device)
    student = load_student(cfg, dtype, device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_optim_steps,
    )
    scaler = None  # bf16 needs no scaler

    # --- resume ---
    resume_dir = (
        Path(cfg.resume_from) if cfg.resume_from
        else latest_checkpoint(output_dir)
    )
    start_step = 0
    if resume_dir:
        # Re-load student weights from the checkpoint (overrides base init).
        student = AutoModelForCausalLM.from_pretrained(
            resume_dir, torch_dtype=dtype,
        ).to(device)
        if cfg.gradient_checkpointing:
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        start_step = load_checkpoint(resume_dir, optimizer, scheduler, scaler)
        print(f"[resume] continuing from step {start_step}", file=sys.stderr)

    # --- log ---
    print(f"[info] total optim steps: {total_optim_steps:,}",
          file=sys.stderr)
    print(f"[info] saves every {cfg.save_steps} steps to {output_dir}",
          file=sys.stderr)

    student.train()
    optimizer.zero_grad()
    micro = 0
    optim_step = start_step
    t0 = time.time()
    log_loss_acc = 0.0
    log_kl_acc = 0.0
    log_ce_acc = 0.0
    log_n = 0

    for epoch in range(cfg.num_train_epochs):
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad():
                t_out = teacher(**batch)
                t_logits = t_out.logits.detach()
            s_out = student(**batch)
            loss, parts = distill_loss(
                s_out.logits, t_logits, batch["labels"],
                kl_alpha=cfg.kl_alpha, kl_temperature=cfg.kl_temperature,
            )
            (loss / cfg.gradient_accumulation_steps).backward()
            log_loss_acc += parts["loss"]
            log_kl_acc += parts["kl"]
            log_ce_acc += parts["ce"]
            log_n += 1
            micro += 1
            if micro % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(), cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                if optim_step % cfg.logging_steps == 0:
                    dt = time.time() - t0
                    avg_loss = log_loss_acc / log_n
                    avg_kl = log_kl_acc / log_n
                    avg_ce = log_ce_acc / log_n
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"[step {optim_step:>6} / {total_optim_steps}] "
                        f"loss={avg_loss:.4f}  kl={avg_kl:.4f}  ce={avg_ce:.4f}  "
                        f"lr={lr:.2e}  micro/s={micro/dt:.2f}",
                        file=sys.stderr,
                    )
                    log_loss_acc = log_kl_acc = log_ce_acc = 0.0
                    log_n = 0

                if optim_step % cfg.save_steps == 0:
                    save_checkpoint(student, optimizer, scheduler, scaler,
                                    optim_step, output_dir, cfg)

                if cfg.max_steps > 0 and optim_step >= cfg.max_steps:
                    break

        if cfg.max_steps > 0 and optim_step >= cfg.max_steps:
            break

    # Final save.
    save_checkpoint(student, optimizer, scheduler, scaler, optim_step,
                    output_dir, cfg)
    # Save tokenizer too — needed for inference / GGUF export later.
    tokenizer.save_pretrained(output_dir)
    print(f"[done] final student saved at {output_dir}", file=sys.stderr)


def main() -> int:
    cfg = _parse_args()
    print("[config]", json.dumps(cfg.__dict__, indent=2, default=str),
          file=sys.stderr)
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
