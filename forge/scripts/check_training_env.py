#!/usr/bin/env python3
"""Pre-flight check for the training environment.

Verifies that everything the training YAMLs assume is actually
installed and importable, before LLaMA-Factory tries to use it.
Run this once after install_training_deps.sh, and again whenever
the YAML changes (e.g. new optimizer, new attention backend).

Usage:
    python forge/scripts/check_training_env.py [--smoke]

Exit code is non-zero on the first missing piece, so it can be
chained in a script:
    python forge/scripts/check_training_env.py && bash forge/scripts/train.sh ...
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Iterable


def _print(symbol: str, color: int, *args: object) -> None:
    print(f"\033[{color}m{symbol}\033[0m  " + " ".join(str(a) for a in args))


def ok(*args: object) -> None:
    _print("✓", 32, *args)


def warn(*args: object) -> None:
    _print("!", 33, *args)


def fail(*args: object) -> None:
    _print("✗", 31, *args)


def check_module(name: str, *, min_version: str | None = None) -> bool:
    """Import a package and report. Returns True on success."""
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:
        fail(f"{name} not importable: {exc}")
        return False
    version = getattr(mod, "__version__", None) or "unknown"
    if min_version and version != "unknown":
        # naive comparison: split by '.' and lex-compare numeric tuples
        try:
            cur = tuple(int(p) for p in version.split(".")[:3] if p.isdigit())
            need = tuple(int(p) for p in min_version.split(".")[:3] if p.isdigit())
            if cur < need:
                warn(f"{name} {version} (recommended >= {min_version})")
                return True  # not a hard fail
        except ValueError:
            pass
    ok(f"{name} {version}")
    return True


def check_cuda(expect_gpus: int = 1) -> bool:
    try:
        import torch
    except ImportError:
        fail("torch not importable")
        return False
    if not torch.cuda.is_available():
        fail("CUDA not available — torch built without CUDA?")
        return False
    n = torch.cuda.device_count()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        total = torch.cuda.get_device_properties(i).total_memory
        ok(f"cuda:{i} {name} (cap {cap[0]}.{cap[1]}, "
           f"{total / 1024**3:.1f} GiB)")
    if n < expect_gpus:
        fail(f"{n} GPU(s) visible, {expect_gpus} expected "
             f"(wrong --gres request, or not inside the job?)")
        return False
    if torch.cuda.is_bf16_supported():
        ok("BF16 supported (Ampere+ class GPU)")
    else:
        warn("BF16 NOT supported — flip bf16: false / fp16: true in the YAML")
    return True


def check_llamafactory_cli() -> bool:
    import shutil
    if not shutil.which("llamafactory-cli"):
        fail("llamafactory-cli not on PATH — run install_training_deps.sh")
        return False
    ok("llamafactory-cli on PATH")
    return True


def check_dataset(data_dir: Path) -> bool:
    if not data_dir.is_dir():
        fail(f"dataset dir not found: {data_dir}")
        return False
    train = data_dir / "train.jsonl"
    val = data_dir / "val.jsonl"
    info = data_dir / "dataset_info.json"
    missing: list[Path] = [p for p in (train, val) if not p.is_file()]
    if missing:
        fail(f"missing files: {', '.join(str(p) for p in missing)}")
        return False
    train_size = train.stat().st_size / 1024**2
    val_size = val.stat().st_size / 1024**2
    ok(f"dataset at {data_dir}: "
       f"train={train_size:.0f} MiB, val={val_size:.0f} MiB")
    if not info.is_file():
        warn("dataset_info.json not present yet "
             "(train.sh will create it on launch)")
    return True


def check_tokenizer(model_id: str, *, offline: bool = False) -> bool:
    """Smoke-load the tokenizer for the configured model. Confirms that
    transformers + huggingface_hub auth are working without committing
    to a full model download. With ``offline`` it must come from the
    local HF cache — this is what proves prefetch_models.py ran before
    submitting to network-less compute nodes."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        fail("transformers not importable")
        return False
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=False, local_files_only=offline,
        )
    except Exception as exc:
        if offline:
            fail(f"tokenizer for {model_id} not in the local HF cache "
                 f"(HF_HOME set? prefetch_models.py run?): {exc}")
        else:
            fail(f"could not load tokenizer for {model_id}: {exc}")
        return False
    src = "local cache" if offline else "HF Hub"
    ok(f"tokenizer for {model_id} (vocab {len(tok)}, from {src})")
    return True


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "italgiure_corpus" / "pretraining",
        help="Directory expected to contain train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only check the smoke-test deps (skip 96 GB-tier checks)",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip the HF tokenizer fetch (useful offline / CI)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Assume no network (HPC compute nodes): force the HF stack "
             "offline and require the tokenizer in the local cache",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip the GPU/CUDA checks (login nodes have no GPU)",
    )
    parser.add_argument(
        "--expect-gpus",
        type=int,
        default=1,
        help="Fail if fewer CUDA devices are visible (multi-GPU jobs)",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Also check the multi-GPU stack (deepspeed) is importable",
    )
    args = parser.parse_args(argv)

    if args.offline:
        # Must happen before transformers/huggingface_hub are imported.
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("== Python ==")
    ok(f"Python {sys.version.split()[0]} at {sys.executable}")

    print()
    print("== Required Python packages ==")
    pkgs = [
        ("torch", "2.1"),
        ("transformers", "4.45"),
        ("peft", "0.10"),
        ("accelerate", "0.30"),
        ("datasets", "2.20"),
        ("safetensors", None),
        ("llamafactory", None),
    ]
    if args.multi_gpu:
        pkgs.append(("deepspeed", "0.19"))
    failed = False
    for name, min_v in pkgs:
        if not check_module(name, min_version=min_v):
            failed = True

    print()
    print("== CLI tools ==")
    if not check_llamafactory_cli():
        failed = True

    print()
    print("== GPU / CUDA ==")
    if args.cpu_only:
        warn("GPU checks skipped (--cpu-only)")
    elif not check_cuda(expect_gpus=args.expect_gpus):
        failed = True

    print()
    print("== Dataset ==")
    if not check_dataset(args.data_dir):
        failed = True

    print()
    print("== Tokenizer (HF cache + auth check) ==")
    if not args.skip_tokenizer:
        # smoke uses Qwen3-1.7B-Base; production uses Qwen3-32B-Base.
        # Tokenizer is the same family so the smaller download is fine
        # for both.
        if not check_tokenizer("Qwen/Qwen3-1.7B-Base", offline=args.offline):
            failed = True
    else:
        warn("tokenizer check skipped (--skip-tokenizer)")

    print()
    if failed:
        fail("pre-flight FAILED — fix the items above before training")
        return 1
    ok("pre-flight passed — ready to train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
