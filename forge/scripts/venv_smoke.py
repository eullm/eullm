#!/usr/bin/env python3
"""Does a freshly built Forge venv do, on a GPU, what the pipeline asks of it?

Written for the venv of current releases on Leonardo (setup_venv_latest.sh),
before it replaces the Playground's. Each check runs whatever the others did,
prints PASS or FAIL with the reason, and the last line is SMOKE PASSED only
when every required check passed. Models are read from the local caches
only (compute nodes have no internet).

Checks:
  gpu        driver, CUDA build, device, a bf16 matmul and SDPA
  bnb        bitsandbytes 8-bit layer on the GPU
  load       a small instruct model in bf16, eight greedy tokens
  load8      the same model quantized to 8 bits through transformers
  lora       peft LoRA: a few optimiser steps lower the loss; save and reload
  trl        GRPOTrainer imports (only with --trl-path)
  qwen35     a Qwen3.5 forward pass, with the fla kernels if they load
  moe        the 30B-A3B MoE in 8 bits on one GPU, one forward (line A)

Usage:
    python venv_smoke.py --small Qwen/Qwen3-4B-Instruct-2507 \\
        --qwen35 /path/to/Qwen3.5-4B --moe Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        [--trl-path DIR] [--skip moe,qwen35]
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import tempfile
import time
import traceback

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, required: bool = True):
    """Run the decorated function as one check and record its outcome."""
    def wrap(fn):
        def run(*args, **kwargs):
            t0 = time.time()
            try:
                note = fn(*args, **kwargs) or ""
                RESULTS.append((name, True, note))
                print(f"[smoke] PASS {name} ({time.time() - t0:.0f} s) {note}", flush=True)
            except Exception as exc:  # every failure is reported, none stops the rest
                RESULTS.append((name, not required, f"{type(exc).__name__}: {exc}"))
                print(f"[smoke] FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
        return run
    return wrap


@check("gpu")
def gpu() -> str:
    import torch
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    # Warm up first: the first products pay cuBLAS's start-up, and timed
    # with them an A100 read 4 TFLOP/s (job 63409258) against its ~300.
    for _ in range(3):
        a @ a
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    tflops = 20 * 2 * 8192**3 / (time.time() - t0) / 1e12
    q = torch.randn(4, 16, 1024, 128, device="cuda", dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    return (f"driver/gpu: {smi}; torch {torch.__version__} (CUDA {torch.version.cuda}); "
            f"bf16 matmul {tflops:.0f} TFLOP/s")


@check("bnb")
def bnb() -> str:
    import bitsandbytes as bnb_mod
    import torch
    layer = bnb_mod.nn.Linear8bitLt(1024, 1024, has_fp16_weights=False).cuda()
    out = layer(torch.randn(2, 1024, device="cuda", dtype=torch.float16))
    assert torch.isfinite(out).all(), "non-finite output"
    return f"bitsandbytes {bnb_mod.__version__}"


def _generate(model, tok) -> str:
    import torch
    msgs = [{"role": "user", "content": "Che cos'è il codice civile? Rispondi in una frase."}]
    # return_dict=True explicitly: transformers 5 changed what the call
    # returns by default, and the smoke must work on either side of it.
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)
    ids = enc["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    assert text.strip(), "empty generation"
    return repr(text)


@check("load")
def load(small: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(small)
    model = AutoModelForCausalLM.from_pretrained(small, dtype=torch.bfloat16, device_map="cuda")
    return _generate(model, tok)


@check("load8")
def load8(small: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(small)
    model = AutoModelForCausalLM.from_pretrained(
        small, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="cuda")
    return _generate(model, tok)


@check("lora")
def lora(small: str) -> str:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(small)
    base = AutoModelForCausalLM.from_pretrained(small, dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                                            task_type="CAUSAL_LM"))
    text = ("Art. 1453 c.c. Nei contratti con prestazioni corrispettive, quando uno dei "
            "contraenti non adempie le sue obbligazioni, l'altro può a sua scelta chiedere "
            "l'adempimento o la risoluzione del contratto.")
    batch = tok([text] * 4, return_tensors="pt").to("cuda")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    for _ in range(6):
        loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not go down: {losses}"
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        PeftModel.from_pretrained(base, d)
    return f"loss {losses[0]:.3f} -> {losses[-1]:.3f}, adapter saved and reloaded"


@check("trl", required=False)
def trl_import(path: str) -> str:
    sys.path.insert(0, path)
    import trl
    from trl import GRPOConfig, GRPOTrainer  # noqa: F401
    return f"trl {trl.__version__} from {path}"


@check("qwen35")
def qwen35(path: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        import fla  # noqa: F401
        kernels = "fla kernels importable"
    except Exception as exc:
        kernels = f"fla NOT importable ({type(exc).__name__}): reference path, slow"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    batch = tok(["Il termine per l'appello civile è"] * 2, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model(**batch)
    torch.cuda.synchronize()
    first = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        out = model(**batch)
    torch.cuda.synchronize()
    assert torch.isfinite(out.logits).all(), "non-finite logits"
    # The first forward compiles the Triton kernels; the second is the speed.
    return f"{kernels}; first forward {first:.2f} s, second {time.time() - t0:.2f} s"


@check("moe")
def moe(path: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="cuda")
    loaded = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30
    # What 8 bits actually reached. bitsandbytes quantizes nn.Linear layers;
    # experts stored as fused 3-D tensors stay in bf16, and then the "8-bit"
    # 30B needs ~57 GiB (job 63409258) instead of the ~31 of September.
    by_dtype: dict[str, int] = {}
    for prm in model.parameters():
        key = str(prm.dtype).replace("torch.", "")
        by_dtype[key] = by_dtype.get(key, 0) + prm.numel() * prm.element_size()
    for buf in model.buffers():
        key = str(buf.dtype).replace("torch.", "")
        by_dtype[key] = by_dtype.get(key, 0) + buf.numel() * buf.element_size()
    split = ", ".join(f"{k} {v / 2**30:.1f} GiB" for k, v in sorted(by_dtype.items()))
    return (f"loaded in {loaded:.0f} s, peak {mem:.1f} GiB on the GPU; weights by dtype: "
            f"{split}; {_generate(model, tok)}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--small", required=True,
                   help="small instruct model (HF id in the cache, or a path)")
    p.add_argument("--qwen35", help="a Qwen3.5 checkpoint (path or id)")
    p.add_argument("--moe", help="the 30B-A3B MoE (path or id)")
    p.add_argument("--trl-path", help="directory with TRL (setup_grpo_libs.sh)")
    p.add_argument("--skip", default="", help="comma-separated checks to skip")
    a = p.parse_args(argv)
    skip = set(filter(None, a.skip.split(",")))
    print(f"[smoke] python {sys.version.split()[0]}, venv {sys.prefix}, "
          f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}", flush=True)

    import torch

    def free():
        gc.collect()
        torch.cuda.empty_cache()

    gpu()
    if "bnb" not in skip:
        bnb()
    for name, fn in (("load", load), ("load8", load8), ("lora", lora)):
        if name not in skip:
            fn(a.small)
            free()
    if a.trl_path and "trl" not in skip:
        trl_import(a.trl_path)
    if a.qwen35 and "qwen35" not in skip:
        qwen35(a.qwen35)
        free()
    if a.moe and "moe" not in skip:
        moe(a.moe)
        free()

    failed = [n for n, passed, _ in RESULTS if not passed]
    print("\n[smoke] summary")
    for n, passed, note in RESULTS:
        print(f"  {'PASS' if passed else 'FAIL'} {n:<8} {note}")
    print("SMOKE PASSED" if not failed else f"SMOKE FAILED: {' '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
