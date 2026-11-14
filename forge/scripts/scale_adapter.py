#!/usr/bin/env python3
"""A LoRA adapter at a share of itself, or kept in some layers only, written as a new adapter.

    python forge/scripts/scale_adapter.py --adapter <dir> --out <dir> --scale 0.5 \\
        [--layers 0-11,24-35]

What `legal_eval.py --adapter-scale/--adapter-layers` does at examination
time, written to disk, so that a chosen point is packaged like any trained
adapter (merge into the base, GGUF, the product's exams:
`submit_grpo_round.sh` with LINKS=0 over a run directory holding it). The
merged update of a LoRA is B·A·alpha/r, linear in B: every ``lora_B`` tensor
is multiplied by ``--scale``, and set to zero outside ``--layers``. The
config and every other file are copied unchanged.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

LAYER = re.compile(r"\.layers\.(\d+)\.")


def parse_layers(spec: str) -> set[int]:
    out: set[int] = set()
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--layers", default="", help="keep the LoRA in these layers only")
    a = ap.parse_args(argv)

    from safetensors.torch import load_file, save_file

    src = a.adapter / "adapter_model.safetensors"
    if not src.is_file():
        ap.error(f"no adapter_model.safetensors in {a.adapter}")
    if a.out.exists():
        ap.error(f"{a.out} exists")
    keep = parse_layers(a.layers) if a.layers else None
    tensors = load_file(str(src))
    scaled = dropped = 0
    for name, t in tensors.items():
        if "lora_B" not in name:
            continue
        m = LAYER.search(name)
        if keep is not None and not (m and int(m.group(1)) in keep):
            tensors[name] = t.zero_()
            dropped += 1
        elif a.scale != 1.0:
            tensors[name] = t * a.scale
            scaled += 1
    a.out.mkdir(parents=True)
    for f in a.adapter.iterdir():
        if f.is_file() and f.name != src.name:
            shutil.copy2(f, a.out / f.name)
    save_file(tensors, str(a.out / src.name), metadata={"format": "pt"})
    print(f"[scale] {a.out}: {scaled} lora_B scaled by {a.scale:g}, {dropped} dropped "
          f"outside layers {a.layers or 'all'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
