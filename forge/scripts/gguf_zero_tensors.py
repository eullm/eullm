#!/usr/bin/env python3
"""A GGUF with some tensors set to zero, the rest of it untouched: an ablation.

    # which tensors there are (names, types, sizes), by a regex on the name
    python forge/scripts/gguf_zero_tensors.py --gguf <dir or .gguf> --match 'ple|ngram' --list
    # the copy: matching tensors zeroed, every other file a symlink
    python forge/scripts/gguf_zero_tensors.py --gguf <dir> --match '<regex>' --out <dir>

Written to ask Qwen3.8-Flash-Next its closed book with its n-gram (PLE)
tables switched off on Leonardo, where its BF16 weights do not fit one node
but its GGUF does (`docs/research/flash-next-ngram.md`). The PLE layer's
output is its gated value plus a convolution of it; with the value
projection at zero both are zero, so zeroing that one small tensor per PLE
layer switches the layer off exactly (`legal_eval.py --ple-off` does the
same on the BF16 model).

A split GGUF is a directory of parts: only a part holding a matching tensor
is copied (and its bytes overwritten), the others are linked, so the copy
costs the size of those parts. Zero bytes decode to zero in every ggml type
(each block's scale is zero), so the tensor is zero whatever its
quantisation. Refuses an output directory that exists.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path


def _gguf():
    try:
        import gguf
    except ImportError:  # the llama.cpp checkout's copy
        lcpp = os.environ.get("LCPP_DIR", os.path.expandvars("$WORK/llama.cpp"))
        sys.path.insert(0, str(Path(lcpp) / "gguf-py"))
        import gguf
    return gguf


def parts_of(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob("*.gguf") if not p.name.startswith("mmproj"))
    return [path]


def matching(path: Path, pattern: str):
    """(part, name, type, offset, n_bytes) of every tensor whose name matches."""
    gguf = _gguf()
    rx = re.compile(pattern)
    for part in parts_of(path):
        reader = gguf.GGUFReader(part)
        for t in reader.tensors:
            if rx.search(t.name):
                yield part, t.name, t.tensor_type.name, int(t.data_offset), int(t.n_bytes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", type=Path, required=True,
                    help="a .gguf, or the directory of a split one")
    ap.add_argument("--match", required=True, help="regex on tensor names")
    ap.add_argument("--out", type=Path, help="directory of the copy")
    ap.add_argument("--list", action="store_true", help="only list the matching tensors")
    a = ap.parse_args(argv)

    found = list(matching(a.gguf, a.match))
    for part, name, ttype, _, n in found:
        print(f"{part.name}  {name}  {ttype}  {n / 2**20:.1f} MiB")
    print(f"[zero] {len(found)} tensors match {a.match!r}", flush=True)
    if a.list:
        return 0
    if not found:
        print("[zero] nothing to zero: no copy written", file=sys.stderr)
        return 1
    if a.out is None:
        ap.error("--out is needed to write the copy")
    if a.out.exists():
        ap.error(f"{a.out} exists")
    a.out.mkdir(parents=True)
    touched = {part for part, *_ in found}
    for part in parts_of(a.gguf) + sorted(a.gguf.glob("mmproj*.gguf") if a.gguf.is_dir() else []):
        dst = a.out / part.name
        if part in touched:
            shutil.copyfile(part, dst)
        else:
            dst.symlink_to(part.resolve())
    chunk = b"\0" * (1 << 20)
    for part, name, _, offset, n in found:
        with open(a.out / part.name, "r+b") as f:
            f.seek(offset)
            left = n
            while left:
                k = min(left, len(chunk))
                f.write(chunk[:k])
                left -= k
    print(f"[zero] written {a.out}: {len(touched)} part(s) copied, "
          f"{len(parts_of(a.gguf)) - len(touched)} linked", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
