#!/usr/bin/env python3
"""What a job's server logs say that its results do not: the expert cache's
statistics and llama.cpp's VRAM pool and errors, a line per server where
there is something to say and a count over all of them.

    python3 tools/lumi/log_checks.py --queue $CAMPAIGN_DIR JOB

- The expert cache, with LLAMA_MOE_CACHE_STATS=N (patch 0002): every N steps
  of up to 8 tokens a line with the MiB copied a step and the share of the
  experts found in VRAM, and the cache's totals when the server stops, apart
  for batches of up to 8 tokens and larger ones (at 32 requests a decode step
  is larger, so only the totals speak for it).
- The pool (patch 0011): "pool[N]: ... freeing ..." each time it gives cached
  buffers back to keep VRAM free; ROCm errors ("HSA_STATUS_ERROR..."), "out
  of memory", and NaN, which a soak must not show (docs/lumi/allocation-plan.md,
  the HIP graphs of October 2026).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

WINDOW = re.compile(r"llama_moe_cache: \d+ steps of ([\d.]+) tokens.*?; ([\d.]+) MiB/step copied, "
                    r"([\d.]+)% of the experts in VRAM")
TOTALS = re.compile(r"llama_moe_cache: (ubatch\s*[<>]=?\s*8): hits = \d+, misses = \d+, "
                    r"hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB")
POOL = re.compile(r"pool\[\d+\]: [\d.]+ MiB free, freeing")
HSA = re.compile(r"HSA_STATUS_ERROR\w*")
OOM = re.compile(r"out of memory", re.I)
NAN = re.compile(r"\bnan\b", re.I)


def read_log(path: str) -> dict:
    """The counts and the cache's figures of one server log."""
    out = {"windows": [], "totals": {}, "pool": 0, "hsa": {}, "oom": 0, "nan": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = WINDOW.search(line)
            if m:
                out["windows"].append(tuple(float(g) for g in m.groups()))
            m = TOTALS.search(line)
            if m:
                kind = re.sub(r"\s+", " ", m.group(1))
                out["totals"][kind] = (float(m.group(2)), float(m.group(3)))
            if POOL.search(line):
                out["pool"] += 1
            for h in HSA.findall(line):
                out["hsa"][h] = out["hsa"].get(h, 0) + 1
            out["oom"] += bool(OOM.search(line))
            out["nan"] += bool(NAN.search(line))
    return out


def server_name(path: str) -> str:
    name = os.path.basename(path)
    return name[:-len(".log")].replace(".server", " server")


def describe(name: str, r: dict) -> list:
    lines = []
    if r["windows"] or r["totals"]:
        parts = []
        if r["windows"]:
            vram = [w[2] for w in r["windows"]]
            tokens = sum(w[0] for w in r["windows"]) / len(vram)
            mib = sum(w[1] for w in r["windows"]) / len(vram)
            parts.append(f"{len(vram)} windows of {tokens:.1f} tokens, "
                         f"{sum(vram) / len(vram):.1f}% of the experts in VRAM "
                         f"({min(vram):.1f}-{max(vram):.1f}), {mib:.1f} MiB/step copied")
        for k, (rate, mib) in sorted(r["totals"].items()):
            parts.append(f"at exit {k}: {rate:.2f}% hit, {mib:,.0f} MiB copied")
        lines.append(f"cache {name}: " + "; ".join(parts))
    trouble = []
    if r["hsa"]:
        trouble.append(", ".join(f"{k} x{v}" for k, v in sorted(r["hsa"].items())))
    if r["oom"]:
        trouble.append(f"out of memory x{r['oom']}")
    if r["nan"]:
        trouble.append(f"NaN x{r['nan']}")
    if r["pool"] or trouble:
        lines.append(f"pool {name}: {r['pool']} flushes (pool[...] freeing)"
                     + ("; " + "; ".join(trouble) if trouble else ""))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queue", required=True)
    ap.add_argument("job")
    args = ap.parse_args(argv)
    paths = sorted(glob.glob(os.path.join(args.queue, "results", "logs", args.job, "*.log")))
    if not paths:
        print(f"no server logs for job {args.job}")
        return 0
    totals = {"pool": 0, "hsa": 0, "oom": 0, "nan": 0}
    for path in paths:
        r = read_log(path)
        for line in describe(server_name(path), r):
            print(line)
        totals["pool"] += r["pool"]
        totals["hsa"] += sum(r["hsa"].values())
        totals["oom"] += r["oom"]
        totals["nan"] += r["nan"]
    print(f"{len(paths)} server logs: {totals['pool']} pool flushes, {totals['hsa']} HSA errors, "
          f"{totals['oom']} out of memory, {totals['nan']} NaN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
