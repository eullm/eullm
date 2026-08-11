#!/usr/bin/env python3
"""Measure achievable DRAM read/copy bandwidth, multi-threaded, no root.

Why this exists: on a CPU-only board, LLM *decode* is memory-bandwidth-bound,
not compute-bound — every generated token streams the active weight set
through the cores once, so tok/s is essentially `bandwidth / bytes_per_token`.
Measurements on a Radxa Orion O6 (docs/arm-cix-p1-cpu-profile.md § 9.2) put
two different models at 36-39 GB/s of effective bandwidth, converging within
9% of each other despite different sizes and architectures. That number is
the memory wall.

Knowing the board's *achievable* peak turns that from an observation into a
decision. If the hardware gives ~45 GB/s, the CPU path is already extracting
most of it and there is nothing left for an integrated GPU (which shares the
same DRAM) to win on decode — only prefill, which is compute-bound, is worth
accelerating. If it gives 70+ GB/s, the gap is real and a Vulkan/OpenCL
backend is worth building and testing.

Deliberately dependency-light: numpy only, installable into a plain venv
without root (`python -m venv ~/.venv-bench && ~/.venv-bench/bin/pip install
numpy`). Nothing here needs privileged counters or a kernel module.

Method: one private array per thread, each far larger than any cache, touched
once before timing so every page is resident and faulted in. Each thread then
runs a read-only reduction (READ) or an out-of-place copy (COPY) over its own
buffer, repeatedly. numpy releases the GIL inside these ufunc loops, so the
threads genuinely run in parallel and the aggregate is what the memory
subsystem delivered. Single-threaded numbers are not useful here: one core
cannot saturate a modern memory controller, so a 1-thread result understates
the board badly.

COPY moves twice the bytes it reads (one read + one write stream), and both
are counted, which is the STREAM convention.

Usage:
    ~/.venv-bench/bin/pip install numpy
    ~/.venv-bench/bin/python bench/mem_bandwidth.py
    ~/.venv-bench/bin/python bench/mem_bandwidth.py --threads 4 --size-mb 1024
"""

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
except ImportError:
    print(
        "This script requires numpy. Into an existing venv, no root needed:\n"
        "    ~/.venv-bench/bin/pip install numpy",
        file=sys.stderr,
    )
    sys.exit(1)

GIB = 1024**3
MIB = 1024**2


def run_read(buf: "np.ndarray", repeats: int) -> None:
    """Read-only pass: the analogue of streaming weights during decode.

    `max` rather than `sum` on purpose. The reduction has to cost as close to
    nothing as possible per element, or the kernel measures the CPU instead of
    the memory: on the machine this was written on, the same buffer read at
    7.8 GB/s with `sum(dtype=float64)` (every float32 upconverted), 17.7 GB/s
    with a float32 accumulator, and 28.3 GB/s with `max` — a 3.6x spread over
    identical memory traffic. `max` is one SIMD compare per element and no
    accumulator width to widen.
    """
    for _ in range(repeats):
        # numpy releases the GIL inside the reduction loop, so concurrent
        # threads genuinely overlap rather than taking turns.
        buf.max()


def run_copy(src: "np.ndarray", dst: "np.ndarray", repeats: int) -> None:
    """Read + write pass, the STREAM `copy` kernel."""
    for _ in range(repeats):
        np.copyto(dst, src)


def measure(kind: str, threads: int, size_bytes: int, repeats: int, rounds: int):
    """Return per-round aggregate GB/s for `kind` in {"read", "copy"}."""
    n_elems = size_bytes // 4  # float32
    # Allocate and fault in every page before timing: a first-touch page fault
    # storm inside the timed region would be measured as bandwidth it is not.
    srcs = [np.ones(n_elems, dtype=np.float32) for _ in range(threads)]
    dsts = [np.empty(n_elems, dtype=np.float32) for _ in range(threads)] if kind == "copy" else []
    for i, s in enumerate(srcs):
        s[:] = 1.0
        if kind == "copy":
            np.copyto(dsts[i], s)

    # Bytes touched per repeat, per thread. A copy streams the source in and
    # the destination out, so both count.
    per_repeat = size_bytes * (2 if kind == "copy" else 1)
    total_bytes = per_repeat * repeats * threads

    results = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in range(rounds):
            t0 = time.perf_counter()
            if kind == "copy":
                futures = [
                    pool.submit(run_copy, srcs[i], dsts[i], repeats) for i in range(threads)
                ]
            else:
                futures = [pool.submit(run_read, srcs[i], repeats) for i in range(threads)]
            for f in futures:
                f.result()
            elapsed = time.perf_counter() - t0
            results.append(total_bytes / elapsed / 1e9)
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="concurrent streams (default: every visible core — fewer will understate the board)",
    )
    p.add_argument(
        "--size-mb",
        type=int,
        default=512,
        help="private buffer size PER THREAD in MiB (default 512; must far exceed last-level cache)",
    )
    p.add_argument("--repeats", type=int, default=4, help="passes over the buffer per timed round")
    p.add_argument("--rounds", type=int, default=3, help="timed rounds; the best one is reported")
    p.add_argument(
        "--kinds",
        nargs="+",
        default=["read", "copy"],
        choices=["read", "copy"],
        help="which kernels to run",
    )
    args = p.parse_args()

    size_bytes = args.size_mb * MIB
    footprint = size_bytes * args.threads * (2 if "copy" in args.kinds else 1)
    print(f"memory bandwidth — {args.threads} threads x {args.size_mb} MiB")
    print(f"  peak footprint: {footprint / GIB:.1f} GiB — make sure this fits in RAM\n")

    best = {}
    for kind in args.kinds:
        rounds = measure(kind, args.threads, size_bytes, args.repeats, args.rounds)
        best[kind] = max(rounds)
        detail = ", ".join(f"{r:.1f}" for r in rounds)
        print(f"  {kind.upper():5s} best {best[kind]:6.1f} GB/s   (rounds: {detail})")

    if "read" in best:
        print(
            f"\nDecode ceiling implied by READ: a model streaming B bytes per token\n"
            f"cannot exceed {best['read']:.0f}/B tok/s. For reference, 5.0 GB/token\n"
            f"caps at {best['read'] / 5.0:.1f} tok/s."
        )
    print(
        "\nCompare against the effective bandwidth eullm actually achieves during\n"
        "decode (docs/arm-cix-p1-cpu-profile.md § 9.2). The gap between the two is\n"
        "all the headroom any accelerator sharing this DRAM could ever recover."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
