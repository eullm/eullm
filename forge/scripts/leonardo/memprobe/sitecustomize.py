"""CUDA memory probe, installed by putting this directory on PYTHONPATH.

Python imports `sitecustomize` automatically at interpreter startup, which
makes this the one place we can instrument LLaMA-Factory's worker processes
without forking it: the training entry point is theirs, not ours, and by the
time an OOM propagates the allocator state that caused it is gone.

Three OOMs on the Qwen3-30B-A3B teacher were diagnosed by inference — the
exception says how much was free and how much was requested, and nothing
about *what* had taken the rest. This prints, for every rank:

  * on a CUDA OOM: `torch.cuda.memory_summary()`, which breaks the reserved
    pool down by allocation size and shows fragmentation;
  * at exit, whatever the outcome: peak allocated and peak reserved per
    device, which are the numbers a run log needs and which cannot be
    recovered afterwards.

Deliberately defensive throughout. A probe that raises inside an excepthook
or an atexit handler would turn a diagnosable failure into an undiagnosable
one, so every path is wrapped and failure is silent.
"""

from __future__ import annotations

import atexit
import os
import sys

_TAG = f"[memprobe rank {os.environ.get('RANK', os.environ.get('LOCAL_RANK', '?'))}]"


def _torch():
    """Return torch if it is already imported and CUDA is usable, else None.

    Never imports torch itself: sitecustomize runs before anything else, and
    importing a multi-second dependency at startup in every subprocess — pip,
    the shell's python, LLaMA-Factory's own helpers — would be a real cost for
    no benefit. If the process never loaded torch, there is nothing to report.
    """
    mod = sys.modules.get("torch")
    try:
        if mod is None or not mod.cuda.is_available():
            return None
    except Exception:
        return None
    return mod


def _emit(line: str) -> None:
    try:
        print(f"{_TAG} {line}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _dump_totals(when: str) -> None:
    torch = _torch()
    if torch is None:
        return
    try:
        gib = 1024 ** 3
        for i in range(torch.cuda.device_count()):
            # Skip devices this process never allocated on. Dataloader and
            # preprocessing workers import torch without touching CUDA, and
            # four zero lines each would bury the ranks that matter.
            if torch.cuda.max_memory_allocated(i) == 0:
                continue
            _emit(
                f"{when} dev{i}: "
                f"peak_allocated={torch.cuda.max_memory_allocated(i) / gib:.2f} GiB  "
                f"peak_reserved={torch.cuda.max_memory_reserved(i) / gib:.2f} GiB  "
                f"now_allocated={torch.cuda.memory_allocated(i) / gib:.2f} GiB  "
                f"now_reserved={torch.cuda.memory_reserved(i) / gib:.2f} GiB"
            )
    except Exception as exc:
        _emit(f"{when}: could not read memory stats ({type(exc).__name__})")


def _dump_summary() -> None:
    torch = _torch()
    if torch is None:
        return
    try:
        _emit("OOM — allocator summary for the current device follows")
        print(torch.cuda.memory_summary(), file=sys.stderr, flush=True)
    except Exception as exc:
        _emit(f"could not produce memory_summary ({type(exc).__name__})")


_previous_excepthook = sys.excepthook


def _excepthook(exc_type, exc, tb):  # noqa: ANN001 - signature is fixed
    try:
        name = getattr(exc_type, "__name__", "")
        if "OutOfMemory" in name or "out of memory" in str(exc):
            _dump_totals("at OOM")
            _dump_summary()
    except Exception:
        pass
    _previous_excepthook(exc_type, exc, tb)


sys.excepthook = _excepthook

# The exception may never reach the excepthook — torch.distributed.elastic and
# the Trainer both catch and re-raise through their own machinery — so the
# peak numbers are also emitted unconditionally at interpreter shutdown.
atexit.register(_dump_totals, "at exit")

# Announce once, from the first rank only: every child process imports
# sitecustomize too, and eight preprocessing workers saying "installed" is
# noise in a log whose whole purpose is to be readable.
if os.environ.get("RANK", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0":
    _emit("installed (peaks at exit, allocator summary on CUDA OOM)")
