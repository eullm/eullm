"""Which devices a point gets, which cores drive them, and what they did.

Device numbers in a point are the job's own (0..n-1). They are translated
through what the job was given — Slurm's ROCR_VISIBLE_DEVICES or
CUDA_VISIBLE_DEVICES — so that a server pinned to "device 2" is pinned to the
right physical GCD on a partial allocation as well as on a whole node.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time

# LUMI-G: the cores closest to each GCD, from LUMI's documented binding
# (mask_cpu fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,
# fe00000000,fe0000000000 for GCDs 0..7). GCD and NUMA numbering do not
# match — GCDs 0-1 sit on NUMA 3, 2-3 on 1, 4-5 on 0, 6-7 on 2 — and the
# first core of each CCD is reserved, so seven cores per GCD.
LUMI_GCD_CORES = {
    0: "49-55",
    1: "57-63",
    2: "17-23",
    3: "25-31",
    4: "1-7",
    5: "9-15",
    6: "33-39",
    7: "41-47",
}

VISIBLE_ENV = {"rocm": "ROCR_VISIBLE_DEVICES", "cuda": "CUDA_VISIBLE_DEVICES"}


def parse_devices(text: str) -> list:
    """'0-7', '0,1,4' or '0-3,6' → sorted device numbers."""
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def physical_map(backend: str, logical: list, environ=None) -> dict:
    """logical device → physical id, through what Slurm made visible."""
    environ = os.environ if environ is None else environ
    given = environ.get(VISIBLE_ENV[backend], "")
    phys = [p.strip() for p in given.split(",") if p.strip()]
    if not phys:
        return {d: str(d) for d in logical}
    if len(phys) < len(logical):
        raise ValueError(
            f"{len(logical)} devices asked for, the job was given {len(phys)} ({given})"
        )
    return {d: phys[i] for i, d in enumerate(logical)}


def aligned_group(free: set, devices: list, width: int):
    """`width` free devices forming an aligned block of the node's list —
    a pair on one MI250X module, a half node, the whole node — or None.
    Single devices are taken lowest first, which keeps pairs whole longest."""
    for start in range(0, len(devices) - width + 1, width):
        block = devices[start : start + width]
        if len(block) == width and all(d in free for d in block):
            return block
    return None


def cores_for(binding: str, physical: list):
    """The core list a server on these physical devices should run on, or
    None when there is no binding to apply."""
    if binding != "lumi":
        return None
    try:
        return ",".join(LUMI_GCD_CORES[int(p)] for p in physical)
    except (KeyError, ValueError):
        return None


# rocm-smi's text output: one line per device and quantity, e.g.
#   GPU[3]          : GPU use (%): 97
#   GPU[3]          : VRAM Total Used Memory (B): 33554432000
ROCM_USE = re.compile(r"GPU\[(\d+)\].*?GPU use \(%\):\s*(\d+)")
ROCM_USED = re.compile(r"GPU\[(\d+)\].*?VRAM Total Used Memory \(B\):\s*(\d+)")
ROCM_TOTAL = re.compile(r"GPU\[(\d+)\].*?VRAM Total Memory \(B\):\s*(\d+)")


def parse_rocm_smi(text: str) -> dict:
    """{device: {"used": bytes, "total": bytes, "use": percent}}."""
    out: dict = {}
    for pattern, key in ((ROCM_USE, "use"), (ROCM_USED, "used"), (ROCM_TOTAL, "total")):
        for m in pattern.finditer(text):
            out.setdefault(m.group(1), {})[key] = int(m.group(2))
    return out


def parse_nvidia_smi(text: str) -> dict:
    """From `nvidia-smi --query-gpu=index,memory.used,memory.total,
    utilization.gpu --format=csv,noheader,nounits` (MiB)."""
    out = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4 or not parts[0].isdigit():
            continue
        idx, used, total, use = parts
        try:
            out[idx] = {
                "used": int(float(used)) * 2**20,
                "total": int(float(total)) * 2**20,
                "use": int(float(use)),
            }
        except ValueError:
            continue
    return out


SMI = {
    "rocm": (["rocm-smi", "--showuse", "--showmeminfo", "vram"], parse_rocm_smi),
    "cuda": (
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        parse_nvidia_smi,
    ),
}


def read_smi(backend: str) -> dict:
    cmd, parse = SMI[backend]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                             encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return parse(out)


class NodeSampler:
    """One reader of the node's devices for every point on it: eight points
    each polling rocm-smi would cost more than they measure. Keeps samples in
    memory for the points' windows and appends them to `log_path`."""

    def __init__(self, backend: str, interval_s: float = 2.0, log_path=None, reader=None):
        self.backend, self.interval_s, self.log_path = backend, interval_s, log_path
        self.reader = reader or (lambda: read_smi(backend))
        self.samples: list = []  # (time, {physical id: {...}})
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=60)

    def sample_once(self) -> None:
        t, reading = time.time(), self.reader()
        if not reading:
            return
        with self._lock:
            self.samples.append((t, reading))
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"t": round(t, 1), "d": reading}, separators=(",", ":")))
                f.write("\n")

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def window(self, start: float, end: float, physical: list) -> dict:
        """Per device over [start, end]: peak and total VRAM in MiB, mean and
        max use %, sample count."""
        with self._lock:
            inside = [r for t, r in self.samples if start <= t <= end]
        stats = {}
        for dev in physical:
            rows = [r[dev] for r in inside if dev in r]
            used = [r["used"] for r in rows if "used" in r]
            use = [r["use"] for r in rows if "use" in r]
            total = next((r["total"] for r in rows if "total" in r), None)
            stats[dev] = {
                "samples": len(rows),
                "vram_peak_mib": round(max(used) / 2**20) if used else None,
                "vram_total_mib": round(total / 2**20) if total else None,
                "use_mean": round(sum(use) / len(use), 1) if use else None,
                "use_max": max(use) if use else None,
            }
        return stats

    def latest(self, max_age_s: float = 10.0, since: float = 0.0) -> dict:
        """The newest reading if it is younger than `max_age_s` and was taken
        at `since` or later, else a fresh one: what the devices hold now,
        before a point starts."""
        with self._lock:
            last = self.samples[-1] if self.samples else None
        if last and last[0] >= since and time.time() - last[0] <= max_age_s:
            return last[1]
        return self.reader() or {}

    def forget_before(self, t: float) -> None:
        with self._lock:
            self.samples = [s for s in self.samples if s[0] >= t]
