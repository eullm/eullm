"""tools/lumi/log_checks.py: the expert cache's statistics and llama.cpp's
pool and errors, read from a job's server logs."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("log_checks",
                                              ROOT / "tools" / "lumi" / "log_checks.py")
log_checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log_checks)

# Lines as patch 0002 and 0011 print them.
WINDOW = ("llama_moe_cache: 64 steps of 8.0 tokens: 61.2 ms/step = 20.1 ms to the routers of"
          " 94 layers + 3.2 ms in the cache + 37.9 ms after; {mib} MiB/step copied, {share}% of"
          " the experts in VRAM{tail}\n")
CACHE = (WINDOW.format(mib="812.4", share="71.0", tail="")
         + WINDOW.format(mib="790.0", share="75.0",
                         tail="; copying takes 30.1 ms/step (26.9 GB/s)")
         + "llama_moe_cache: ubatch <= 8: hits = 730, misses = 270, hit rate = 73.00%,"
           " uploaded = 52000.00 MiB\n"
         + "llama_moe_cache: ubatch  > 8: hits = 100, misses = 900, hit rate = 10.00%,"
           " uploaded = 9000.00 MiB\n")
POOL = ("INFO llama-cpp-2: ROCm0 pool[0]: {free} MiB free, freeing {freed} MiB of cached buffers"
        " before allocating 64.00 MiB\n")
SOAK = POOL.format(free="300.00", freed="1200.00") + POOL.format(free="280.00", freed="900.00")
BAD = (":0:rocdevice.cpp            : 2993: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION\n"
       "ERROR llama-cpp-2: ROCm error: out of memory\n"
       "WARN eullm: logits are nan at step 12\n")


def write(d, name, text):
    os.makedirs(d, exist_ok=True)
    (Path(d) / name).write_text(text, encoding="utf-8")


def test_the_logs_say_the_cache_share_the_flushes_and_the_errors(tmp_path, capsys):
    logs = tmp_path / "results" / "logs" / "42"
    write(logs, "bump-moe-ab12.server0.log", CACHE)
    write(logs, "bump-soak-32b-cd34.server0.log", SOAK)
    write(logs, "bump-soak-35b-a3b-ef56.server0.log", BAD)
    write(logs, "bump-mtp-graded-0000.server0.log", "nothing to say\n")
    assert log_checks.main(["--queue", str(tmp_path), "42"]) == 0
    out = capsys.readouterr().out
    assert ("cache bump-moe-ab12 server0: 2 windows of 8.0 tokens, 73.0% of the experts in "
            "VRAM (71.0-75.0), 801.2 MiB/step copied; at exit ubatch <= 8: 73.00% hit, "
            "52,000 MiB copied; at exit ubatch > 8: 10.00% hit, 9,000 MiB copied") in out
    assert "pool bump-soak-32b-cd34 server0: 2 flushes (pool[...] freeing)\n" in out
    assert ("pool bump-soak-35b-a3b-ef56 server0: 0 flushes (pool[...] freeing); "
            "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION x1; out of memory x1; NaN x1") in out
    assert "bump-mtp-graded" not in out
    assert out.strip().endswith(
        "4 server logs: 2 pool flushes, 1 HSA errors, 1 out of memory, 1 NaN")


def test_a_job_without_logs_says_so(tmp_path, capsys):
    assert log_checks.main(["--queue", str(tmp_path), "7"]) == 0
    assert "no server logs for job 7" in capsys.readouterr().out
