"""budget.sh bills by GPU, as CINECA does: 8 local hours a GPU-hour.

Until 2026-11-20 it counted every job as a whole node (32), and read the Fast
Lane at 1,767 node-hours where saldo had about 650.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "budget.sh"

STUBS = {
    "saldo": 'echo "account start end total local cons pct mt mc"\n'
             'echo "AIFAC_P02_1147 20260902 20261102 40000 x 40261 100 0 0"\n'
             'echo "AIFAC_F02_1400 20261102 20270202 240000 x 19193 8 78260 20785"\n',
    # 1 GPU for 10 h, 3 GPUs for 2 h, a whole node for 1 h, 8 cores and no GPU for 1 h
    "sacct": "printf '36000|cpu=8,gres/gpu=1,node=1|8\\n7200|cpu=24,gres/gpu=3,node=1|24\\n"
             "3600|cpu=32,gres/gpu=4,node=1|32\\n3600|cpu=8,mem=4G,node=1|8\\n'\n",
    "squeue": "printf '2:00:00 1 gres/gpu:1 8\\n1:00:00 1 gres/gpu:3 24\\n"
              "30:00 4 gres/gpu:4 32\\n'\n",
}


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_billed_by_gpu_and_the_open_allocation(tmp_path):
    for name, body in STUBS.items():
        p = tmp_path / name
        p.write_text("#!/bin/bash\n" + body, encoding="utf-8")
        p.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    out = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env,
                         encoding="utf-8", check=True).stdout
    assert "allocation        240000 core-h" in out           # the Fast Lane, not the Playground
    assert "consumed             168 core-h" in out           # 80 + 48 + 32 + 8
    assert "still bookable       104 core-h" in out           # 16 + 24 + 64
    assert "straight line" in out and "GPUs round the clock" in out
