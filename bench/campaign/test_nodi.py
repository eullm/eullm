"""tools/lumi/nodi.sh: the node-hours of each of the last 7 days, the average
a day, what is left and what a day must spend to use it by the plan's end,
from sacct and lumi-allocations (stand-ins here, as squeue)."""

from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import budget

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "lumi" / "nodi.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash")


def test_a_job_is_billed_on_the_days_it_ran():
    now = dt.datetime(2026, 10, 25, 9, 0)
    jobs = budget.parse_sacct(
        "1|eullm-campaign|standard-g|COMPLETED|14400|1|cpu=56,gres/gpu=8,mem=480G,node=1|"
        "2026-10-23T22:00:00|2026-10-24T02:00:00\n"
        "2|eullm-campaign|small-g|RUNNING|3600|1|cpu=28,gres/gpu=4,mem=200G,node=1|"
        "2026-10-25T08:00:00|Unknown\n")
    days = budget.per_day(jobs, dt.date(2026, 10, 23), now)
    # A node is 4 GPU-hours an hour; four GCDs of small-g 0.5 each.
    assert days == {dt.date(2026, 10, 23): 8.0, dt.date(2026, 10, 24): 8.0,
                    dt.date(2026, 10, 25): 2.0}


def test_nodi_prints_the_week_and_the_pace_to_the_plans_end(tmp_path):
    now = dt.datetime.now().replace(microsecond=0)
    day = dt.datetime.combine(now.date() - dt.timedelta(days=3), dt.time())
    rows = [f"1|eullm-campaign|standard-g|COMPLETED|86400|1|cpu=56,gres/gpu=8,mem=480G,node=1|"
            f"{day.isoformat()}|{(day + dt.timedelta(days=1)).isoformat()}",
            f"2|eullm-campaign|small-g|RUNNING|3600|1|cpu=28,gres/gpu=4,mem=200G,node=1|"
            f"{(now - dt.timedelta(hours=1)).isoformat()}|Unknown"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sacct").write_text("#!/bin/bash\ncat <<'EOF'\n" + "\n".join(rows) + "\nEOF\n")
    (bin_dir / "lumi-allocations").write_text(
        "#!/bin/bash\ncat <<'EOF'\nData updated: 2026-10-25 06:00:00\n"
        "  project_465003366     |  0/1000 (0.0%) core/hours|  4135/18000 (23.0%) gpu/hours|\n"
        "EOF\n")
    (bin_dir / "squeue").write_text(
        "#!/bin/bash\ncase \"$*\" in *RUNNING*) echo eullm-campaign/small-g ;; "
        "*) echo eullm-campaign/standard-g; echo eullm-campaign/standard-g ;; esac\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", USER="tester",
               PY=sys.executable, CAMPAIGN_DIR=str(tmp_path / "campaign"))
    env.pop("EULLM_REPO", None)
    p = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
                       timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    out = p.stdout
    assert f"{day.date()}  {day.strftime('%a %d-%m'):<14}   24.0  {'#' * 24}" in out
    assert "last 7 days, to now" in out and "24.5  (3.5 a day)" in out
    assert "3,466 node-hours billed by lumi-allocations (updated 2026-10-25 06:00:00)" in out
    assert "to the plan's end 2027-03-09" in out and "to the allocation's end 2027-03-12" in out
    assert "the hours run out on" not in out and "they last past 2027-03-09" in out
    assert "the straight line 2026-09-12 to 2027-03-09" in out
    assert "Running: eullm-campaign/small-g x1" in out
    assert "Waiting: eullm-campaign/standard-g x2" in out


def test_the_margin_is_the_spend_ahead_of_the_straight_line_in_days_of_it():
    """04-11-2026: 1,648 of 4,500 node-hours spent 53.9 days into the 179 from
    12-09-2026 to the end of 09-03-2027; the line had 1,355 by then, so 293
    ahead, about 11.6 days of its 25.1 a day without a job."""
    now = dt.datetime(2026, 11, 4, 21, 40)
    m = budget.margin(4500, dt.date(2026, 9, 12), dt.date(2027, 3, 9), now, 1648)
    assert m["line_node_hours_per_day"] == 25.1
    assert 290 < m["ahead_node_hours"] < 296 and 11.5 < m["days"] < 11.8
    behind = budget.margin(4500, dt.date(2026, 9, 12), dt.date(2027, 3, 9), now, 1200)
    assert behind["ahead_node_hours"] < 0 and behind["days"] < 0
    # Before the start nothing is due; after the end all of it.
    assert budget.margin(100, dt.date(2026, 9, 12), dt.date(2027, 3, 9),
                         dt.datetime(2026, 9, 1), 0)["ahead_node_hours"] == 0
    assert budget.margin(100, dt.date(2026, 9, 12), dt.date(2027, 3, 9),
                         dt.datetime(2027, 4, 1), 100)["ahead_node_hours"] == 0
