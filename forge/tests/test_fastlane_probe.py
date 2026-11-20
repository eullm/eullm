"""The Fast Lane probe report reads the scheduler's answer correctly.

The report decides whether the project can use whole nodes, long jobs and
16 + 16 GPUs, so a misread state or reason would steer three months of
allocation. These tests feed it scheduler output of the shapes Slurm prints.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "fastlane_probe_report.py"
spec = importlib.util.spec_from_file_location("fastlane_probe_report", SCRIPT)
flp = importlib.util.module_from_spec(spec)
sys.modules["fastlane_probe_report"] = flp
spec.loader.exec_module(flp)

MANIFEST = (
    "id\tname\tnodes\tgpus\ttime\tqos\thold\tsubmitted\tnote\n"
    "101\tg3-2h\t1\t3\t02:00:00\tnormal\t0\t2026-11-03T10:00:00\t\n"
    "102\tn1-24h\t1\t4\t24:00:00\tnormal\t0\t2026-11-03T10:00:00\t\n"
    "103\tn8-2h\t8\t4\t02:00:00\tnormal\t0\t2026-11-03T10:00:00\t\n"
    "104\tpar-a\t4\t4\t00:30:00\tnormal\t900\t2026-11-03T10:00:00\t\n"
    "105\tpar-b\t4\t4\t00:30:00\tnormal\t900\t2026-11-03T10:00:00\t\n"
    "\tlprod-n4-8h\t4\t4\t08:00:00\tboost_qos_lprod\t0\t2026-11-03T10:00:01\t"
    "sbatch: error: QOSMaxWallDurationPerJobLimit\n"
)

SACCT = (
    "101|COMPLETED|2026-11-03T10:00:00|2026-11-03T10:01:30|2026-11-03T10:01:35|None\n"
    "102|PENDING|2026-11-03T10:00:00|Unknown|Unknown|Priority\n"
    "103|PENDING|2026-11-03T10:00:00|Unknown|Unknown|QOSMaxGRESPerUser\n"
    "104|COMPLETED|2026-11-03T10:00:00|2026-11-03T10:05:00|2026-11-03T10:20:00|None\n"
    "105|COMPLETED|2026-11-03T10:00:00|2026-11-03T10:10:00|2026-11-03T10:25:00|None\n"
)

SQUEUE = (
    "102|PENDING|Priority|2026-11-04T08:00:00\n"
    "103|PENDING|QOSMaxGRESPerUser|N/A\n"
    "999|RUNNING|None|2026-11-03T09:00:00\n"
)

NOW = datetime(2026, 11, 3, 11, 0, 0)


def rows(tmp_path):
    (tmp_path / "manifest.tsv").write_text(MANIFEST, encoding="utf-8")
    manifest = flp.read_manifest(tmp_path / "manifest.tsv")
    return flp.build(manifest, flp.parse_sacct(SACCT), flp.parse_squeue(SQUEUE), NOW)


def test_each_probe_gets_the_right_verdict(tmp_path):
    by_name = {r["name"]: r for r in rows(tmp_path)}
    assert by_name["g3-2h"]["verdict"] == "started"
    assert by_name["g3-2h"]["waited_s"] == 90
    assert by_name["n1-24h"]["verdict"] == "waiting"
    assert by_name["n1-24h"]["waited_s"] == 3600  # still waiting: counted to now
    assert by_name["n8-2h"]["verdict"] == "limit"
    assert by_name["lprod-n4-8h"]["verdict"] == "rejected"


def test_limit_reasons_are_told_from_a_busy_machine():
    for reason in ("QOSMaxGRESPerUser", "AssocGrpGRES", "QOSMaxWallDurationPerJobLimit",
                   "QOSMaxNodePerJobLimit", "AssocMaxJobsLimit", "QOSGrpGRES"):
        assert flp.classify("PENDING", reason) == "limit", reason
    for reason in ("Resources", "Priority", "ReqNodeNotAvail", "BeginTime"):
        assert flp.classify("PENDING", reason) == "waiting", reason
    assert flp.classify("TIMEOUT", "None") == "started"


def test_sixteen_plus_sixteen_overlap_is_seen(tmp_path):
    assert flp.concurrency(rows(tmp_path)).startswith("SÌ")


def test_sixteen_plus_sixteen_one_after_the_other_is_not_a_no(tmp_path):
    sacct = SACCT.replace("2026-11-03T10:10:00|2026-11-03T10:25:00",
                          "2026-11-03T10:21:00|2026-11-03T10:36:00")
    manifest = flp.read_manifest(_write(tmp_path))
    r = flp.build(manifest, flp.parse_sacct(sacct), flp.parse_squeue(SQUEUE), NOW)
    assert flp.concurrency(r).startswith("non dimostrato")


def test_sixteen_plus_sixteen_blocked_by_a_limit(tmp_path):
    sacct = SACCT.replace(
        "105|COMPLETED|2026-11-03T10:00:00|2026-11-03T10:10:00|2026-11-03T10:25:00|None",
        "105|PENDING|2026-11-03T10:00:00|Unknown|Unknown|QOSMaxGRESPerUser")
    manifest = flp.read_manifest(_write(tmp_path))
    r = flp.build(manifest, flp.parse_sacct(sacct), flp.parse_squeue(""), NOW)
    assert flp.concurrency(r) == "NO, limite: QOSMaxGRESPerUser"


def test_report_names_the_limits_and_the_planned_start(tmp_path):
    text = flp.render(rows(tmp_path), final=False, now=NOW)
    assert "QOSMaxGRESPerUser" in text
    assert "partenza prevista 04/11 08:00" in text
    assert "QOSMaxWallDurationPerJobLimit" in text  # the rejected submission's note
    assert "Forma più grande partita:** par-a" in text


def _write(tmp_path):
    path = tmp_path / "manifest.tsv"
    path.write_text(MANIFEST, encoding="utf-8")
    return path


def test_scheduler_output_is_read_as_utf8(monkeypatch):
    """sacct/squeue print job names as their users wrote them: read the
    output as UTF-8, not with the host locale, so a non-ASCII name is kept."""
    class Out:
        stdout = "citt\u00e0-job|COMPLETED\n"
        stderr = ""
        returncode = 0

    seen = []

    def run(cmd, **kwargs):
        seen.append(kwargs)
        return Out()

    monkeypatch.setattr(flp.subprocess, "run", run)
    assert flp.run(["sacct"]) == "citt\u00e0-job|COMPLETED\n"
    assert seen[0].get("encoding") == "utf-8"
