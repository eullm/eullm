"""Tests for the allocation queue/idle accounting.

Two of these encode bugs that were live: the first version charged
dependency wait to the scheduler (77 hours reported against 7 real), and an
earlier helper left trailing whitespace on a value it was supposed to trim.
The attribution rules are the whole point of the script — if they drift, the
numbers stay plausible and become wrong, which is the worst failure mode for
something feeding a report.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leonardo" / "queue_stats.py"


def _load():
    spec = importlib.util.spec_from_file_location("queue_stats", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qs = _load()


def at(stamp: str) -> datetime:
    return datetime.strptime(stamp, qs.FMT)


# The real Phase-1 chain: three 24 h jobs submitted together, so each waits
# for its predecessor. Job 3 also waited 7h06m for a free node.
CHAIN = "\n".join([
    "56803262|eullm-p1|TIMEOUT|2026-09-08T20:50:00|2026-09-08T20:57:02|2026-09-09T21:00:06|86584|1",
    "56818854|eullm-p1|TIMEOUT|2026-09-08T22:08:00|2026-09-09T21:03:26|2026-09-10T21:07:03|86617|1",
    "56818982|eullm-p1|RUNNING|2026-09-08T22:09:00|2026-09-11T04:13:23|Unknown|21397|1",
    # Submitted but never started: must not appear anywhere.
    "57294041|eullm-p6|PENDING|2026-09-11T09:50:00|Unknown|Unknown|0|1",
])
NOW = at("2026-09-11T10:10:00")


def test_pending_jobs_are_skipped():
    jobs = qs.parse(CHAIN, NOW)
    assert [j["id"] for j in jobs] == ["56803262", "56818854", "56818982"]


def test_running_job_counts_as_busy_until_now():
    jobs = qs.parse(CHAIN, NOW)
    assert jobs[-1]["end"] == NOW


def test_queue_wait_excludes_dependency_wait():
    """The bug this file exists for.

    56818982 was submitted 2026-09-08 22:09 and started 2026-09-11 04:13 —
    54 hours later. All but 7h06m of that was waiting for two predecessors
    it was explicitly told to wait for.
    """
    jobs = qs.parse(CHAIN, NOW)
    job = next(j for j in jobs if j["id"] == "56818982")
    wait = (job["start"] - qs.gate_of(job, jobs)).total_seconds()
    assert wait == pytest.approx(7 * 3600 + 6 * 60, abs=60)


def test_first_job_of_a_chain_measures_from_submission():
    """With no predecessor there is nothing to subtract."""
    jobs = qs.parse(CHAIN, NOW)
    job = next(j for j in jobs if j["id"] == "56803262")
    wait = (job["start"] - qs.gate_of(job, jobs)).total_seconds()
    assert wait == pytest.approx(7 * 60, abs=30)


def test_unrelated_job_names_are_not_treated_as_a_chain():
    """A differently-named job must not become someone's predecessor."""
    raw = "\n".join([
        "1|alpha|COMPLETED|2026-09-01T00:00:00|2026-09-01T00:00:00|2026-09-01T05:00:00|18000|1",
        "2|beta|COMPLETED|2026-09-01T00:00:00|2026-09-01T06:00:00|2026-09-01T07:00:00|3600|1",
    ])
    jobs = qs.parse(raw, at("2026-09-01T08:00:00"))
    beta = next(j for j in jobs if j["id"] == "2")
    # Gate is beta's own submission, not alpha's end, so the wait is 6 h.
    assert (beta["start"] - qs.gate_of(beta, jobs)).total_seconds() == 6 * 3600


def test_parallel_jobs_do_not_double_count_calendar():
    a, b = at("2026-09-01T00:00:00"), at("2026-09-01T01:00:00")
    merged = qs.merge([(a, b), (a, b)])
    assert len(merged) == 1
    assert (merged[0][1] - merged[0][0]).total_seconds() == 3600


def test_merge_joins_touching_intervals():
    a, b, c = (at("2026-09-01T00:00:00"), at("2026-09-01T01:00:00"),
               at("2026-09-01T02:00:00"))
    assert qs.merge([(a, b), (b, c)]) == [(a, c)]


def test_gap_with_a_job_queued_is_the_cluster_s_fault():
    jobs = qs.parse(CHAIN, NOW)
    busy = qs.merge([(j["start"], j["end"]) for j in jobs])
    gaps = qs.gaps_of(busy, jobs, at("2026-09-08T00:00:00"), NOW)
    stall = next(g for g in gaps if g["from"] == at("2026-09-10T21:07:03"))
    assert stall["cause"] == "cluster full"
    assert "56818982" in stall["waiting"]


def test_gap_with_nothing_queued_is_ours():
    """The allocation opened on the 2nd; the first job was submitted on the 8th."""
    jobs = qs.parse(CHAIN, NOW)
    busy = qs.merge([(j["start"], j["end"]) for j in jobs])
    gaps = qs.gaps_of(busy, jobs, at("2026-09-02T00:00:00"), NOW)
    head = gaps[0]
    assert head["from"] == at("2026-09-02T00:00:00")
    assert head["cause"] == "queue empty"
    assert head["waiting"] == []


def test_totals_separate_the_two_kinds_of_idle():
    jobs = qs.parse(CHAIN, NOW)
    s = qs.summarise(jobs, at("2026-09-02T00:00:00"), NOW)
    # 7h06m stall + a 3-minute handover, and six days before we started.
    assert s["idle_cluster_full_hours"] == pytest.approx(7.15, abs=0.1)
    assert s["idle_queue_empty_hours"] == pytest.approx(164.95, abs=0.1)
    # One node throughout, so node-hours and busy calendar agree.
    assert s["mean_nodes_when_busy"] == pytest.approx(1.0, abs=0.05)


def test_dependency_wait_is_reported_but_kept_apart():
    jobs = qs.parse(CHAIN, NOW)
    s = qs.summarise(jobs, at("2026-09-02T00:00:00"), NOW)
    assert s["resource_wait_seconds"] == pytest.approx(7 * 3600 + 16 * 60, abs=120)
    assert s["dependency_wait_seconds"] > 60 * 3600


def test_hours_formats_and_survives_negatives():
    assert qs.hours(0) == "0h00m"
    assert qs.hours(3600) == "1h00m"
    assert qs.hours(25620) == "7h07m"
    assert qs.hours(-90) == "-0h01m"


def test_malformed_sacct_rows_are_ignored():
    raw = "\n".join([
        "not|enough|fields",
        "9|x|COMPLETED|garbage|2026-09-01T00:00:00|2026-09-01T01:00:00|3600|1",
        "10|x|COMPLETED|2026-09-01T00:00:00|2026-09-01T00:00:00|2026-09-01T01:00:00|3600|1",
    ])
    jobs = qs.parse(raw, at("2026-09-01T02:00:00"))
    assert [j["id"] for j in jobs] == ["10"]
