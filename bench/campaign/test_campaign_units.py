"""Unit tests: spec expansion, the file queue, devices, billing and pace."""

import datetime as dt
import os
from types import SimpleNamespace

import pytest

import budget
import campaign
from devices import (
    NodeSampler,
    aligned_group,
    cores_for,
    parse_devices,
    parse_nvidia_smi,
    parse_rocm_smi,
    physical_map,
)
from fsqueue import ORPHAN_GRACE_S, Queue
from spec import SpecError, expand, normalize, order_key, width

SPEC = {
    "campaign": "c-test",
    "defaults": {"repeats": 2},
    "groups": [
        {"name": "one", "priority": 5, "set": {"gcds": 1},
         "axes": {"model": ["a", "b"], "batch": [1, 16]}},
        {"name": "wide", "priority": 5, "set": {"gcds": 8, "replica_gcds": 2, "model": "a"}},
    ],
}


# ── spec ─────────────────────────────────────────────────────────────────


def test_expand_is_the_cartesian_product_and_stable():
    a, b = expand(SPEC), expand(SPEC)
    assert len(a) == 5
    assert [p["id"] for p in a] == [p["id"] for p in b]
    assert len({p["id"] for p in a}) == 5


def test_hints_do_not_change_the_id():
    other = dict(SPEC, groups=[dict(SPEC["groups"][0], priority=99, est_s=1)])
    ids = {p["id"] for p in expand(other)}
    assert ids <= {p["id"] for p in expand(SPEC)}


def test_normalize_fills_ctx_concurrency_and_mode():
    p = normalize({"model": "m", "batch": 16, "slot_ctx": 4096})
    assert p["ctx"] == 65536 and p["concurrency"] == 16 and p["mode"] == "single"
    r = normalize({"model": "m", "gcds": 8, "replica_gcds": 1, "batch": 4})
    assert r["replicas"] == 8 and r["mode"] == "replicas" and r["concurrency"] == 32
    s = normalize({"model": "m", "gcds": 8, "replica_gcds": 2})
    assert s["replicas"] == 4 and s["mode"] == "split-replicas"
    assert normalize({"model": "m", "kv": "q8_0"})["kv"] == "q8_0/q8_0"


@pytest.mark.parametrize("bad", [
    {"model": "m", "gcds": 3},
    {"model": "m", "gcds": 4, "replica_gcds": 8},
    {"kind": "workload", "model": "m"},
    {"model": "m", "kind": "nope"},
    {"gcds": 1},
    {"kind": "finetune", "model": "m.gguf"},
    {"kind": "finetune", "model": "m.gguf", "data": "d", "gcds": 2},
    {"kind": "finetune", "model": "m.gguf", "data": "d", "ft_ctx": 300},
    {"kind": "finetune", "model": "m.gguf", "data": "d", "optimizer": "lion"},
    {"kind": "finetune", "model": "m.gguf", "data": "d", "lr": 0},
])
def test_normalize_rejects(bad):
    with pytest.raises(SpecError):
        normalize(bad)


def test_finetune_fields_belong_to_finetune_points_only():
    ft = normalize({"kind": "finetune", "model": "m.gguf", "data": "d.jsonl",
                    "train_tensors": ["blk.*.attn_*"]})
    assert ft["ft_ctx"] == 512 and ft["optimizer"] == "adamw" and ft["keep_output"] is False
    # The other kinds keep the fields, and so the ids, they had before.
    assert "ft_ctx" not in normalize({"model": "m"})


def test_finetune_command_resolves_the_queue_dirs():
    import point

    p = normalize({"kind": "finetune", "model": "m-f32.gguf", "data": "d.jsonl",
                   "train_tensors": ["blk.*.attn_*", "output.weight"], "lr": 1e-4})
    ctx = point.Context("/bin/eullm", "rocm", "none", ["3"], 0, "/w",
                        sets_dir="/q/sets", f32_dir="/q/f32")
    cmd = point.finetune_command(p, ctx, "/w/o.gguf", "/w/r.json")
    assert cmd[:3] == ["/bin/eullm", "finetune", "/q/f32/m-f32.gguf"]
    assert cmd[cmd.index("--data") + 1] == "/q/sets/d.jsonl"
    assert cmd[cmd.index("--lr") + 1] == "0.0001"
    assert cmd[cmd.index("--device") + 1] == "0"  # the one device left visible
    assert cmd[cmd.index("--train-tensors") + 1] == "blk.*.attn_*,output.weight"
    assert "--no-progress" in cmd


def test_gsm8k_train_text_drops_the_calculator_annotations():
    import json

    import campaign

    row = {"question": "Natalia sold 48 clips in April and half as many in May. How many? ",
           "answer": "In May: 48/2 = <<48/2=24>>24.\nIn all: 48+24 = <<48+24=72>>72.\n#### 72"}
    [doc] = campaign.gsm8k_train_text((json.dumps(row) + "\n\n").encode())
    assert doc == ("Question: Natalia sold 48 clips in April and half as many in May. How many?\n"
                   "In May: 48/2 = 24.\nIn all: 48+24 = 72.\nAnswer: 72")


def test_finetune_models_are_f32_files_not_pulls(tmp_path):
    import campaign

    spec = {"campaign": "c", "f32": {"a-f32.gguf": "Org/A"}, "groups": [
        {"name": "serve", "set": {"model": "qwen3-8b"}},
        {"name": "ft", "set": {"kind": "finetune", "data": "d.jsonl"},
         "axes": {"model": ["a-f32.gguf", "b-f32.gguf"]}},
    ]}
    specs = [("s.json", spec)]
    assert campaign.pull_refs(specs) == [("qwen3-8b", "qwen3-8b")]
    assert campaign.f32_refs(specs) == [("a-f32.gguf", "Org/A"), ("b-f32.gguf", None)]
    (tmp_path / "f32").mkdir()
    (tmp_path / "f32" / "a-f32.gguf").write_bytes(b"GGUF")
    assert campaign.missing_f32(str(tmp_path), specs) == [("b-f32.gguf", None)]


def test_order_is_priority_then_width():
    pts = [dict(id="x", priority=1, gcds=1), dict(id="y", priority=1, gcds=8),
           dict(id="z", priority=2, gcds=1), dict(id="w", priority=1, gcds=1, exclusive=True)]
    assert [p["id"] for p in sorted(pts, key=order_key)] == ["z", "w", "y", "x"]
    assert width(pts[3]) == 8


# ── queue ────────────────────────────────────────────────────────────────


def test_queue_add_is_idempotent_and_claim_exclusive(tmp_path):
    q = Queue(str(tmp_path))
    p = expand(SPEC)[0]
    assert q.add(p) and not q.add(p)
    assert q.claim(p["id"], {"job": "1"})["id"] == p["id"]
    assert q.claim(p["id"], {"job": "2"}) is None  # another node got there first
    assert q.where(p["id"]) == "running"
    assert not q.add(p)  # known while running too


def test_failure_is_retried_then_kept(tmp_path):
    q = Queue(str(tmp_path))
    p = expand(SPEC)[0]
    q.add(p)
    q.claim(p["id"], {"job": "1"})
    assert q.finish(p["id"], "failed", "boom") == "todo"
    q.claim(p["id"], {"job": "1"})
    assert q.finish(p["id"], "failed", "boom again") == "failed"
    assert q.load("failed", p["id"])["notes"] == ["boom", "boom again"]


def test_a_failure_that_must_not_be_retried_stays_failed_with_its_note(tmp_path):
    """A server death is the soak's finding: retrying it would replace the
    crash with a later healthy run and hide it from the report."""
    q = Queue(str(tmp_path))
    p = expand(SPEC)[0]
    q.add(p)
    q.claim(p["id"], {"job": "1"})
    assert q.finish(p["id"], "failed", "server exited 134", retry=False) == "failed"
    failed = q.load("failed", p["id"])
    assert failed["attempts"] == 1
    assert failed["notes"] == ["server exited 134"]


def test_stale_claims_go_back(tmp_path):
    q = Queue(str(tmp_path))
    a, b, c = expand(SPEC)[:3]
    for p in (a, b, c):
        q.add(p)
    q.claim(a["id"], {"job": "dead"})
    q.claim(b["id"], {"job": "alive"})
    # c: claimed by a node that died before writing its owner file
    os.rename(os.path.join(q.root, "todo", c["id"] + ".json"),
              os.path.join(q.root, "running", c["id"] + ".json"))
    now = os.path.getmtime(os.path.join(q.root, "running", c["id"] + ".json"))
    assert q.requeue_stale({"alive"}, now, 3600) == [a["id"]]
    assert q.requeue_stale({"alive"}, now + ORPHAN_GRACE_S + 1, 3600) == [c["id"]]
    assert q.where(b["id"]) == "running"


def test_blocked_points_can_be_unblocked(tmp_path):
    q = Queue(str(tmp_path))
    p = expand(SPEC)[0]
    q.add(p)
    q.claim(p["id"], {"job": "1"})
    q.finish(p["id"], "blocked", "model not pulled")
    assert q.counts()["blocked"] == 1
    assert q.unblock() == [p["id"]] and q.where(p["id"]) == "todo"


# ── devices ──────────────────────────────────────────────────────────────


def test_parse_devices():
    assert parse_devices("0-3,6") == [0, 1, 2, 3, 6]
    assert parse_devices("7,1") == [1, 7]


def test_physical_map_follows_what_slurm_gave():
    assert physical_map("rocm", [0, 1], {}) == {0: "0", 1: "1"}
    assert physical_map("rocm", [0, 1], {"ROCR_VISIBLE_DEVICES": "4,5"}) == {0: "4", 1: "5"}
    with pytest.raises(ValueError):
        physical_map("cuda", [0, 1, 2], {"CUDA_VISIBLE_DEVICES": "0"})


def test_aligned_groups_keep_pairs_on_a_module():
    devs = list(range(8))
    assert aligned_group(set(devs), devs, 2) == [0, 1]
    assert aligned_group({1, 2, 3, 4, 5}, devs, 2) == [2, 3]
    assert aligned_group({1, 2, 3, 4, 5, 6, 7}, devs, 4) == [4, 5, 6, 7]
    assert aligned_group({0, 1, 2, 3, 4, 5, 6}, devs, 8) is None
    assert aligned_group({5}, devs, 1) == [5]


def test_lumi_cores_follow_the_documented_mask():
    assert cores_for("lumi", ["0"]) == "49-55"
    assert cores_for("lumi", ["4", "5"]) == "1-7,9-15"
    assert cores_for("none", ["0"]) is None
    assert cores_for("lumi", ["GPU-uuid"]) is None


ROCM_SMI = """
============================ ROCm System Management Interface ============================
GPU[0]          : GPU use (%): 97
GPU[1]          : GPU use (%): 0
GPU[0]          : VRAM Total Memory (B): 68702699520
GPU[0]          : VRAM Total Used Memory (B): 33554432000
GPU[1]          : VRAM Total Memory (B): 68702699520
GPU[1]          : VRAM Total Used Memory (B): 11288576
"""


def test_parse_smi_outputs():
    r = parse_rocm_smi(ROCM_SMI)
    assert r["0"] == {"use": 97, "total": 68702699520, "used": 33554432000}
    assert r["1"]["use"] == 0
    n = parse_nvidia_smi("0, 40000, 65536, 88\n1, 10, 65536, 0\n")
    assert n["0"] == {"used": 40000 * 2**20, "total": 65536 * 2**20, "use": 88}


def test_sampler_window_per_device(tmp_path):
    readings = iter([parse_rocm_smi(ROCM_SMI)] * 3)
    s = NodeSampler("rocm", log_path=str(tmp_path / "n.jsonl"), reader=lambda: next(readings))
    for _ in range(3):
        s.sample_once()
    w = s.window(0, 1e12, ["0", "1", "2"])
    assert w["0"]["vram_peak_mib"] == 32000 and w["0"]["use_mean"] == 97.0
    assert w["2"]["samples"] == 0
    assert len((tmp_path / "n.jsonl").read_text().splitlines()) == 3


# ── budget ───────────────────────────────────────────────────────────────


def test_lumi_billing():
    assert budget.billed_gpu_hours("standard-g", 3600, 1, {}) == 4
    one_gcd = {"gres/gpu": "1", "cpu": "7", "mem": "60G"}
    assert budget.billed_gpu_hours("small-g", 3600, 1, one_gcd) == 0.5
    # cores or memory beyond a GCD's share are billed as more GCDs
    assert budget.billed_gpu_hours("small-g", 3600, 1, dict(one_gcd, cpu="16")) == 1.0
    assert budget.billed_gpu_hours("small-g", 3600, 1, dict(one_gcd, mem="128G")) == 1.0
    assert budget.billed_gpu_hours("small", 3600, 1, one_gcd) == 0


def test_parse_sacct():
    text = ("123|eullm-campaign|standard-g|RUNNING|7200|1|cpu=56,gres/gpu=8,mem=480G,node=1\n"
            "124|eullm-bench|small-g|COMPLETED|3600|1|cpu=7,gres/gpu=1,mem=60G,node=1\n"
            "125|x|small-g|PENDING|0|1|\n")
    jobs = budget.parse_sacct(text)
    assert [j["gpu_hours"] for j in jobs] == [8.0, 0.5, 0.0]


def test_sacct_with_times_says_what_the_last_day_spent_and_what_burns():
    text = ("1|eullm-campaign|standard-g|COMPLETED|172800|1|cpu=56,gres/gpu=8,node=1|"
            "2026-10-08T10:00:00|2026-10-10T10:00:00\n"
            "2|eullm-campaign|standard-g|RUNNING|21600|1|cpu=56,gres/gpu=8,node=1|"
            "2026-10-11T04:00:00|Unknown\n"
            "3|eullm-concurrency|small-g|COMPLETED|3600|1|cpu=7,gres/gpu=1,mem=60G|"
            "2026-10-10T20:00:00|2026-10-10T21:00:00\n"
            "4|x|standard-g|PENDING|0|1||Unknown|Unknown\n")
    jobs = budget.parse_sacct(text)
    assert [j["rate"] for j in jobs] == [4.0, 4.0, 0.5, 4.0]
    assert jobs[1]["end"] is None and jobs[3]["start"] is None
    now = dt.datetime(2026, 10, 11, 10, 0)
    # The last day: none of the first (ended at 10:00 the day before), the
    # second's 6 hours so far, the small-g hour.
    assert budget.spent_between(jobs, now - dt.timedelta(days=1), now) == 24.0 + 0.5
    assert budget.burning(jobs) == 4.0
    start, end = dt.date(2026, 9, 12), dt.date(2027, 3, 12)
    # Six nodes around the clock against a 24.9 a day line: 400 behind in 3.4 days.
    rec = budget.recovery(4500, start, end, 400.0, 6 * 24)
    assert rec["line_node_hours_per_day"] == 24.9 and rec["catch_up_days"] == 3.4
    assert budget.recovery(4500, start, end, 400.0, 20)["catch_up_days"] is None
    assert budget.recovery(4500, start, end, -5.0, 20)["catch_up_days"] == 0.0


def test_pace_against_the_calendar():
    start, end = dt.date(2026, 9, 12), dt.date(2027, 3, 12)
    p = budget.pace(4500, start, end, dt.datetime(2026, 10, 12), 0)
    assert 15 < p["calendar_pct"] < 18
    assert p["behind_node_hours"] == p["target_to_date"] > 700
    assert 1.1 < p["needed_nodes_continuous"] < 1.3


def test_budget_after_the_end_date_reports_instead_of_crashing(capsys, monkeypatch):
    """pace() rightly returns None for the needed rates once no hours are
    left, but cmd_budget formatted them with :8.1f -- TypeError, killing a
    spend report that prints fine. The rates line now says the allocation
    is over."""
    monkeypatch.setattr(campaign.budget, "read_sacct", lambda start, account=None: [])
    args = SimpleNamespace(start="2026-09-12", end="2026-10-01", account=None,
                           budget_node_hours=4500.0, json=False)
    assert campaign.cmd_budget(args) == 0
    out = capsys.readouterr().out
    assert "days left" in out and "spent " in out
    assert "allocation ended 2026-10-01; nothing left to pace" in out


LUMI_ALLOCATIONS = """Data updated: 2026-10-21 09:38:55
  Project               |   CPU (used/allocated)|   GPU (used/allocated)|  Storage (used/allocated)
  ---------------------------------------------------------------------------------------
  project_465003366     |  0/1000 (0.0%) core/hours|  4135/18000 (23.0%) gpu/hours|  \
1341/90000 (1.5%) TB/hours
"""


def test_lumi_allocations_are_read_for_the_project():
    billed = budget.parse_allocations(LUMI_ALLOCATIONS, "project_465003366")
    assert billed == {"used": 4135.0, "allocated": 18000.0, "updated": "2026-10-21 09:38:55"}
    assert budget.parse_allocations(LUMI_ALLOCATIONS, "project_000000000") is None
    assert budget.parse_allocations(LUMI_ALLOCATIONS)["used"] == 4135.0


def test_budget_prints_what_is_billed(capsys, monkeypatch, tmp_path):
    """The spend on what is billed, as the other sites report it: allocation,
    billed, the straight line to date, what is left and the rate a day."""
    monkeypatch.setattr(campaign.budget, "read_sacct", lambda start, account=None: [])
    alloc = tmp_path / "alloc.txt"
    alloc.write_text(LUMI_ALLOCATIONS)
    args = SimpleNamespace(start="2026-09-12", end="2027-03-12", account="project_465003366",
                           budget_node_hours=4500.0, json=False, allocations=str(alloc))
    assert campaign.cmd_budget(args) == 0
    out = capsys.readouterr().out
    assert "--- billed (lumi-allocations, updated 2026-10-21 09:38:55) ---" in out
    assert "18000" in out and "4500.0" in out and "4135" in out and "1033.8" in out
    assert "13865" in out and "3466.2" in out
    assert "needed a day" in out and "nodes around the clock" in out


def test_the_day_the_allocation_runs_out_at_a_rate():
    """Spent by need, the allocation can end before its date: the day it runs
    out at the last day's rate is when more hours have to be asked for."""
    now = dt.datetime(2026, 10, 21, 12, 0)
    end = dt.date(2027, 3, 12)
    # 3,466 node-hours left at four nodes around the clock (96 a day): 36 days.
    out = budget.runs_out(3466.2, 96.0, now, end)
    assert out["date"] == "2026-11-26"
    assert 105 < out["days_early"] < 106
    # At the straight line's rate or slower it lasts to the end.
    assert budget.runs_out(3466.2, 24.0, now, end) == {"date": None, "days_early": None}
    assert budget.runs_out(3466.2, 0.0, now, end) == {"date": None, "days_early": None}


def test_the_billed_table_says_when_it_runs_out(capsys):
    billed = {"used": 4135.0, "allocated": 18000.0, "updated": "2026-10-21 09:38:55"}
    now = dt.datetime(2026, 10, 21, 12, 0)
    campaign.print_billed(billed, dt.date(2026, 9, 12), dt.date(2027, 3, 12), now, 96.0)
    out = capsys.readouterr().out
    assert "runs out on 2026-11-26, 105 days before the end" in out
    campaign.print_billed(billed, dt.date(2026, 9, 12), dt.date(2027, 3, 12), now, 20.0)
    assert "lasts to the end" in capsys.readouterr().out
    # A day of four nodes in a week of one and a half: the week's date is the
    # one to go by, and both are shown.
    campaign.print_billed(billed, dt.date(2026, 9, 12), dt.date(2027, 3, 12), now, 96.0, 36.0)
    out = capsys.readouterr().out
    assert "last 7 days, a day" in out and "144.0" in out
    assert "at the last 24 h" in out and "runs out on 2026-11-26" in out
    assert "at the last 7 days" in out and "runs out on 2027-01-25" in out
    assert "ask for more hours by 2027-01-25" in out


def test_shipped_campaigns_expand():
    """The specs in tools/lumi/campaigns are valid, and every model named in
    one is either a catalog id, covered by the spec's own `pull` list, or the
    `-text` twin of one of those its `text_only` names (text_only_model.sh)."""
    import glob
    import json

    import campaign

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    with open(os.path.join(root, "catalog", "v1", "catalog.json")) as f:
        catalog = {m["id"] for m in json.load(f)["models"]}
    paths = sorted(glob.glob(os.path.join(root, "tools", "lumi", "campaigns", "*.json")))
    assert paths
    for path in paths:
        with open(path) as f:
            spec = json.load(f)
        points = expand(spec)
        assert points, path
        pulled = {campaign.hf_ref_to_id(r) for r in spec.get("pull", [])}
        twins = set(spec.get("text_only", []))
        assert twins <= catalog | pulled, f"{path}: text_only names a model with no source"
        pulled |= {m + "-text" for m in twins}
        served = {p["model"] for p in points if p["kind"] != "finetune"}
        unknown = served - catalog - pulled
        assert not unknown, f"{path}: models with no source: {sorted(unknown)}"
        # A model a finetune point trains is converted from the repo the
        # spec's f32 map names, and its data is a set prefetch writes.
        for f, repo in campaign.f32_refs([(path, spec)]):
            assert repo and repo.count("/") == 1, f"{path}: {f} has no repo to convert"
        for p in points:
            if p["kind"] == "finetune":
                name = p["data"].rsplit(".jsonl", 1)[0]
                assert name in campaign.FINETUNE_SETS, f"{path}: {p['data']} is not prefetched"


def test_a_runner_leaves_points_it_cannot_run_in_the_queue():
    import point

    assert point.can_run({"kind": "throughput"})
    assert not point.can_run({"kind": "throughput", "runner": point.RUNNER_VERSION + 1})
    assert not point.can_run({"kind": "something-new"})
    assert not point.can_run({"kind": "throughput", "runtime": "unknown-server"})


def test_a_point_can_take_the_cores_of_every_device_it_reserved():
    import point

    node = [str(d) for d in range(8)]
    every = set(range(64))
    # One GCD's seven cores, or with "cores": "reserved" the node's 56 for an
    # exclusive point whose experts the CPU computes.
    assert campaign.point_cores("lumi", ["0"], node, False, every) == "49-55"
    assert campaign.point_cores("lumi", ["0"], node, True, every) == (
        "49-55,57-63,17-23,25-31,1-7,9-15,33-39,41-47")
    assert campaign.point_cores("lumi", ["0"], node, True, set(range(49, 56))) is None
    assert campaign.point_cores("none", ["0"], node, True, every) is None
    wide = normalize({"model": "m", "exclusive": True, "cores": "reserved"})
    assert wide["runner"] == 6 and point.can_run(wide)
    assert not point.can_run(wide | {"runner": point.RUNNER_VERSION + 1})
    assert "cores" not in normalize({"model": "m"})
    with pytest.raises(SpecError):
        normalize({"model": "m", "cores": "all"})


def test_a_point_gives_its_servers_the_environment_it_names(tmp_path, monkeypatch):
    import point

    started = []

    class Recorder:
        def __init__(self, binary, port, args, env, log, cores=None, runtime="eullm"):
            started.append(env)

        def start(self):
            pass

    monkeypatch.setattr(point, "Server", Recorder)
    p = normalize({"model": "m", "env": {"LLAMA_MOE_CACHE_STATS": 64}})
    assert p["env"] == {"LLAMA_MOE_CACHE_STATS": "64"} and p["runner"] == 7
    ctx = point.Context("/bin/eullm", "rocm", "none", ["3"], 0, str(tmp_path))
    point.start_servers(p | {"id": "p1"}, ctx)
    assert started[0]["LLAMA_MOE_CACHE_STATS"] == "64"
    assert "env" not in normalize({"model": "m"})
    with pytest.raises(SpecError):
        normalize({"model": "m", "env": ["LLAMA_MOE_CACHE_STATS=64"]})


def test_a_point_names_a_file_of_its_own_in_its_environment(tmp_path, monkeypatch):
    import point

    started = []

    class Recorder:
        def __init__(self, binary, port, args, env, log, cores=None, runtime="eullm"):
            started.append(env)

        def start(self):
            pass

    monkeypatch.setattr(point, "Server", Recorder)
    p = normalize({"model": "m", "env": {"LLAMA_MOE_CACHE_TRACE": "{workdir}/{point}.moe-trace",
                                         "LLAMA_MOE_CACHE_STATS": "64"}})
    # an older runner would pass the braces on
    assert p["runner"] == 8 and point.RUNNER_VERSION >= 8
    assert normalize({"model": "m", "env": {"LLAMA_MOE_CACHE_STATS": "64"}})["runner"] == 7
    ctx = point.Context("/bin/eullm", "rocm", "none", ["3"], 0, str(tmp_path))
    point.start_servers(p | {"id": "p1"}, ctx)
    assert started[0]["LLAMA_MOE_CACHE_TRACE"] == f"{tmp_path}/p1.moe-trace"
    assert started[0]["LLAMA_MOE_CACHE_STATS"] == "64"


def test_a_point_planned_for_one_engine_build_runs_only_in_its_jobs():
    import point

    labelled = normalize({"model": "m", "engine_label": "next3"})
    assert labelled["runner"] == 4 and labelled["engine_label"] == "next3"
    # A decision point planned for a build is measured for its choices too,
    # which a runner of before would leave out: it waits for one that can.
    decision = normalize({"model": "m", "kind": "decision", "engine_label": "next6"})
    assert decision["runner"] == 5
    assert not point.can_run(decision | {"runner": point.RUNNER_VERSION + 1}, "next6")
    assert normalize({"model": "m", "kind": "decision"})["runner"] == 2
    assert "engine_label" not in normalize({"model": "m", "engine_label": ""})
    plain = normalize({"model": "m"})
    assert point.can_run(labelled, "next3")
    assert not point.can_run(labelled, "next2")
    assert not point.can_run(labelled, "")  # a job submitted without a label
    assert point.can_run(plain, "")
    assert not point.can_run(plain, "next3")
    # Planned for a build, a spec is new points: none done before is replaced.
    spec = {"campaign": "c", "groups": [{"name": "g", "set": {"model": "m"}}]}
    (before,) = expand(spec)
    (after,) = expand(spec, engine_label="next3")
    assert before["id"] != after["id"]
    assert after["engine_label"] == "next3" and "engine_label" not in before


def test_new_fields_leave_old_ids_alone_and_mark_their_points():
    old = normalize({"model": "m"})
    assert "runtime" not in old and "runner" not in old
    rt = normalize({"model": "m", "runtime": "llama-server"})
    assert rt["runner"] == 2 and rt["runtime_args"] == []
    dec = normalize({"kind": "decision", "model": "jev"})
    assert dec["runner"] == 2 and dec["decision_mode"] == "shared_prefix"
    assert dec["mode"] == "single" and dec["questions"] == 8
    cold = normalize({"model": "m", "cold": True})
    assert cold["runner"] == 3 and cold["cold"] is True
    assert normalize({"kind": "decision", "model": "jev", "cold": 1})["runner"] == 3
    for bad in ({"model": "m", "runtime": "vllm-maybe"},
                {"kind": "finetune", "model": "m", "data": "d", "cold": True},
                {"kind": "finetune", "model": "m", "data": "d", "runtime": "ollama"},
                {"kind": "decision", "model": "jev", "questions": 65},
                {"kind": "decision", "model": "jev", "decision_mode": "fast"}):
        with pytest.raises(SpecError):
            normalize(bad)


def test_answers_digest_ignores_noise_below_four_decimals():
    import point

    a = {"q": {"type": "choice", "choice": "x", "probabilities": {"x": 0.71234, "y": 0.28766}}}
    b = {"q": {"type": "choice", "choice": "x", "probabilities": {"x": 0.712341, "y": 0.287659}}}
    c = {"q": {"type": "choice", "choice": "y", "probabilities": {"x": 0.4, "y": 0.6}}}
    assert point.answers_digest(a) == point.answers_digest(b) != point.answers_digest(c)


def test_a_decision_is_the_same_when_its_choices_are_whatever_the_decimals():
    import point

    a = {"u": {"type": "noul", "noul": 0.81},
         "t": {"type": "choice", "choice": "tech", "probabilities": {"tech": 0.7, "legal": 0.3}},
         "s": {"type": "score", "score": 1.4, "probabilities": {"1": 0.6, "2": 0.4}}}
    moved = {"u": {"type": "noul", "noul": 0.79},
             "t": {"type": "choice", "choice": "tech",
                   "probabilities": {"tech": 0.66, "legal": 0.34}},
             "s": {"type": "score", "score": 1.45, "probabilities": {"1": 0.55, "2": 0.45}}}
    flipped = dict(moved, s={"type": "score", "score": 1.6, "probabilities": {"1": 0.4, "2": 0.6}})
    assert point.answers_digest(a) != point.answers_digest(moved)
    assert point.answers_choices(a) == point.answers_choices(moved)
    assert point.answers_choices(a) != point.answers_choices(flipped)
    # A noul across one half is a different decision.
    no = dict(a, u={"type": "noul", "noul": 0.4})
    assert point.answers_choices(a) != point.answers_choices(no)
    shift = point.largest_shift(point.answers_probabilities(a), point.answers_probabilities(moved))
    assert abs(shift - 0.05) < 1e-9
    # One question of three decided otherwise.
    assert point.decisions_changed(point.answers_decisions(a),
                                   point.answers_decisions(flipped)) == (1, 3)
    assert point.decisions_changed(point.answers_decisions(a),
                                   point.answers_decisions(moved)) == (0, 3)


def test_failed_points_can_be_retried_by_group(tmp_path):
    q = Queue(str(tmp_path))
    for pid in ("moe-1", "moe-2", "rt-1"):
        q.add({"id": pid, "kind": "throughput"})
        q.claim(pid, {"job": "j"})
        q.finish(pid, "failed", "first")
        q.claim(pid, {"job": "j"})
        q.finish(pid, "failed", "second")
    assert sorted(q.ids("failed")) == ["moe-1", "moe-2", "rt-1"]
    assert sorted(q.retry("moe-")) == ["moe-1", "moe-2"]
    p = q.load("todo", "moe-1")
    assert p["attempts"] == 0 and p["notes"] == ["first", "second"]
    assert q.ids("failed") == ["rt-1"]


def test_status_gives_one_line_per_group_and_cause_not_the_server_logs():
    from campaign import stopped_lines

    log = "\n2026-10-07T12:20:42Z  INFO eullm::api: Decision traces: off"
    points = [
        {"id": f"lc-{i}", "group": "long-context",
         "notes": ["PointError: warm-up failed on server 0: Prefill failed: "
                   "Prompt (36491 tokens) does not fit in context window (32768)" + log]}
        for i in range(3)
    ] + [
        {"id": "m-1", "group": "moe-480b",
         "notes": ["TimeoutExpired: Command '['taskset', '-c', '49-55', 'eullm', 'serve']' "
                   "timed out after 900 seconds"]},
        {"id": "m-2", "group": "moe-480b",
         "notes": ["PointError: server on port 18000 not ready in 900 s:" + log]},
    ]
    lines = stopped_lines("failed", points)
    assert lines[0].startswith("  failed 5, by group and cause")
    assert "  3  long-context: PointError: warm-up failed" in lines[1]
    assert lines[1].endswith("does not fit in context window (32768)")
    assert any(line.endswith("moe-480b: TimeoutExpired: Command '…' timed out after 900 seconds")
               for line in lines)
    assert any(line.endswith("moe-480b: PointError: server on port 18000 not ready in 900 s")
               for line in lines)
    assert len(lines) == 4 and not any("INFO" in line for line in lines)

    each = stopped_lines("failed", points, each=True)
    assert len(each) == 5 and each[0].startswith("  failed lc-0 (long-context): PointError")
    assert stopped_lines("blocked", []) == []


def test_a_load_records_the_file_system_its_model_is_on_through_links(tmp_path, monkeypatch):
    import point

    store, flash = tmp_path / "store", tmp_path / "flash"
    (store / "m").mkdir(parents=True)
    flash.mkdir()
    (flash / "m-00001-of-00002.gguf").write_bytes(b"GGUF")
    (store / "m" / "m-00001-of-00002.gguf").symlink_to(flash / "m-00001-of-00002.gguf")
    (store / "m" / "manifest.json").write_text('{"gguf_file": "m-00001-of-00002.gguf"}')
    monkeypatch.setenv("EULLM_MODELS_DIR", str(store))
    root = "/" + os.path.realpath(tmp_path).split("/")[1]
    assert point.model_storage({"model": "m"}) == root
    assert point.model_storage({"model": "absent"}) is None
    monkeypatch.setenv("OLLAMA_MODELS", str(flash))
    assert point.model_storage({"model": "absent", "runtime": "ollama"}) == root
    assert point.mount_root("/scratch/project_1/someone/eullm-models/x.gguf") == "/scratch"


def test_a_cold_point_drops_every_part_of_its_model_from_the_page_cache(tmp_path, monkeypatch):
    import point

    d = tmp_path / "store" / "m"
    d.mkdir(parents=True)
    for i, size in ((1, 10), (2, 7)):
        (d / f"M-0000{i}-of-00002.gguf").write_bytes(b"x" * size)
    (d / "manifest.json").write_text('{"gguf_file": "M-00001-of-00002.gguf"}')
    monkeypatch.setenv("EULLM_MODELS_DIR", str(tmp_path / "store"))
    files = point.model_files({"model": "m"})
    assert [os.path.basename(f) for f in files] == ["M-00001-of-00002.gguf",
                                                    "M-00002-of-00002.gguf"]
    assert point.model_files({"model": "m", "runtime": "ollama"}) == []
    assert point.evict(files + [str(tmp_path / "gone.gguf")]) == 17

    ctx = point.Context("/bin/eullm", "rocm", "none", ["0"], 0, str(tmp_path))
    assert point.cache_state({"model": "m"}, ctx) == "cold"
    ctx.model_seen.add("m")
    assert point.cache_state({"model": "m"}, ctx) == "warm"
    assert point.cache_state({"model": "m", "cold": True}, ctx) == "evicted"


def test_the_report_puts_what_a_group_varies_beside_what_it_measured():
    import report

    rows = [
        {"campaign": "c06", "group": "mtp", "kind": "throughput", "outcome": "measured",
         "model": "m", "batch": "1", "extra_args": "--fit-strict", "agg_tok_s_mean": "100"},
        {"campaign": "c06", "group": "mtp", "kind": "throughput", "outcome": "measured",
         "model": "m", "batch": "1", "extra_args": "--fit-strict --mtp 2",
         "agg_tok_s_mean": "130"},
        {"campaign": "c06", "group": "mtp", "kind": "throughput", "outcome": "measured",
         "model": "m", "batch": "1", "extra_args": "--fit-strict --mtp 2",
         "agg_tok_s_mean": "134"},
        {"campaign": "c06", "group": "mtp", "kind": "throughput", "outcome": "does-not-fit",
         "model": "m", "batch": "1", "extra_args": "--fit-strict --mtp 3"},
        {"campaign": "c08", "group": "dec", "kind": "decision", "outcome": "measured",
         "model": "jev", "concurrency": "4", "dec_per_s": "12.5"},
    ]
    text = report.report(rows, {("c06", "mtp"): 2}, {"c06"})
    assert "=== c06 / mtp (throughput, 3 measured) ===" in text
    lines = text.splitlines()
    head = next(line for line in lines if "tok/s" in line)
    assert head.split()[0] == "extra_args"  # model and batch do not vary
    assert any("--mtp 2" in line and "132" in line and line.rstrip().endswith("2")
               for line in lines)  # the two results averaged, counted
    assert any("(none)" in line and "100" in line for line in lines)
    assert "not measured: does-not-fit 1, failed 2" in text
    assert "c08" not in text


def test_the_report_tells_points_apart_by_their_environment():
    import campaign
    import report

    def result(env, tok_s):
        return campaign.row_of({"campaign": "c10", "group": "evict", "kind": "throughput",
                                "outcome": "measured",
                                "params": {"model": "m", "batch": 32, "env": env},
                                "throughput": {"aggregate_tok_s_mean": tok_s}})

    rows = [result({"LLAMA_MOE_CACHE_STATS": "64", "LLAMA_MOE_CACHE_EVICT": "lru"}, 12),
            result({"LLAMA_MOE_CACHE_STATS": "64", "LLAMA_MOE_CACHE_TRACE": "{workdir}/t"}, 36)]
    lines = report.report(rows, {}, {"c10"}).splitlines()
    assert lines[1].split()[0] == "env"
    # one row each, without the variable both set alike
    assert any("LLAMA_MOE_CACHE_EVICT=lru" in line and "12" in line for line in lines)
    assert any("LLAMA_MOE_CACHE_TRACE=" in line and "36" in line for line in lines)
    assert not any("STATS" in line for line in lines)


def test_leftover_servers_are_those_of_this_job_outside_every_running_point():
    import campaign

    ps = ("  101   101 eullm\n  202   202 eullm\n  303   300 ollama\n"
          "  404   404 python3\n  505   505 llama-server\n  606   606 eullm\n")
    cgroups = {101: "0::/job_7/step_batch", 202: "0::/job_7/step_batch",
               303: "0::/job_7/step_batch", 404: "0::/job_7/step_batch",
               505: "0::/job_7/step_batch", 606: "0::/job_8/step_batch"}
    got = campaign.orphan_servers(ps, keep_pgids={101}, job="7", cgroup_of=cgroups.get)
    # 101 is a running point's server, 404 is no server, 606 is another job's.
    assert got == [(202, "eullm"), (303, "ollama"), (505, "llama-server")]


def test_collect_averages_throughput_latency_across_repeats():
    """summary.csv carries means: one stalled last repeat (ttft 500 after 50
    and 52) must not decide the ttft/decode/prefill the report compares
    engines on, while the headline aggregate already is a mean."""
    r = {"params": {"model": "m"}, "campaign": "c", "group": "g", "point": "p",
         "kind": "throughput", "outcome": "measured", "job": "j", "host": "h",
         "devices": [], "load": {},
         "throughput": {"aggregate_tok_s_mean": 100.0, "aggregate_tok_s_cv_pct": 1.0,
                        "repeats": [
                            {"ttft_ms_p50": 50.0, "decode_tok_s_mean": 100.0,
                             "prefill_tok_s_mean": 500.0},
                            {"ttft_ms_p50": 52.0, "decode_tok_s_mean": 102.0,
                             "prefill_tok_s_mean": 510.0},
                            {"ttft_ms_p50": 500.0, "decode_tok_s_mean": 20.0,
                             "prefill_tok_s_mean": 100.0}]},
         "device_stats": {}}
    row = campaign.row_of(r)
    # point.mean rounds like every other stored number (r1)
    assert row["ttft_ms_p50"] == pytest.approx(200.7)
    assert row["decode_tok_s"] == pytest.approx(74.0)
    assert row["prefill_tok_s"] == pytest.approx(370.0)


def test_collect_keeps_a_single_repeat_latency_as_is():
    """One repeat has no variance to average away: the row keeps its values,
    None where the repeat did not measure."""
    r = {"params": {}, "campaign": "c", "group": "g", "point": "p",
         "kind": "throughput", "outcome": "measured", "job": "j", "host": "h",
         "devices": [], "load": {},
         "throughput": {"aggregate_tok_s_mean": 90.0, "aggregate_tok_s_cv_pct": None,
                        "repeats": [{"ttft_ms_p50": 60.0}]},
         "device_stats": {}}
    row = campaign.row_of(r)
    assert row["ttft_ms_p50"] == 60.0
    assert row["decode_tok_s"] is None and row["prefill_tok_s"] is None


def _measured(group, model, round_, ended, decode, job="1", tok_s=None, **kw):
    return dict(campaign="c01", group=group, kind="throughput", outcome="measured",
                model=model, batch="16", round=round_, ended=ended, job=job,
                decode_tok_s=str(decode), agg_tok_s_mean=str(tok_s or decode), **kw)


def test_a_round_is_compared_with_the_last_earlier_measurement_of_each_configuration():
    """The before and after of an engine build: each configuration of the new
    round against the latest round that measured it before, not an older one,
    named in a line a group, with the slowest configuration by name."""
    import report

    rows = [
        _measured("dense", "a", "r-old", "2026-10-10T00:00:00", 50),
        _measured("dense", "a", "r-prev", "2026-10-15T00:00:00", 100),
        _measured("dense", "a", "r-new", "2026-10-21T00:00:00", 110),
        _measured("dense", "a", "r-new", "2026-10-21T01:00:00", 110),
        _measured("dense", "b", "r-prev", "2026-10-15T00:00:00", 200),
        _measured("dense", "b", "r-new", "2026-10-21T00:00:00", 140),
        _measured("dense", "c", "r-new", "2026-10-21T00:00:00", 70),
        _measured("moe", "d", "r-prev", "2026-10-15T00:00:00", 40),
        _measured("moe", "d", "r-new", "2026-10-21T00:00:00", 41),
        dict(_measured("moe", "e", "r-new", "2026-10-21T00:00:00", 0), outcome="failed"),
    ]
    text = report.compare(rows, lambda r: r["round"] == "r-new")
    lines = text.splitlines()
    dense = next(line for line in lines if "c01 / dense" in line)
    # 1.10 and 0.70: the median, 0.90, is below 0.95
    assert dense.startswith("[!!] ")
    assert "2 compared, decode new/old 0.90 (0.70-1.10), tok/s 0.90" in dense
    assert "slowest b 0.70 (200 -> 140, r-prev)" in dense
    assert "1 first measured" in dense
    moe = next(line for line in lines if "c01 / moe" in line)
    assert not moe.startswith("[!!]") and "1 compared, decode new/old 1.02" in moe
    assert lines[-1].startswith("all: 3 configurations compared, new/old median 1.02 (0.70-1.10)")
    assert "1 below 0.95" in lines[-1]


def test_a_throughput_point_is_read_by_its_decode_speed_first():
    """With a long prompt a point's tokens/s is mostly how many tokens the
    model wrote before stopping: Qwen3.6-35B-A3B's went from 32.1 to 13.6
    between two rounds at the same decode speed. Decode decides, tokens/s
    stands beside it, and a workload is still read by its tokens/s."""
    import report

    rows = [
        _measured("long", "q", "r-prev", "2026-10-15T00:00:00", 92.2, tok_s=32.1),
        _measured("long", "q", "r-new", "2026-10-21T00:00:00", 89.7, tok_s=13.6),
        dict(campaign="c01", group="soak", kind="workload", outcome="measured", model="q",
             round="r-prev", ended="2026-10-15T00:00:00", agg_tok_s_mean="100"),
        dict(campaign="c01", group="soak", kind="workload", outcome="measured", model="q",
             round="r-new", ended="2026-10-21T00:00:00", agg_tok_s_mean="102"),
    ]
    text = report.compare(rows, lambda r: r["round"] == "r-new")
    long_ = next(line for line in text.splitlines() if "c01 / long" in line)
    assert not long_.startswith("[!!]")
    assert "decode new/old 0.97 (0.97-0.97), tok/s 0.42" in long_
    assert "c01 / soak: 1 compared, tok/s new/old 1.02" in text


def test_a_job_is_compared_with_what_came_before_it_only():
    """A job's rows against measurements that ended before them: a later
    round of the same configuration is not its 'before'."""
    import report

    rows = [
        _measured("dense", "a", "r1", "2026-10-10T00:00:00", 100, job="10"),
        _measured("dense", "a", "r2", "2026-10-12T00:00:00", 120, job="20"),
        _measured("dense", "a", "r3", "2026-10-14T00:00:00", 90, job="30"),
    ]
    text = report.compare(rows, lambda r: r["job"] == "20")
    assert "1 compared, decode new/old 1.20" in text and "r1" in text
    assert report.compare(rows[:1], lambda r: r["job"] == "10").endswith(
        "nothing to compare: no configuration measured before")


def test_a_job_asks_for_the_time_its_points_need_not_48_hours():
    one_gcd = [dict(normalize({"model": f"m{i}", "kind": "workload", "sets": ["gsm8k"]}),
                    est_s=1500) for i in range(8)]
    alone = [dict(normalize({"model": f"m{i}", "exclusive": True}), est_s=1200) for i in range(8)]
    assert campaign.walltime_s([]) == 0
    # c06's mtp-ab: eight 25-minute points side by side on one node.
    assert campaign.walltime_s(one_gcd) == 2 * 3600
    # c10's ram-default: eight 20-minute points one after another.
    assert campaign.walltime_s(alone) == 5 * 3600
    # Shared by two jobs, each asks for less; never more than Slurm gives.
    assert campaign.walltime_s(alone, nodes=2) == 3 * 3600
    assert campaign.walltime_s(alone * 20) == campaign.MAX_WALLTIME_S
    # Four GCDs of small-g take the one-GCD points in two waves.
    assert campaign.walltime_s(one_gcd, node_devices=4) == 3 * 3600


def test_a_job_is_timed_from_how_long_its_points_took_before(tmp_path, capsys):
    import json

    from fsqueue import Queue

    spec = {"campaign": "c-time", "defaults": {"gcds": 1},
            "groups": [{"name": "g", "priority": 5, "est_s": 3600,
                        "set": {"kind": "workload", "sets": ["gsm8k"], "limit": 300},
                        "axes": {"model": [f"m{i}" for i in range(8)]}}]}
    queue = str(tmp_path / "q")
    q = Queue(queue)
    for p in expand(spec, "r-new"):
        q.add(p)
    # Estimated at an hour each, eight side by side: 4 hours asked.
    assert campaign.main(["walltime", "--queue", queue]) == 0
    assert capsys.readouterr().out.strip() == "4:00:00"
    # The same configurations took ten minutes in an earlier round.
    os.makedirs(os.path.join(queue, "results", "c-time"))
    for p in expand(spec, "r-old"):
        params = {k: v for k, v in p.items() if k not in campaign.PARAM_SKIP}
        with open(os.path.join(queue, "results", "c-time", f"{p['id']}.1.json"), "w") as f:
            json.dump({"schema": campaign.SCHEMA, "campaign": "c-time", "group": "g",
                       "params": dict(params, engine_label="old", duration_s=0),
                       "elapsed_s": 600}, f)
    assert campaign.main(["walltime", "--queue", queue]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "1:00:00"
    assert "8 of 8 points timed from an earlier run" in out.err


def test_a_long_round_goes_as_chains_of_short_jobs():
    alone = [dict(normalize({"model": f"m{i}", "exclusive": True}), est_s=3600)
             for i in range(40)]
    # 40 node-hours over two chains: 8-hour jobs, four in each.
    assert campaign.chain_plan(alone, nodes=2, max_hours=8) == (8 * 3600, 4)
    # Work that fits one short job stays one job, asked for what it needs.
    assert campaign.chain_plan(alone[:2], max_hours=8) == (5 * 3600, 1)
    assert campaign.chain_plan([], max_hours=8) == (0, 0)
    # A soak that stretches to a day counts for a job's time, not the day:
    # three 8-hour jobs (one for it, the rest margin that ends at once), not nine.
    soak = dict(normalize({"model": "s", "kind": "workload", "sets": ["gsm8k"],
                           "exclusive": True, "min_duration_s": 7200,
                           "max_duration_s": 86400}))
    assert campaign.chain_plan([soak], max_hours=8) == (8 * 3600, 3)


def test_a_server_log_is_read_as_utf8(tmp_path):
    """`Server.log_tail` reads the engine's log as UTF-8: the last lines of a
    failure are shown as written, not as mojibake, on any host locale."""
    import point

    log = tmp_path / "server.log"
    log.write_text("errore: modello non trovato a citt\u00e0\n", encoding="utf-8")
    server = point.Server("/bin/eullm", 18080, [], {}, str(log))
    assert "citt\u00e0" in server.log_tail()

def test_a_result_file_is_read_as_utf8(tmp_path):
    """`collect` reads a result's JSON as UTF-8, like `measured_s` does: a
    group or model named in non-ASCII text comes through unchanged."""
    import json

    results = tmp_path / "results"
    (results / "c").mkdir(parents=True)
    (results / "c" / "p.j1.json").write_text(json.dumps(
        {"schema": campaign.SCHEMA, "campaign": "c", "group": "gr\u00fcppe",
         "point": "p"}, ensure_ascii=False), encoding="utf-8")
    [row] = campaign.result_rows(str(results))
    assert row["group"] == "gr\u00fcppe"

def test_a_campaign_spec_is_read_as_utf8(tmp_path):
    """A spec is written by hand and may name a campaign or a group in
    non-ASCII text: it is read as UTF-8, not with the host's locale."""
    import json

    spec = tmp_path / "citta.json"
    spec.write_text(json.dumps({"campaign": "città", "groups": []},
                               ensure_ascii=False), encoding="utf-8")
    [(path, loaded)] = campaign.load_specs([str(spec)])
    assert path == str(spec) and loaded["campaign"] == "città"


def test_store_gguf_reads_a_manifest_of_utf8_text(tmp_path, monkeypatch):
    """A model whose manifest names non-ASCII text still resolves its
    `gguf_file`: the store's manifest is UTF-8, whatever the host locale."""
    import json

    import point

    d = tmp_path / "models" / "m"
    d.mkdir(parents=True)
    (d / "a.gguf").write_bytes(b"GGUF first")
    (d / "z\u00c1.gguf").write_bytes(b"GGUF named")
    (d / "manifest.json").write_text(json.dumps(
        {"name": "Mod\u00e8le", "gguf_file": "z\u00c1.gguf"},
        ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("EULLM_MODELS_DIR", str(tmp_path / "models"))
    # The sorted fallback would pick a.gguf: only the manifest names z\u00c1.gguf.
    assert point.store_gguf("m") == os.path.join(str(d), "z\u00c1.gguf")
