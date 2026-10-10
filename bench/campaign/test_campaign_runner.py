"""The runner end to end, against fake_eullm.py instead of a GPU."""

import argparse
import csv
import json
import os
import sys
import time

import pytest

import campaign
from devices import aligned_group
from fsqueue import Queue
from spec import SpecError, expand, normalize

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_eullm.py")


@pytest.fixture
def engine(tmp_path):
    """fake_eullm.py behind an executable path, as EULLM_BIN would be."""
    path = tmp_path / "eullm"
    path.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n")
    path.chmod(0o755)
    return str(path)


def run_args(queue, engine, **kw):
    a = dict(queue=queue, results=None, devices="0-3", backend="rocm", bind="none",
             site="test", engine=engine, walltime_s=3600, margin_s=0, port_base=0,
             poll_s=0.2, sample_s=0.5, settle_s=0.0, give_up_s=900.0)
    a.update(kw)
    return argparse.Namespace(**a)


def free_port_base():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return 20000 + s.getsockname()[1] % 20000


SPEC = {
    "campaign": "c-e2e",
    "defaults": {"repeats": 2, "num_predict": 12, "est_s": 60},
    "groups": [
        {"name": "one", "priority": 1, "est_s": 60, "set": {"gcds": 1},
         "axes": {"model": ["qwen3-8b", "qwen3-4b"], "batch": [1, 4]}},
        {"name": "rep", "priority": 1, "est_s": 60,
         "set": {"model": "qwen3-8b", "gcds": 2, "replica_gcds": 1, "batch": 2}},
        {"name": "excl", "priority": 2, "est_s": 60,
         "set": {"model": "qwen3-8b", "gcds": 1, "exclusive": True}},
        {"name": "gone", "priority": 0, "est_s": 60, "set": {"model": "missing-model"}},
        {"name": "huge", "priority": 0, "est_s": 60, "set": {"model": "huge-model"}},
        {"name": "load", "priority": 0, "est_s": 60, "set": {
            "kind": "workload", "model": "qwen3-4b", "sets": ["tiny"], "concurrency": 3,
            "min_duration_s": 0, "max_duration_s": 0, "interval_s": 1}},
    ],
}


def test_runner_drains_a_campaign(tmp_path, engine, capsys):
    qdir = str(tmp_path / "q")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(SPEC))
    assert campaign.main(["plan", str(spec_path), "--queue", qdir]) == 0
    os.makedirs(os.path.join(qdir, "sets"), exist_ok=True)
    with open(os.path.join(qdir, "sets", "tiny.jsonl"), "w") as f:
        for i in range(5):
            f.write(json.dumps({"id": f"t{i}", "answer": "4" if i < 4 else "5",
                                "grader": "number",
                                "messages": [{"role": "user", "content": "2+2?"}]}) + "\n")

    runner = campaign.Runner(run_args(qdir, engine, port_base=free_port_base()))
    runner.loop()

    counts = Queue(qdir).counts()
    assert counts == {"todo": 0, "running": 0, "done": 8, "failed": 0, "blocked": 1}
    out = capsys.readouterr().out
    results = [json.loads(line.split(" ", 1)[1]) for line in out.splitlines()
               if line.startswith("BENCH_RESULT ")]
    assert len(results) == 8
    by_group = {}
    for r in results:
        assert r["schema"] == "eullm.bench/1"
        assert r["ok"] == (r["group"] != "huge")
        assert r["engine"]["version"].startswith("eullm 0.0.0-fake")
        by_group.setdefault(r["group"], []).append(r)

    one = by_group["one"][0]["throughput"]
    assert len(one["repeats"]) == 2 and one["aggregate_tok_s_mean"] > 0
    assert one["repeats"][0]["ttft_ms_p50"] is not None
    rep = by_group["rep"][0]
    assert rep["params"]["replicas"] == 2 and len(rep["load"]["load_ms"]) == 2
    excl = by_group["excl"][0]
    assert excl["neighbours_at_start"] == 0  # exclusive: alone on the node
    # The first load of a model on this node is cold, later ones warm. (Two
    # points loading the same model at the same moment are both cold.)
    assert excl["load"]["cache"] == "cold"
    assert "warm" in {r.get("load", {}).get("cache") for r in results}

    # Too large for the devices is a result, the memory boundary: recorded
    # and done, not retried.
    assert by_group["huge"][0]["outcome"] == "does-not-fit"

    work = by_group["load"][0]["workload"]
    assert work["accuracy"]["tiny"] == {"graded": 5, "correct": 4, "accuracy": 0.8}
    assert work["requests"] == 5 and work["failed"] == 0
    answers = os.path.join(qdir, "results", "c-e2e",
                           f"{by_group['load'][0]['point']}.{runner.job}.answers.jsonl")
    assert len(open(answers).read().splitlines()) == 5

    summary = json.load(open(os.path.join(qdir, "results", f"{runner.job}.summary.json")))
    assert summary["points"]["done"] == 8 and summary["points"]["blocked"] == 1
    assert set(summary["assigned_fraction"]) == {"0", "1", "2", "3"}

    assert campaign.main(["collect", "--queue", qdir]) == 0
    rows = list(csv.DictReader(open(os.path.join(qdir, "results", "summary.csv"))))
    assert len(rows) == 8 and {r["kind"] for r in rows} == {"throughput", "workload"}
    assert {r["outcome"] for r in rows} == {"measured", "does-not-fit"}
    # What the engine was asked to do is in the row: the --load-threads of a
    # c09 point, the --fit-strict of the shipped specs.
    assert "extra_args" in rows[0] and "load_storage" in rows[0]
    capsys.readouterr()
    assert campaign.main(["report", "--queue", qdir]) == 0
    text = capsys.readouterr().out
    assert "=== c-e2e / one (throughput," in text and "tok/s" in text


def test_workload_stretches_to_the_time_it_is_given(tmp_path, engine):
    qdir = str(tmp_path / "q")
    q = Queue(qdir)
    spec = {"campaign": "c-soak", "groups": [{"name": "soak", "set": {
        "kind": "workload", "model": "qwen3-4b", "sets": ["tiny"], "concurrency": 2,
        "min_duration_s": 60, "max_duration_s": 3}}]}
    with pytest.raises(SpecError):  # max below min
        expand(spec)
    spec["groups"][0]["set"].update(min_duration_s=60, max_duration_s=7200)
    p = expand(spec)[0]
    q.add(p)
    r = campaign.Runner(run_args(qdir, engine, walltime_s=4 * 3600, margin_s=600))
    got = r.duration_for(p, r.deadline - time.time() - r.args.margin_s)
    assert got == 7200  # capped by its maximum
    got = r.duration_for(p, campaign.LOAD_ALLOWANCE_S + 1800)
    assert got == 1800  # stretched to what is free
    assert r.duration_for(p, campaign.LOAD_ALLOWANCE_S + 30) is None  # below its minimum


def test_backfill_does_not_delay_a_waiting_wide_point(tmp_path, engine):
    qdir = str(tmp_path / "q")
    spec = {"campaign": "c-bf", "groups": [
        {"name": "wide", "priority": 1, "est_s": 600, "set": {"model": "m", "gcds": 4}},
        {"name": "long", "priority": 0, "est_s": 7200, "set": {"model": "m", "gcds": 1}},
        {"name": "short", "priority": 0, "est_s": 300, "set": {"model": "n", "gcds": 1}},
    ]}
    todo = sorted(expand(spec), key=campaign.order_key)
    r = campaign.Runner(run_args(qdir, engine, walltime_s=10 * 3600))
    now = time.time()
    # Two devices are busy with a point ending in 1000 s: the wide point
    # must wait for them, so only what ends before then may start.
    busy = campaign.Running({"id": "x"}, [0, 1], [0, 1], 0, None, now + 1000)
    r.running["x"] = busy
    r.free -= {0, 1}
    p, use, reserved, duration = r.plan_next(now, todo)
    assert p["group"] == "short" and use == [2]


def test_time_left_parsing():
    assert campaign.parse_time_left("1-23:59:30") == 86400 + 23 * 3600 + 59 * 60 + 30
    assert campaign.parse_time_left("47:00:00") == 47 * 3600
    assert campaign.parse_time_left("59:30") == 59 * 60 + 30
    assert campaign.parse_time_left("UNLIMITED") is None


def test_stop_releases_the_running_points(tmp_path, engine):
    """The walltime case: SIGTERM sets `stop`; servers die at once and the
    interrupted point goes back to todo for the next job, not to failed."""
    import threading

    qdir = str(tmp_path / "q")
    q = Queue(qdir)
    spec = {"campaign": "c-stop", "groups": [{"name": "soak", "set": {
        "kind": "workload", "model": "qwen3-4b", "sets": ["tiny"], "concurrency": 2,
        "min_duration_s": 60, "max_duration_s": 3600}}]}
    p = expand(spec)[0]
    q.add(p)
    os.makedirs(os.path.join(qdir, "sets"))
    with open(os.path.join(qdir, "sets", "tiny.jsonl"), "w") as f:
        f.write(json.dumps({"id": "t0", "answer": "4", "grader": "number",
                            "messages": [{"role": "user", "content": "2+2?"}]}) + "\n")
    r = campaign.Runner(run_args(qdir, engine, port_base=free_port_base(),
                                 walltime_s=4 * 3600))
    t = threading.Thread(target=r.loop)
    t.start()
    for _ in range(100):
        if q.counts()["running"]:
            break
        time.sleep(0.1)
    time.sleep(1.0)  # the workload is under way
    servers = [s for run in r.running.values() for s in run.ctx.servers]
    assert servers
    r.stop.set()
    t.join(timeout=60)
    assert not t.is_alive()
    assert q.counts()["todo"] == 1 and q.counts()["running"] == 0
    assert all(s.proc.poll() is not None for s in servers)
    assert r.counts["released"] == 1


def test_hf_ids_follow_the_engine_rule():
    assert campaign.hf_ref_to_id("hf.co/Qwen/Qwen3-235B-A22B-GGUF:Q4_K_M") == \
        "qwen3-235b-a22b-gguf-q4_k_m"
    assert campaign.hf_ref_to_id("hf.co/ggml-org/gpt-oss-120b-GGUF:MXFP4") == \
        "gpt-oss-120b-gguf-mxfp4"
    assert campaign.hf_ref_to_id("hf.co/owner/Some Repo") == "some-repo"


def test_pulls_name_the_hf_ref_or_the_catalog_id(tmp_path, engine, capsys):
    spec = {"campaign": "c", "pull": ["hf.co/Qwen/Qwen3-8B-GGUF:Q8_0"], "groups": [
        {"name": "g", "axes": {"model": ["qwen3-8b-gguf-q8_0", "qwen3-14b", "qwen3-8b"]}}]}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(spec))
    assert campaign.main(["pulls", str(path), "--engine", engine]) == 0
    # fake_eullm lists qwen3-8b and qwen3-4b
    assert capsys.readouterr().out.split() == ["qwen3-14b", "hf.co/Qwen/Qwen3-8B-GGUF:Q8_0"]


def test_a_new_round_measures_everything_again(tmp_path, capsys):
    spec = {"campaign": "c", "groups": [{"name": "g", "axes": {"model": ["a", "b"]}}]}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(spec))
    q = str(tmp_path / "q")
    campaign.main(["plan", str(path), "--queue", q])
    campaign.main(["plan", str(path), "--queue", q])
    campaign.main(["plan", str(path), "--queue", q, "--round", "v0.7.21"])
    out = capsys.readouterr().out
    assert "2 points, 2 new" in out and "2 points, 0 new" in out
    assert "(round v0.7.21): 2 points, 2 new" in out
    assert Queue(q).counts()["todo"] == 4


def test_when_free_follows_alignment_and_overruns(tmp_path, engine):
    r = campaign.Runner(run_args(str(tmp_path / "q"), engine, devices="0-7"))
    now = time.time()
    # Free: 1, 2, 5, 6 — four devices, but no aligned block of four.
    r.free = {1, 2, 5, 6}
    for name, devs, end in (("a", [0], now + 100), ("b", [3], now + 200),
                            ("c", [4], now + 300), ("d", [7], now + 400)):
        r.running[name] = campaign.Running({"id": name}, devs, devs, 0, None, end)
    assert r.when_free(4, now) == now + 200  # 0-3 complete once a and b end
    assert r.when_free(8, now) == now + 400
    # A point past its estimate is given a quarter of it again, at least 5 min.
    late = campaign.Running({"id": "late"}, [1], [1], 0, None, now - 10)
    late.started = now - 4010
    r.running = {"late": late}
    r.free = {0, 2, 3, 4, 5, 6, 7}
    assert abs(r.when_free(2, now) - (now + 1000)) < 1


def test_finetune_points_train_block_and_record_the_boundary(tmp_path, engine, capsys):
    qdir = tmp_path / "q"
    spec = {"campaign": "c-ft", "groups": [
        {"name": "ft", "priority": 1, "est_s": 60,
         "set": {"kind": "finetune", "data": "text.jsonl", "epochs": 2, "ft_ctx": 256},
         "axes": {"model": ["tiny-f32.gguf", "huge-f32.gguf", "notf32-q4.gguf",
                            "absent-f32.gguf"]}},
        {"name": "keep", "priority": 0, "est_s": 60,
         "set": {"kind": "finetune", "data": "text.jsonl", "model": "tiny-f32.gguf",
                 "optimizer": "sgd", "keep_output": True}},
    ]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    assert campaign.main(["plan", str(spec_path), "--queue", str(qdir)]) == 0
    assert "absent-f32.gguf" in capsys.readouterr().out  # named as still to convert
    (qdir / "f32").mkdir()
    for name in ("tiny-f32.gguf", "huge-f32.gguf", "notf32-q4.gguf"):
        (qdir / "f32" / name).write_bytes(b"GGUF")
    (qdir / "sets").mkdir()
    (qdir / "sets" / "text.jsonl").write_text(json.dumps({"text": "Question: 2+2?"}) + "\n")

    runner = campaign.Runner(run_args(str(qdir), engine, port_base=free_port_base()))
    runner.loop()

    # A quantized or absent model waits for one that can be trained.
    assert Queue(str(qdir)).counts() == {"todo": 0, "running": 0, "done": 3, "failed": 0,
                                         "blocked": 2}
    out = capsys.readouterr().out
    results = {(r["params"]["model"], r["params"]["optimizer"]): r
               for r in (json.loads(line.split(" ", 1)[1]) for line in out.splitlines()
                         if line.startswith("BENCH_RESULT "))}
    assert len(results) == 3
    trained = results[("tiny-f32.gguf", "adamw")]
    assert trained["outcome"] == "measured" and trained["kind"] == "finetune"
    ft = trained["finetune"]
    assert ft["schema"] == "eullm.finetune/1" and ft["n_ctx"] == 256
    assert ft["model"] == str(qdir / "f32" / "tiny-f32.gguf")
    assert ft["data"] == str(qdir / "sets" / "text.jsonl")
    assert len(ft["per_epoch"]) == 2
    assert results[("huge-f32.gguf", "adamw")]["outcome"] == "does-not-fit"

    # The trained model is deleted unless the point keeps it.
    kept = results[("tiny-f32.gguf", "sgd")]["point"]
    assert [p.name for p in (qdir / "results").rglob("*.gguf")] == [f"{kept}.gguf"]

    assert campaign.main(["collect", "--queue", str(qdir)]) == 0
    rows = {r["point"]: r for r in csv.DictReader(open(qdir / "results" / "summary.csv"))}
    row = rows[trained["point"]]
    assert row["kind"] == "finetune" and row["ft_ctx"] == "256" and row["optimizer"] == "adamw"
    assert float(row["ft_loss_before"]) == 1.9
    assert float(row["ft_loss_after"]) == pytest.approx(1.3)
    assert float(row["ft_tok_s"]) == 4000.5
    assert row["ft_trainable_params"] == "440467456"
    assert row["lr"] == "1e-06" and row["epochs"] == "2" and row["ft_mem_est_mib"] == "9216"


def test_an_engine_without_finetune_blocks_the_points(tmp_path, capsys):
    old = tmp_path / "eullm"
    old.write_text("#!/bin/sh\n[ \"$1\" = --version ] && { echo 'eullm 0.7.20'; exit 0; }\n"
                   "echo \"error: unrecognized subcommand '$1'\" >&2\nexit 2\n")
    old.chmod(0o755)
    qdir = tmp_path / "q"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"campaign": "c-old", "groups": [
        {"name": "ft", "est_s": 60,
         "set": {"kind": "finetune", "data": "t.jsonl", "model": "m-f32.gguf"}}]}))
    assert campaign.main(["plan", str(spec_path), "--queue", str(qdir),
                          "--engine", str(old)]) == 0
    assert "has no `finetune` command" in capsys.readouterr().out
    (qdir / "f32").mkdir()
    (qdir / "f32" / "m-f32.gguf").write_bytes(b"GGUF")
    (qdir / "sets").mkdir()
    (qdir / "sets" / "t.jsonl").write_text(json.dumps({"text": "x"}) + "\n")

    campaign.Runner(run_args(str(qdir), str(old), port_base=free_port_base())).loop()
    counts = Queue(str(qdir)).counts()
    assert counts["blocked"] == 1 and counts["failed"] == 0 and counts["done"] == 0


def test_decision_points_queue_behind_one_worker_and_answer_the_same(tmp_path, engine, capsys,
                                                                     monkeypatch):
    qdir = tmp_path / "q"
    spec = {"campaign": "c-dec", "defaults": {"kind": "decision", "requests": 24,
                                              "distinct_states": 5, "questions": 4},
            "groups": [
        {"name": "one", "est_s": 60, "set": {"model": "jev-style-fake"},
         "axes": {"concurrency": [1, 4]}},
        {"name": "rep", "est_s": 60,
         "set": {"model": "jev-style-fake", "gcds": 2, "replica_gcds": 1, "concurrency": 4}},
        {"name": "gone", "est_s": 60, "set": {"model": "missing-jev"}},
    ]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    assert campaign.main(["plan", str(spec_path), "--queue", str(qdir)]) == 0
    monkeypatch.setenv("FAKE_EULLM_DELAY_S", "0.002")
    campaign.Runner(run_args(str(qdir), engine, port_base=free_port_base())).loop()

    assert Queue(str(qdir)).counts() == {"todo": 0, "running": 0, "done": 3, "failed": 0,
                                         "blocked": 1}
    out = capsys.readouterr().out
    results = [json.loads(line.split(" ", 1)[1]) for line in out.splitlines()
               if line.startswith("BENCH_RESULT ")]
    by = {(r["group"], r["params"]["concurrency"]): r for r in results}
    alone, queued = by[("one", 1)]["decision"], by[("one", 4)]["decision"]
    assert alone["requests"] == 24 and alone["failed"] == 0 and alone["decisions_per_s"] > 0
    # One worker: four clients wait behind each other, one does not.
    assert queued["wait_ms_p50"] > alone["wait_ms_p50"]
    # Each of the 5 states was asked several times, and answered the same.
    assert alone["consistency"] == {"compared": 19, "identical": 19, "rate": 1.0}
    assert by[("rep", 4)]["params"]["replicas"] == 2
    assert campaign.main(["collect", "--queue", str(qdir)]) == 0
    rows = list(csv.DictReader(open(qdir / "results" / "summary.csv")))
    assert {r["kind"] for r in rows} == {"decision"} and all(r["dec_per_s"] for r in rows)


def test_llama_server_and_ollama_are_measured_like_the_engine(tmp_path, engine, capsys,
                                                             monkeypatch):
    store = tmp_path / "models" / "qwen3-8b"
    store.mkdir(parents=True)
    (store / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"GGUF")
    (store / "manifest.json").write_text(json.dumps({"gguf_file": "Qwen3-8B-Q4_K_M.gguf"}))
    monkeypatch.setenv("EULLM_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LLAMA_SERVER_BIN", engine)
    monkeypatch.setenv("OLLAMA_BIN", engine)
    qdir = tmp_path / "q"
    spec = {"campaign": "c-rt", "defaults": {"repeats": 1, "num_predict": 12},
            "groups": [
        {"name": "rt", "est_s": 60, "set": {"model": "qwen3-8b", "batch": 2},
         "axes": {"runtime": ["eullm", "llama-server", "ollama"]}},
        {"name": "rt-chat", "est_s": 60,
         "set": {"kind": "workload", "model": "qwen3-8b", "sets": ["tiny"], "concurrency": 2,
                 "runtime": "llama-server"}},
        {"name": "rt-gone", "est_s": 60,
         "set": {"model": "not-in-store", "runtime": "llama-server"}},
    ]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    assert campaign.main(["plan", str(spec_path), "--queue", str(qdir)]) == 0
    (qdir / "sets").mkdir()
    with open(qdir / "sets" / "tiny.jsonl", "w") as f:
        for i in range(4):
            f.write(json.dumps({"id": f"t{i}", "answer": "4", "grader": "number",
                                "messages": [{"role": "user", "content": "2+2?"}]}) + "\n")
    campaign.Runner(run_args(str(qdir), engine, port_base=free_port_base())).loop()

    assert Queue(str(qdir)).counts()["done"] == 4
    assert Queue(str(qdir)).counts()["blocked"] == 1  # no GGUF to give llama-server
    out = capsys.readouterr().out
    results = {(r["group"], r["params"].get("runtime")): r
               for r in (json.loads(line.split(" ", 1)[1]) for line in out.splitlines()
                         if line.startswith("BENCH_RESULT "))}
    for rt in ("eullm", "llama-server", "ollama"):
        t = results[("rt", rt)]["throughput"]["repeats"][0]
        assert t["ok"] == 2 and t["generated_tokens"] > 0 and t["ttft_ms_p50"] is not None
    chat = results[("rt-chat", "llama-server")]["workload"]
    assert chat["accuracy"]["tiny"]["accuracy"] == 1.0  # translated to OpenAI and back


def test_devices_of_a_server_that_outlives_its_kill_wait_for_it(tmp_path, engine):
    class Proc:
        alive = True

        def poll(self):
            return None if self.alive else -9

    class Straggler:
        proc = Proc()

    class Ctx:
        stragglers = [Straggler()]

    class Run:
        p = {"id": "x-1"}
        reserved = [2, 3]
        started = time.time()
        ctx = Ctx()

    runner = campaign.Runner(run_args(str(tmp_path / "q"), engine, port_base=free_port_base()))
    runner.free -= {2, 3}
    runner.running["x-1"] = Run()
    runner.finished.append((Run(), "failed", "boom", time.time()))
    runner.reap()
    assert not {2, 3} & runner.free and runner.draining
    Straggler.proc.alive = False
    assert runner.reap() == 1
    assert {2, 3} <= runner.free and not runner.draining


def test_stop_does_not_raise_when_the_server_outlives_the_wait(tmp_path):
    import subprocess

    import point

    s = point.Server("sleep", 0, [], dict(os.environ), str(tmp_path / "log"))
    s.proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    assert s.stop(kill_wait_s=5) is True  # SIGTERM is enough for sleep
    assert s.stop() is True  # already gone


def _three_points(qdir, name):
    q = Queue(qdir)
    spec = {"campaign": name, "defaults": {"repeats": 1, "num_predict": 4},
            "groups": [{"name": "g", "est_s": 60, "set": {"gcds": 1},
                        "axes": {"model": ["qwen3-8b", "qwen3-4b", "qwen3-1.7b"]}}]}
    for p in expand(spec):
        q.add(p)


def test_a_device_just_given_back_is_judged_once_it_has_settled(tmp_path, engine, capsys,
                                                                 monkeypatch):
    """The memory of a point's server goes a moment after the point ends. A
    reading from that moment saw it as a leftover's, set the device aside,
    and a job with nothing else running ended after one point (09-10-2026)."""
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    qdir = str(tmp_path / "q")
    _three_points(qdir, "c-settle")
    r = campaign.Runner(run_args(qdir, engine, devices="0-0", port_base=free_port_base(),
                                 settle_s=0.6))

    def reader():
        just_freed = time.time() - r.freed_at.get(0, 0.0) < 0.4
        held = 30 * 2**30 if just_freed else 50 * 2**20
        return {"0": {"used": held, "total": 64 * 2**30, "use": 0}}

    r.sampler.reader = reader
    r.loop()
    out = capsys.readouterr().out
    assert "with no point on them" not in out
    assert sum(line.startswith("BENCH_RESULT ") for line in out.splitlines()) == 3


def test_a_job_waits_for_a_device_set_aside_then_goes_on_without_it(tmp_path, engine, capsys,
                                                                    monkeypatch):
    """With nothing else running, a job no longer ends while a device is set
    aside: it waits for it, and leaves it out once it has held memory for
    give_up_s, after the points the other devices could run."""
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    qdir = str(tmp_path / "q")
    _three_points(qdir, "c-give-up")
    r = campaign.Runner(run_args(qdir, engine, devices="0-1", port_base=free_port_base(),
                                 give_up_s=1.0))

    def reader():
        return {"0": {"used": 50 * 2**20, "total": 64 * 2**30, "use": 0},
                "1": {"used": 40 * 2**30, "total": 64 * 2**30, "use": 0}}

    r.sampler.reader = reader
    r.loop()
    out = capsys.readouterr().out
    assert "devices [1] hold 40960 MiB with no point on them" in out
    assert "devices [1] still hold memory after" in out
    assert sum(line.startswith("BENCH_RESULT ") for line in out.splitlines()) == 3
    assert r.devices == [0, 1] and r.lost == {1} and not r.quarantined


def test_a_device_holding_vram_with_no_point_on_it_is_set_aside(tmp_path, engine, capsys,
                                                                monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    qdir = str(tmp_path / "q")
    q = Queue(qdir)
    spec = {"campaign": "c-dirty", "defaults": {"repeats": 1, "num_predict": 4},
            "groups": [{"name": "g", "est_s": 60, "set": {"gcds": 1},
                        "axes": {"model": ["qwen3-8b", "qwen3-4b", "qwen3-1.7b"]}}]}
    for p in expand(spec):
        q.add(p)
    r = campaign.Runner(run_args(qdir, engine, devices="0-1", port_base=free_port_base()))
    calls = {"n": 0}

    def reader():
        # Device 1 holds 40 GiB, a leftover's, for the first few readings.
        calls["n"] += 1
        held = 40 * 2**30 if calls["n"] < 8 else 50 * 2**20
        return {"0": {"used": 100 * 2**20, "total": 64 * 2**30, "use": 0},
                "1": {"used": held, "total": 64 * 2**30, "use": 0}}

    r.sampler.reader = reader
    r.loop()
    out = capsys.readouterr().out
    assert "devices [1] hold 40960 MiB with no point on them" in out
    results = [json.loads(line.split(" ", 1)[1]) for line in out.splitlines()
               if line.startswith("BENCH_RESULT ")]
    assert len(results) == 3
    for res in results:
        assert res["vram_at_start_mib"] and max(res["vram_at_start_mib"].values()) <= 2048

    # Once the device holds nothing again it goes back to the free ones.
    r.free, r.quarantined = {0}, {1: 40960}
    r.sampler.samples.clear()  # the next reading is a fresh, clean one
    calls["n"] = 100
    r.reap()
    assert r.free == {0, 1} and not r.quarantined
    assert "devices [1] clean again" in capsys.readouterr().out


# devices.py documents the LUMI pairing: GCDs 0-1 sit on NUMA 3, 2-3 on 1,
# 4-5 on 0, 6-7 on 2.
NUMA_OF_GCD = {"0": "3", "1": "3", "2": "1", "3": "1",
               "4": "0", "5": "0", "6": "2", "7": "2"}


def test_a_given_up_device_keeps_its_slot_so_pairs_stay_on_one_module(tmp_path, engine,
                                                                      monkeypatch):
    """aligned_group strides self.devices, so positions in that list ARE the
    GCD pairing. Removing a lost device shifted every device after it onto a
    neighbour's NUMA domain: with 0 gone, a gcds=2 point was handed logical
    [1, 2] -- physical GCDs 1 and 2, NUMA 3 and 1, two MI250X modules with
    each other's core bindings. The slot stays; the lost device is only
    never free again, so its block is skipped and the rest stay aligned."""
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    qdir = tmp_path / "q"
    qdir.mkdir()
    r = campaign.Runner(run_args(str(qdir), engine, devices="0-7",
                                 port_base=free_port_base(), give_up_s=1.0))
    r.free -= {0}
    r.quarantined.update({0: 40960})
    r.quarantined_at[0] = time.time() - 3600
    r.sampler.reader = lambda: {
        "0": {"used": 40 * 2**30, "total": 64 * 2**30, "use": 0},
        **{str(d): {"used": 50 * 2**20, "total": 64 * 2**30, "use": 0}
           for d in range(1, 8)},
    }
    r.reap()

    assert r.devices == list(range(8)) and r.lost == {0} and not r.quarantined
    assert 0 not in r.free
    pair = normalize({"kind": "throughput", "model": "m", "gcds": 2, "batch": 1})
    used, reserved = r.devices_for(pair)
    assert reserved == [2, 3]
    phys = [r.physical[d] for d in reserved]
    assert {NUMA_OF_GCD[p] for p in phys} == {"1"}
    half = normalize({"kind": "throughput", "model": "m", "gcds": 4, "batch": 1})
    _, reserved_half = r.devices_for(half)
    assert reserved_half == [4, 5, 6, 7]
    assert len({NUMA_OF_GCD[p] for p in [r.physical[d] for d in reserved_half]}) == 2
