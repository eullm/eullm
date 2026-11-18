"""Every Leonardo GPU job asks for the shape that gets placed.

The rules in forge/CLAUDE.md ("Leonardo: the shape of a job that starts")
were each measured once and then broken again by the next new script:
24 h jobs and whole-node jobs were found to wait for days on 2026-09-21/23,
and on 2026-10-01 the new GRPO script asked for 24 h and four GPUs anyway and
sat in the queue all afternoon. A rule that lives only in prose is
re-learnt at the cost of a day each time; this one fails CI instead.

Static checks on the #SBATCH headers:

* ``boost_usr_prod`` jobs ask for at most two hours — chains of 2 h links,
  never one long job;
* at most three GPUs, with at most 8 cores and 123 GB of memory per GPU
  (a quarter of a node). Asking for a fourth GPU, 32 cores or more than
  three quarters of the memory makes the job need an idle whole node.

The scripts that already ask for a whole node belong to finished phases and
are listed below with the reason; a new script is not added to that list
without the same kind of reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

LEONARDO = Path(__file__).resolve().parents[1] / "scripts" / "leonardo"
GPU_PARTITION = "boost_usr_prod"
MAX_SECONDS = 2 * 3600
MAX_GPUS = 3
CORES_PER_GPU = 8
MEM_GB_PER_GPU = 123  # 494 GB of a node / 4

#: Whole-node scripts kept as they are. Each needs all four GPUs of one node
#: by construction (a sharded 30B teacher, or a measurement of the 4-GPU
#: case itself). Phase 1 and phase 2 ended on 2026-09-29.
WHOLE_NODE = {
    "sbatch_backfill_probe.slurm": "measures how long a whole-node request waits",
    "sbatch_p1_logprob_probe.slurm": "30B teacher sharded over four GPUs",
    "sbatch_p3_truncation.slurm": "30B teacher sharded over four GPUs",
    "sbatch_phase1.slurm": "30B MoE continued pretraining, ZeRO-3 over four GPUs",
    "sbatch_phase2.slurm": "30B teacher + student, four GPUs",
    "sbatch_phase2_cds.slurm": "30B teacher + student, four GPUs",
    "sbatch_phase2_split_graze.slurm": "split-node teacher/student, four GPUs",
}


#: The one-letter spellings of the options these checks read. sbatch takes
#: `-n 32` as readily as `--ntasks=32`; a checker that knows only the long
#: form lets the short one through unread.
SHORT_OPTIONS = {"N": "nodes", "n": "ntasks", "c": "cpus-per-task", "t": "time", "p": "partition"}

DIRECTIVE = re.compile(
    r"#SBATCH\s+(?:--(?P<long>[\w-]+)(?:(?:=|\s+)(?P<value>[^\s#]+))?"
    r"|-(?P<short>[A-Za-z])\s*(?P<short_value>[^\s#]+))"
)


def _headers(path: Path) -> dict[str, str]:
    """The #SBATCH options as sbatch reads them: `--name=value`, `--name
    value` or `-x value`, and a flag such as `--exclusive` as an empty value.
    sbatch stops at the first line that is not a comment or blank, so a
    directive written after a command asks for nothing and is not read."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            break
        m = DIRECTIVE.match(line)
        if not m:
            continue
        if m.group("long"):
            out[m.group("long")] = m.group("value") or ""
        else:
            short = m.group("short")
            out[SHORT_OPTIONS.get(short, short)] = m.group("short_value")
    return out


def _seconds(walltime: str) -> int:
    days = 0
    if "-" in walltime:
        d, walltime = walltime.split("-", 1)
        days = int(d)
        # days-hours[:minutes[:seconds]]: after the dash slurm reads hours
        # first, so "1-12" is a day and twelve hours.
        parts = [int(p) for p in walltime.split(":")]
        parts += [0] * (3 - len(parts))
    else:
        parts = [int(p) for p in walltime.split(":")]
        # A bare number is minutes, the way slurm reads it and the way status.sh
        # reads it. Left-padded to [0, N, 0] it was read as N *seconds*, so
        # --time=180 passed a check meant to keep jobs under two hours while
        # asking the partition for three.
        if len(parts) == 1:
            parts = [0, parts[0], 0]
        while len(parts) < 3:
            parts.insert(0, 0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def _mem_gb(mem: str) -> float:
    m = re.fullmatch(r"(\d+)([KMGT]?)B?", mem.upper())
    assert m, f"unreadable --mem={mem}"
    scale = {"K": 1 / 1024**2, "M": 1 / 1024, "": 1 / 1024, "G": 1, "T": 1024}
    return int(m.group(1)) * scale[m.group(2)]


def _nodes(headers: dict[str, str]) -> int:
    """Nodes the request spans. --gres and --mem are per node, so everything
    this file reasons about is a per-node number times this."""
    return int(headers.get("nodes", "1"))


def _cores(headers: dict[str, str]) -> int:
    """Cores the request asks for, whichever of the two ways it asks.

    --ntasks counts the whole job, --ntasks-per-node one node of it. Reading
    only the second is what let --ntasks=32 through on three GPUs: the cores
    computed as 1, against a limit of 24.
    """
    per_task = int(headers.get("cpus-per-task", "1"))
    if "ntasks" in headers:
        return per_task * int(headers["ntasks"])
    return per_task * int(headers.get("ntasks-per-node", "1")) * _nodes(headers)


def _gpus(headers: dict[str, str]) -> int:
    m = re.search(r"gpu:(\d+)", headers.get("gres", ""))
    per_node = int(m.group(1)) if m else 0
    return per_node * _nodes(headers)


GPU_SCRIPTS = sorted(p for p in LEONARDO.glob("*.slurm")
                     if _headers(p).get("partition") == GPU_PARTITION)


def test_there_are_gpu_scripts_to_check():
    assert len(GPU_SCRIPTS) >= 10


@pytest.mark.parametrize("script", GPU_SCRIPTS, ids=lambda p: p.name)
def test_two_hours_at_most(script):
    h = _headers(script)
    assert "time" in h, f"{script.name}: no --time, the partition default is 24 h"
    assert _seconds(h["time"]) <= MAX_SECONDS, (
        f"{script.name} asks for {h['time']}: chain 2 h links with "
        "submit_chain.sh instead of one long job")


@pytest.mark.parametrize("script", GPU_SCRIPTS, ids=lambda p: p.name)
def test_a_quarter_node_per_gpu_and_never_the_whole_node(script):
    if script.name in WHOLE_NODE:
        pytest.skip(WHOLE_NODE[script.name])
    h = _headers(script)
    gpus = _gpus(h)
    assert 1 <= gpus <= MAX_GPUS, (
        f"{script.name} asks for {gpus} GPUs: four is a whole node, which waits "
        "for an idle node; three is placed within the hour")
    cpus = _cores(h)
    assert cpus <= CORES_PER_GPU * gpus, (
        f"{script.name}: {cpus} cores for {gpus} GPUs makes the node exclusive")
    assert "mem" in h, f"{script.name}: no --mem, the default may be the whole node"
    assert _mem_gb(h["mem"]) * _nodes(h) <= MEM_GB_PER_GPU * gpus, (
        f"{script.name}: --mem={h['mem']} for {gpus} GPUs")


@pytest.mark.parametrize("script", GPU_SCRIPTS, ids=lambda p: p.name)
def test_never_an_exclusive_node(script):
    if script.name in WHOLE_NODE:
        pytest.skip(WHOLE_NODE[script.name])
    assert "exclusive" not in _headers(script), (
        f"{script.name} asks for --exclusive: a whole node, which waits for an idle one")


def test_every_spelling_of_a_directive_is_read(tmp_path):
    """sbatch reads the short and the spaced forms, and a flag with no value;
    so does the check, or `-c 32` passes where `--cpus-per-task=32` fails."""
    p = tmp_path / "sbatch_case.slurm"
    p.write_text("#!/bin/bash\n#SBATCH -p boost_usr_prod\n#SBATCH -t 120\n#SBATCH -N 1\n"
                 "#SBATCH -n 32\n#SBATCH -c32\n#SBATCH --gres gpu:3\n#SBATCH --mem=340G\n"
                 "#SBATCH --exclusive\n#SBATCH --job-name=x  # trailing comment\n",
                 encoding="utf-8")
    assert _headers(p) == {
        "partition": "boost_usr_prod", "time": "120", "nodes": "1", "ntasks": "32",
        "cpus-per-task": "32", "gres": "gpu:3", "mem": "340G", "exclusive": "",
        "job-name": "x",
    }
    with pytest.raises(AssertionError):
        test_a_quarter_node_per_gpu_and_never_the_whole_node(p)
    with pytest.raises(AssertionError):
        test_never_an_exclusive_node(p)


def test_a_directive_after_the_first_command_is_not_read(tmp_path):
    """sbatch stops reading #SBATCH lines at the first command: a --time
    written below one leaves the job on the partition's 24 h default."""
    p = tmp_path / "sbatch_case.slurm"
    p.write_text("#!/bin/bash\n#SBATCH --partition=boost_usr_prod\n\nset -e\n"
                 "#SBATCH --time=02:00:00\n", encoding="utf-8")
    assert "time" not in _headers(p)
    with pytest.raises(AssertionError):
        test_two_hours_at_most(p)


def test_the_whole_node_list_only_shrinks():
    """An entry for a script that is gone, or no longer whole-node, is removed."""
    for name in WHOLE_NODE:
        path = LEONARDO / name
        assert path.is_file(), f"{name} is gone: drop it from WHOLE_NODE"
        assert _gpus(_headers(path)) == 4, f"{name} no longer asks for 4 GPUs"


def test_the_checks_catch_the_2026_10_01_grpo_script(tmp_path):
    bad = tmp_path / "sbatch_bad.slurm"
    bad.write_text("#!/bin/bash\n#SBATCH --partition=boost_usr_prod\n"
                   "#SBATCH --time=24:00:00\n#SBATCH --gres=gpu:4\n"
                   "#SBATCH --cpus-per-task=32\n#SBATCH --mem=450G\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        test_two_hours_at_most(bad)
    with pytest.raises(AssertionError):
        test_a_quarter_node_per_gpu_and_never_the_whole_node(bad)


def _script(tmp_path, body: str) -> Path:
    p = tmp_path / "sbatch_case.slurm"
    p.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    return p


def test_ntasks_is_a_way_to_ask_for_the_whole_node_and_is_counted(tmp_path):
    """--ntasks counts the whole job, --ntasks-per-node one node of it.

    Only the second was read, so the cores came out as 1 and 32 cores on
    three GPUs passed the check that exists to stop exactly that.
    """
    p = _script(tmp_path, "#SBATCH --partition=boost_usr_prod\n"
                          "#SBATCH --time=02:00:00\n#SBATCH --gres=gpu:3\n"
                          "#SBATCH --ntasks=32\n#SBATCH --mem=340G\n")
    with pytest.raises(AssertionError):
        test_a_quarter_node_per_gpu_and_never_the_whole_node(p)


def test_two_nodes_are_two_of_everything(tmp_path):
    """--gres and --mem are per node, so --nodes multiplies both.

    Two nodes of three GPUs is six, which is past the three-GPU ceiling the
    same check holds a one-node request to.
    """
    p = _script(tmp_path, "#SBATCH --partition=boost_usr_prod\n"
                          "#SBATCH --time=02:00:00\n#SBATCH --nodes=2\n"
                          "#SBATCH --gres=gpu:3\n#SBATCH --cpus-per-task=24\n"
                          "#SBATCH --mem=340G\n")
    with pytest.raises(AssertionError):
        test_a_quarter_node_per_gpu_and_never_the_whole_node(p)


def test_one_node_reads_the_same_as_before(tmp_path):
    """Every script in the tree asks for one node, so nothing moves."""
    p = _script(tmp_path, "#SBATCH --partition=boost_usr_prod\n"
                          "#SBATCH --time=02:00:00\n#SBATCH --nodes=1\n"
                          "#SBATCH --gres=gpu:3\n#SBATCH --cpus-per-task=16\n"
                          "#SBATCH --ntasks-per-node=1\n#SBATCH --mem=340G\n")
    h = _headers(p)
    assert _gpus(h) == 3 and _cores(h) == 16
    test_a_quarter_node_per_gpu_and_never_the_whole_node(p)


def test_a_bare_walltime_is_minutes(tmp_path):
    """Slurm reads --time=180 as 180 minutes, and status.sh does too.

    Left-padded to seconds it came out as three minutes, so --time=180
    passed a check meant to keep a link under two hours while asking the
    partition for three.
    """
    assert _seconds("180") == 180 * 60 == _seconds("03:00:00")
    assert _seconds("120") == _seconds("02:00:00")
    assert _seconds("7200") == 5 * 86400
    assert _seconds("1-12") == 36 * 3600 == _seconds("1-12:00:00")
    p = _script(tmp_path, "#SBATCH --partition=boost_usr_prod\n"
                          "#SBATCH --time=180\n#SBATCH --gres=gpu:3\n"
                          "#SBATCH --cpus-per-task=16\n#SBATCH --mem=340G\n")
    with pytest.raises(AssertionError):
        test_two_hours_at_most(p)
