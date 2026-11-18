"""The watcher and what it waits for, run together as they run on Leonardo.

On 2026-09-27 the open-book generator and `submit_when_ready.sh` were each
tested alone: the generator wrote its ``.done`` marker, the watcher waited for
a file. Nobody ran one after the other. The marker was empty, the watcher
counts only non-empty files, and stage 3 waited all night behind a check that
said "not yet" every twenty minutes. These tests hand the watcher the file the
real producer writes, with `squeue`/`sbatch` stubbed on PATH, so a producer
and its consumer cannot drift apart again without a red test.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from eullm_forge.datasets.openbook_gen import make_openbook_jobs
from eullm_forge.eval import NormIndex

LEONARDO = Path(__file__).resolve().parents[1] / "scripts" / "leonardo"
GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "generate_openbook_pairs.py"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")


def _exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def cluster(tmp_path):
    """A fake scheduler: an empty queue, an sbatch that records, and a chain
    submitter that records instead of submitting."""
    home = tmp_path / "leonardo"
    home.mkdir()
    shutil.copy(LEONARDO / "submit_when_ready.sh", home)
    _exe(home / "submit_chain.sh", '#!/usr/bin/env bash\necho "$*" >> "$CALLS.chain"\n')
    (home / "sbatch_stage3.slurm").write_text("#!/bin/bash\n#SBATCH --job-name=eullm-stage3\n", encoding="utf-8")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    # The queue holds whatever $QUEUED names, as `squeue -n X -o %i` would.
    _exe(bin_ / "squeue", '#!/usr/bin/env bash\nfor a in "$@"; do\n'
                          '  [ "$prev" = "-n" ] && [ "$a" = "${QUEUED:-}" ] && echo 777\n'
                          '  prev="$a"\ndone\nexit 0\n')
    _exe(bin_ / "sbatch", '#!/usr/bin/env bash\necho "$*" >> "$CALLS.sbatch"\necho 4242\n')
    calls = tmp_path / "calls"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "CALLS": str(calls),
           "WAIT_HOME": str(home)}

    def watch(*need: Path, name: str = "") -> tuple[subprocess.CompletedProcess, str, str]:
        args = [a for f in need for a in ("--need", str(f))] + (["--name", name] if name else [])
        now = {**env, **({"QUEUED": os.environ["QUEUED"]} if "QUEUED" in os.environ else {})}
        run = subprocess.run(["bash", str(home / "submit_when_ready.sh"), *args, "--",
                              "sbatch_stage3.slurm", "2"],
                             cwd=tmp_path, env=now, capture_output=True, text=True, encoding="utf-8", timeout=60)
        chain = Path(f"{calls}.chain")
        queued = Path(f"{calls}.sbatch")
        return (run, chain.read_text(encoding="utf-8") if chain.exists() else "",
                queued.read_text(encoding="utf-8") if queued.exists() else "")

    return watch


FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3
RECORDS = [{"code": "codice_civile", "article_num": "", "chunk_index": 0,
            "text": f"Art. {n}. \n \n (Rubrica {n}). \n \n Disciplina {n}.{FILLER}"}
           for n in range(1, 41)]


def _generate_until_done(tmp_path: Path) -> Path:
    """Run the real generator to the end: every teacher job already answered
    by an earlier link, so it only has to say it is done."""
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    out = tmp_path / "openbook-pairs.jsonl"
    with open(tmp_path / "openbook-pairs.jsonl.rejected.jsonl", "w", encoding="utf-8") as f:
        for j in make_openbook_jobs(NormIndex(RECORDS), 10, seed=0):
            if j.kind == "grounded":
                f.write(json.dumps({"key": j.key, "reason": "x"}) + "\n")
    spec = importlib.util.spec_from_file_location("generate_openbook_pairs", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--norms", str(norms), "--no-exam", "--out", str(out),
                     "--limit", "10"]) == 0
    return out.with_name(out.name + ".done")


def test_the_generators_done_marker_starts_stage_3(tmp_path, cluster):
    marker = _generate_until_done(tmp_path)
    run, chain, queued = cluster(marker)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "submitting eullm-stage3" in run.stdout
    assert "sbatch_stage3.slurm 2" in chain
    assert not queued  # no second watcher


def test_an_empty_file_is_named_as_empty_and_does_not_start_anything(tmp_path, cluster):
    marker = tmp_path / "openbook-pairs.jsonl.done"
    marker.touch()
    run, chain, queued = cluster(marker)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "present but EMPTY" in run.stdout
    assert not chain
    assert "wait-eullm-stage3" in queued  # it will look again


def test_a_file_that_never_appeared_is_not_yet(tmp_path, cluster):
    run, chain, _ = cluster(tmp_path / "nothing.done")
    assert "not yet:" in run.stdout and "EMPTY" not in run.stdout
    assert not chain


def test_a_named_chain_is_not_mistaken_for_another_of_the_same_script(tmp_path, cluster,
                                                                   monkeypatch):
    """Another model's exam round is queued under the script's job name; a
    round with a name of its own is still submitted, under that name."""
    marker = _generate_until_done(tmp_path)
    monkeypatch.setenv("QUEUED", "eullm-stage3")
    run, chain, _ = cluster(marker, name="eullm-stage3-8b")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "submitting eullm-stage3-8b" in run.stdout
    assert "sbatch_stage3.slurm 2 -J eullm-stage3-8b" in chain


def test_an_unnamed_chain_still_defers_to_its_script_name(tmp_path, cluster, monkeypatch):
    marker = _generate_until_done(tmp_path)
    monkeypatch.setenv("QUEUED", "eullm-stage3")
    run, chain, _ = cluster(marker)
    assert "already in the queue" in run.stdout and not chain


def test_a_named_watcher_queues_itself_with_its_name(tmp_path, cluster):
    run, _, queued = cluster(tmp_path / "nothing.done", name="eullm-stage3-8b")
    assert "-J wait-eullm-stage3-8b" in queued and "--name eullm-stage3-8b" in queued
