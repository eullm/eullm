"""remerge_round.py: a round's merged weights made again, ancestors first."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "remerge_round.py"


def load():
    spec = importlib.util.spec_from_file_location("remerge_round", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _round(grpo: Path, scratch: Path, name: str, start: Path, present: bool) -> Path:
    d = grpo / name
    (d / "adapter").mkdir(parents=True)
    (d / "adapter" / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(start)}), encoding="utf-8")
    (scratch / name).mkdir(parents=True, exist_ok=True)
    if present:
        (scratch / name / "config.json").write_text("{}", encoding="utf-8")
    (d / "merged").symlink_to(scratch / name)
    return d / "merged"


def test_a_round_is_merged_again_from_the_first_ancestor_still_there(tmp_path):
    grpo, scratch = tmp_path / "grpo", tmp_path / "scratch"
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    a = _round(grpo, scratch, "a", base, present=True)
    b = _round(grpo, scratch, "b", a, present=False)
    c = _round(grpo, scratch, "c", b, present=False)
    _round(grpo, scratch, "d", c, present=False)
    calls = []

    def merge(start, adapter, out):
        calls.append((Path(start), Path(adapter).parent.name, Path(out)))
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "config.json").write_text(json.dumps({"from": start}), encoding="utf-8")

    mod = load()
    assert mod.main([str(grpo / "d")], merge=merge) == 0
    assert [c[1] for c in calls] == ["b", "c", "d"]
    assert calls[0][0] == a                                 # the first one still there
    assert calls[1][0] == calls[0][2]                       # each on the one before
    assert calls[2][2] == scratch / "d"                     # back behind the symlink
    assert (grpo / "d" / "merged" / "config.json").is_file()
    assert sorted(p.name for p in scratch.iterdir()) == ["a", "b", "c", "d"]  # no leftovers
    # a second run finds it there
    calls.clear()
    assert mod.main([str(grpo / "d")], merge=merge) == 0 and not calls


def test_a_dry_run_only_prints_the_chain(tmp_path, capsys):
    grpo, scratch = tmp_path / "grpo", tmp_path / "scratch"
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    a = _round(grpo, scratch, "a", base, present=False)
    _round(grpo, scratch, "b", a, present=False)
    assert load().main([str(grpo / "b"), "--dry-run"],
                       merge=lambda *_: (_ for _ in ()).throw(AssertionError)) == 0
    out = capsys.readouterr().out
    assert out.index("  a  (") < out.index("  b  (")


def test_a_start_recorded_by_its_scratch_path_is_followed_too(tmp_path):
    grpo, scratch = tmp_path / "grpo", tmp_path / "scratch"
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    _round(grpo, scratch, "a", base, present=False)
    _round(grpo, scratch, "b", scratch / "a", present=False)   # the resolved path
    start, rounds = load().chain_of(grpo / "b")
    assert start == base and [r for r, _ in rounds] == ["a", "b"]


def test_an_sft_round_s_adapter_is_found_in_its_directory_or_given(tmp_path):
    # r2b-steps-8b, the short SFT at the root of the r2 line, has no NAME/adapter
    grpo, scratch = tmp_path / "grpo", tmp_path / "scratch"
    base = tmp_path / "published"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    sft = grpo / "root" / "sft" / "final"
    sft.mkdir(parents=True)
    (sft / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base)}), encoding="utf-8")
    (grpo / "root" / "sft" / "checkpoint-100").mkdir()
    (grpo / "root" / "sft" / "checkpoint-100" / "adapter_config.json").write_text(
        "{}", encoding="utf-8")
    (grpo / "root" / "merged").symlink_to(scratch / "root")
    _round(grpo, scratch, "top", grpo / "root" / "merged", present=False)
    mod = load()
    start, rounds = mod.chain_of(grpo / "top")
    assert start == base and rounds == [("root", sft), ("top", grpo / "top" / "adapter")]
    # elsewhere: given by name
    other = tmp_path / "elsewhere" / "lora"
    other.mkdir(parents=True)
    (other / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base)}), encoding="utf-8")
    start, rounds = mod.chain_of(grpo / "top", {"root": other})
    assert rounds[0] == ("root", other)
