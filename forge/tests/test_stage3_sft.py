"""`stage3_sft.py` stops when its run already finished.

A training chain has spare links. Before this, a spare link resumed from the
final checkpoint, trained zero steps and wrote the adapter again — possibly
under the package job merging it — which status.sh rightly called a link
that did no work (2026-09-30).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage3_sft.py"


def _load():
    spec = importlib.util.spec_from_file_location("stage3_sft", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_finished_run_is_not_trained_again(tmp_path, monkeypatch, capsys):
    mod = _load()
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")
    monkeypatch.setattr(mod, "fine_tune_identity",
                        lambda config: (_ for _ in ()).throw(AssertionError("trained again")))
    assert mod.main(["--model", "m", "--pairs", "p.jsonl", "--out", str(tmp_path)]) == 0
    assert "nothing left to do" in capsys.readouterr().out


def test_a_run_without_its_adapter_still_trains(tmp_path, monkeypatch):
    mod = _load()
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").touch()   # empty: not finished
    (tmp_path / "checkpoint-100").mkdir()
    seen = []
    monkeypatch.setattr(mod, "fine_tune_identity",
                        lambda config: seen.append(config.output_dir) or "x/adapter")
    assert mod.main(["--model", "m", "--pairs", "p.jsonl", "--out", str(tmp_path)]) == 0
    assert seen == [str(tmp_path)]
