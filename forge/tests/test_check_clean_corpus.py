"""check_clean_corpus.py: covered sets pass, uncovered ones filtered, damaged files caught."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_clean_corpus.py"


def _mod():
    spec = importlib.util.spec_from_file_location("check_clean_corpus", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w(p: Path, rows) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return p


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _row(art, forma="parafrasi"):
    return {"id": f"x|{art}|{forma}", "code": "codice_penale", "articolo": art, "forma": forma,
            "text": "t"}


def test_uncovered_sets_are_filtered_and_damage_caught(tmp_path, capsys):
    mod = _mod()
    covered = _w(tmp_path / "eval" / "norm-exam.jsonl",
                 [{"metadata": {"code": "codice_penale", "articolo": "1"}}])
    trap = _w(tmp_path / "eval" / "trap-test.jsonl",
              [{"metadata": {"code": "codice_penale", "articolo": "2"}}])
    clean = tmp_path / "clean"
    good = _w(clean / "v3s" / "rewrites.jsonl",
              [_row("1"), _row("2"), _row("2", "caso"), _row("3")])
    other = _w(clean / "v3s" / "program.jsonl", [_row("4")])
    (clean / "v3s" / "manifest.json").write_text(json.dumps(
        {"sets": {"norm-exam": {"sha256": _sha(covered)}},
         "files": {"rewrites.jsonl": _sha(good), "program.jsonl": "0" * 64}}))
    out = tmp_path / "ok"
    rc = mod.main(["--clean", str(clean), "--out", str(out),
                   "--sets", f"norm-exam={covered}", f"trap-test={trap}"])
    text = capsys.readouterr().out
    assert rc == 1                                  # program.jsonl does not match its sha256
    assert "norm-exam: covered" in text and "trap-test: NOT covered" in text
    kept = [json.loads(ln)["articolo"] for ln in (out / "v3s" / "rewrites.jsonl").read_text()
            .splitlines()]
    assert kept == ["1", "3"]                       # every form on the trap test's article goes
    assert not (out / "v3s" / "program.jsonl").exists()
    assert other.exists()
