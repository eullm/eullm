"""teacher_pilot.py: the pilot's rewrites beside the old teacher's, judged by the others."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "teacher_pilot.py"
spec = importlib.util.spec_from_file_location("teacher_pilot", SCRIPT)
tp = importlib.util.module_from_spec(spec)
sys.modules["teacher_pilot"] = tp
spec.loader.exec_module(tp)


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def row(n, style, text="t"):
    return {"code": "codice_civile", "articolo": str(n), "forma": style, "text": text}


def test_the_old_teacher_is_taken_on_the_same_articles_only(tmp_path):
    write(tmp_path / "gemma" / "rewrites-caso.jsonl", [row(1, "caso"), row(2, "caso")])
    write(tmp_path / "gemma" / "rewrites-caso.jsonl.rejects.jsonl", [{"id": "x"}])
    write(tmp_path / "gptoss" / "rewrites-caso.jsonl", [row(1, "caso")])
    pool = tmp_path / "memory" / "rewrites-c.jsonl"
    write(pool, [row(1, "caso", "old"), row(1, "caso", "old again"), row(3, "caso"),
                 row(2, "parafrasi"), row(2, "caso", "old 2")])
    written = tp.collect(tmp_path, [pool])
    assert written == {"gemma-caso.jsonl": 2, "gptoss-caso.jsonl": 1, "qwen30b-caso.jsonl": 2}
    base = [json.loads(x)["text"] for x in (tmp_path / "all" / "qwen30b-caso.jsonl").open()]
    assert base == ["old", "old 2"]                  # one row an article, the first


def test_each_teacher_is_judged_without_itself_and_beside_the_old_one():
    items = [{"id": f"{t}-caso.jsonl:{n}", "forma": "caso"}
             for t in ("gemma", "qwen30b") for n in (1, 2)]

    def v(i, c, bad):
        return {"id": i, "checker": c, "verdict": "non fedele" if bad else "fedele"}

    verdicts = [
        v("gemma-caso.jsonl:1", "gemma4-31b", True),         # its own: not counted
        v("gemma-caso.jsonl:1", "gpt-oss-120b", False),
        v("gemma-caso.jsonl:1", "flashnext", True),
        v("gemma-caso.jsonl:2", "gpt-oss-120b", True),
        v("gemma-caso.jsonl:2", "flashnext", True),
        v("qwen30b-caso.jsonl:1", "gpt-oss-120b", True),
        v("qwen30b-caso.jsonl:1", "flashnext", True),
        v("qwen30b-caso.jsonl:2", "gpt-oss-120b", True),
        v("qwen30b-caso.jsonl:2", "flashnext", True),
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "verdicts.jsonl"
        p.write_text("".join(json.dumps(x) + "\n" for x in verdicts), encoding="utf-8")
        out = tp.report(items, [p])
    assert "gemma, judged by flashnext + gpt-oss-120b:" in out
    assert "gemma: 1/2 flagged by all (50%)" in out
    assert "qwen30b on the same articles: 2/2 (100%)" in out
