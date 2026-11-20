"""split_by_article.py: exam questions split by whether the corpus saw their article."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "split_by_article.py"
_spec = importlib.util.spec_from_file_location("split_by_article", _PATH)
sba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sba)


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_gain_only_where_the_corpus_saw_the_article(tmp_path):
    items = _write(tmp_path / "exam.jsonl", [
        {"id": "q1", "metadata": {"code": "cp", "articolo": "575"}},
        {"id": "q2", "code": "cp", "articolo": "624"},
        {"id": "q3", "code": "cc", "articolo": "2043"},
        {"id": "q4", "code": "cc", "articolo": "1453"},
    ])
    corpus = _write(tmp_path / "rewrites-caso-a.jsonl", [
        {"code": "cp", "articolo": "575", "forma": "caso"},
        {"code": "cp", "articolo": "624", "forma": "caso"},
        {"code": "cc", "articolo": "2043", "forma": "parafrasi"},   # not a counted form
    ])
    _write(tmp_path / "rewrites-caso-a.rejects.jsonl", [{"code": "cc", "articolo": "1453",
                                                         "forma": "caso"}])
    graded = tmp_path / "graded"
    graded.mkdir()
    _write(graded / "answers-base.graded.jsonl",
           [{"id": i, "grade": "wrong"} for i in ("q1", "q2", "q3", "q4")])
    _write(graded / "answers-model.graded.jsonl",
           [{"id": "q1", "grade": "correct"}, {"id": "q2", "grade": "correct"},
            {"id": "q3", "grade": "wrong"}, {"id": "q4", "grade": "wrong"}])
    covered = sba.corpus_articles([corpus, tmp_path / "rewrites-caso-a.rejects.jsonl"],
                                  {"caso", "spiegazione"})
    assert covered == {("cp", "575"), ("cp", "624")}
    models = {g.label: g for g in map(sba.load_graded, sorted(graded.glob("*.jsonl")))}
    lines = sba.report(models, sba.article_of_items(items), covered, "base")
    assert lines[0].startswith("A, article with corpus rows: 2 questions; B, without: 2")
    model_line = next(x for x in lines[2:] if x.startswith("model "))
    assert "2/2" in model_line and "0/2" in model_line
    assert "2:0" in model_line and "0:0" in model_line
