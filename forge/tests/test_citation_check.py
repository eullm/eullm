"""citation_check.py: the articles an answer cites, against the question's."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "citation_check.py"


def _mod():
    spec = importlib.util.spec_from_file_location("citation_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _item(i, gruppo):
    code, art = gruppo[0].split(":")
    return {"id": f"q{i}", "domain": "legal", "lang": "it", "question": "?", "reference": "r",
            "metadata": {"code": code, "articolo": art, "gruppo": gruppo}}


def test_citations_are_read_with_their_code():
    mod = _mod()
    text = ("Secondo l'art. 1120 del codice civile e l'art. 1136 c.c., l'assemblea decide. "
            "Si veda anche l'art. 403 del codice di procedura civile.")
    assert mod.citations(text, "codice_civile") == {
        ("codice_civile", "1120"), ("codice_civile", "1136"), ("codice_procedura_civile", "403")}
    # No code named: the question's.
    assert mod.citations("Ai sensi degli artt. 1176 e 1375", "codice_civile") == {
        ("codice_civile", "1176"), ("codice_civile", "1375")}


def test_wrong_articles_are_counted_by_kind(tmp_path, capsys):
    items = tmp_path / "practical.jsonl"
    rows = [_item(1, ["codice_penale:196"]),
            _item(2, ["codice_processo_amministrativo:107"]),
            _item(3, ["codice_civile:1120", "codice_civile:1136"]),
            _item(4, ["codice_civile:2001", "codice_civile:2002"])]
    items.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    answers = tmp_path / "answers-practical-dev-m.jsonl"
    answers.write_text("".join(json.dumps(a) + "\n" for a in [
        {"id": "q1", "answer": "Secondo l'art. 197 del codice penale, l'ente paga."},   # near
        {"id": "q2", "answer": "Secondo l'art. 403 del codice di procedura civile, no."},  # code
        {"id": "q3", "answer": "Secondo l'art. 1120 del codice civile e l'art. 1136 c.c., sì."},
        {"id": "q4", "answer": "Il testo non lo dice."},                                 # none
    ]), encoding="utf-8")
    graded_dir = tmp_path / "graded"
    graded_dir.mkdir()
    (graded_dir / "answers-practical-dev-m.graded.jsonl").write_text("".join(
        json.dumps({"id": i, "grade": g}) + "\n"
        for i, g in [("q1", "wrong"), ("q2", "wrong"), ("q3", "correct"), ("q4", "partial")]), encoding="utf-8")
    assert _mod().main([str(items), str(answers), "--graded-dir", str(graded_dir)]) == 0
    out = capsys.readouterr().out
    row = next(ln for ln in out.splitlines() if ln.startswith("practical-dev-m")).split()
    # n cites on off near code far only-off
    assert row[1:9] == ["4", "3", "1", "2", "1", "1", "0", "2"]
    assert out.rstrip().endswith("0/1 / 0/1 / 2/2")
    assert "ente paga" not in out
