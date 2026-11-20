"""The nonexistent-article OPD rows are the exam's prompt, with a teacher note."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

from eullm_forge.eval import NormIndex

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_nonexistent_prompts.py"


def _load():
    spec = importlib.util.spec_from_file_location("make_nonexistent_prompts", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(code, num, text):
    return {"text": f"Art. {num}. {text}", "code": code, "article_num": str(num),
            "chunk_index": 0}


def test_rows_ask_past_the_end_of_each_code_as_the_exam_does(tmp_path):
    mod = _load()
    index = NormIndex([_rec("codice_civile", 2968, "Prescrizione e decadenza."),
                       _rec("codice_civile", 2969, "Rilievo d'ufficio della decadenza."),
                       _rec("codice_penale", 734, "Distruzione di bellezze naturali.")])
    exam = tmp_path / "exam.jsonl"
    exam.write_text(json.dumps({"id": "norm-inesistente-codice_civile-3434", "metadata": {
        "tipo": "inesistente", "code": "codice_civile", "articolo": "3434"}}) + "\n", encoding="utf-8")
    rows = mod.rows_for(index, 30, mod.asked([exam]), random.Random(0))

    by_code = {}
    for r in rows:
        code, n = r["id"].split("-")[1], int(r["id"].split("-")[2])
        by_code.setdefault(code, []).append(n)
    assert set(by_code) == {"codice_civile", "codice_penale"}
    assert all(n > 2969 for n in by_code["codice_civile"]) and 3434 not in by_code["codice_civile"]
    assert all(n > 734 for n in by_code["codice_penale"])
    r = rows[0]
    assert r["kind"] == "nonexistent"
    student = r["student"][0]["content"]
    n = r["id"].split("-")[2]
    # the exam's own note and layout, so the student meets the prompt it is asked
    assert student.startswith(f"Nota: nella raccolta normativa non è presente art. {n}")
    assert "Testi normativi di riferimento:" in student and "Domanda: " in student
    teacher = r["teacher"][0]["content"]
    assert teacher.startswith("Nota riservata") and "probabilmente non esiste" in teacher
    assert teacher.endswith(student)


def test_the_script_appends_its_rows_after_the_mixed_ones(tmp_path):
    mod = _load()
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("".join(json.dumps(_rec("codice_civile", n, "Testo.")) + "\n"
                             for n in (1, 2, 3)), encoding="utf-8")
    mix = tmp_path / "prompts.jsonl"
    mix.write_text(json.dumps({"id": "c#1", "kind": "caselaw", "student": [],
                               "teacher": []}) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    assert mod.main(["--norms", str(norms), "--mix-with", str(mix), "--per-code", "5",
                     "--out", str(out)]) == 0
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["id"] == "c#1" and len(rows) == 6
    assert {r["kind"] for r in rows[1:]} == {"nonexistent"}
