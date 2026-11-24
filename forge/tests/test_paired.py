"""Paired comparison of graded answers, and the blind review sheet."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from eullm_forge.eval.paired import (
    Graded,
    compare,
    human_agreement,
    kind_of,
    label_of,
    load_graded,
    mcnemar_p,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mcnemar_matches_the_exact_binomial():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) == 1.0
    # 10:0 -> 2 * 0.5**10
    assert mcnemar_p(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_p(0, 10) == mcnemar_p(10, 0)
    # 169 vs 160 on 207 questions, split 20:11 -- not significant on its own
    assert mcnemar_p(20, 11) > 0.1


def test_kind_and_label_are_read_from_the_names():
    assert kind_of("norm-termine_argomento-codice_civile-1456") == "termine_argomento"
    assert kind_of("norm-termine-codice_civile-1456") == "termine"
    assert kind_of("civ-001") == "?"
    assert label_of(Path("answers-v0.3-open.graded.jsonl")) == "v0.3-open"


def _graded(label, grades, lengths=None):
    g = Graded(label)
    g.grades = dict(grades)
    g.lengths = dict(lengths or {k: 100 for k in grades})
    return g


def test_compare_counts_only_the_questions_where_exactly_one_is_right():
    ids = [f"norm-contenuto-codice_civile-{i}" for i in range(6)]
    a = _graded("a", zip(ids, ["correct", "correct", "correct", "wrong", "partial", "correct"]))
    b = _graded("b", zip(ids, ["correct", "wrong", "wrong", "correct", "correct", "unparsed"]))
    c = compare(a, b)
    assert c.n == 5                      # the unparsed one is left out
    assert (c.a_only, c.base_only) == (2, 2)
    assert c.diff == 0
    # partial counts as right in the lenient split: item 4 is no longer b's alone
    assert (c.a_only_lenient, c.base_only_lenient) == (2, 1)
    assert c.by_kind == {"contenuto": (2, 2)}


def test_the_length_check_says_how_often_the_right_answer_was_the_longer():
    ids = [f"norm-termine-codice_penale-{i}" for i in range(4)]
    a = _graded("a", zip(ids, ["correct"] * 4), {i: 1200 for i in ids})
    b = _graded("b", zip(ids, ["wrong"] * 4), {i: 400 for i in ids})
    assert compare(a, b).longer_wins == 1.0
    assert compare(b, a).longer_wins == 1.0      # same items, seen from the other side


def test_equal_length_disagreements_are_in_neither_length_count():
    """A tie is no evidence of a length preference either way. Counted in the
    denominator only, it would drag longer_wins towards 0 and hide a judge
    that always prefers the longer answer. It is still a disagreement."""
    ids = [f"norm-termine-codice_penale-{i}" for i in range(2)]
    a = _graded("a", [(ids[0], "correct"), (ids[1], "wrong")],
                {ids[0]: 20, ids[1]: 15})
    b = _graded("b", [(ids[0], "wrong"), (ids[1], "correct")],
                {ids[0]: 10, ids[1]: 15})
    out = compare(a, b)
    assert sum(sum(v) for v in out.by_kind.values()) == 2
    assert out.longer_wins == 1.0


def test_human_agreement_counts_the_errors_that_move_a_comparison():
    m = {"a": _graded("a", {"x": "correct", "y": "wrong", "z": "correct", "w": "correct"})}
    rows = [{"chiave": "a|x", "giudizio": "corretto"},
            {"chiave": "a|y", "giudizio": "Corretto "},
            {"chiave": "a|z", "giudizio": "sbagliato"},
            {"chiave": "a|w", "giudizio": ""},              # not labelled yet
            {"chiave": "b|x", "giudizio": "corretto"}]      # a model not given
    h = human_agreement(rows, m)
    assert h["n"] == 3
    assert h["judge_too_kind"] == 1 and h["judge_too_harsh"] == 1
    assert h["same_label"] == pytest.approx(1 / 3)


def _write_graded(d: Path, label: str, rows: list[tuple[str, str, int]]):
    with (d / f"answers-{label}.graded.jsonl").open("w", encoding="utf-8") as f:
        for item, grade, n in rows:
            f.write(json.dumps({"id": item, "question": f"Q {item}", "reference": "R",
                                "answer": "x" * n, "grade": grade}) + "\n")


def test_the_script_prints_counts_never_items_and_writes_the_csv(tmp_path, capsys):
    d = tmp_path / "devbig"
    d.mkdir()
    ids = [f"norm-contenuto-codice_civile-{i}" for i in range(30)]
    _write_graded(d, "base-open", [(i, "wrong", 300) for i in ids])
    _write_graded(d, "new-open", [(i, "correct", 300) for i in ids])
    out = tmp_path / "paired.csv"
    assert _script("compare_graded").main([str(d), "--baseline", "base-open",
                                           "--csv", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "norm-contenuto" not in printed and "Q norm" not in printed
    row = next(csv.DictReader(out.open()))
    assert row["model"] == "new-open" and row["diff"] == "30" and row["verdict"] == "better"


def test_a_missing_baseline_is_an_error(tmp_path):
    d = tmp_path / "devbig"
    d.mkdir()
    _write_graded(d, "a", [("norm-contenuto-codice_civile-1", "correct", 10)])
    assert _script("compare_graded").main([str(d), "--baseline", "zzz"]) == 1


def test_the_review_sheet_is_blind_and_maps_back_to_the_judge(tmp_path):
    d = tmp_path / "devbig"
    d.mkdir()
    ids = [f"norm-termine-codice_civile-{i}" for i in range(20)]
    _write_graded(d, "a", [(i, "correct", 50) for i in ids])
    _write_graded(d, "b", [(i, "correct" if n % 2 else "wrong", 50) for n, i in enumerate(ids)])
    sheet = tmp_path / "review.csv"
    exporter = _script("export_grade_review")
    assert exporter.main([str(d), "--n", "8", "--out", str(sheet)]) == 0
    text = sheet.read_text(encoding="utf-8-sig")
    assert "correct" not in text and "wrong" not in text   # the judge is not shown
    assert "a|" not in text and "b|" not in text           # nor which model wrote it
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig"), delimiter=";"))
    assert len(rows) == 8
    keys = json.loads(exporter.keys_path(sheet).read_text(encoding="utf-8"))
    assert set(keys) == {r["chiave"] for r in rows}
    # disagreements first: 3/4 of the sample is from the 10 split questions
    split = {f"norm-termine-codice_civile-{i}" for i in range(0, 20, 2)}
    assert sum(keys[r["chiave"]].split("|")[1] in split for r in rows) >= 6

    # a person grades everything "corretto": the judge is too harsh wherever it said wrong
    for r in rows:
        r["giudizio"] = "corretto"
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        w.writerows(rows)
    models = {g.label: g for g in map(load_graded, sorted(d.glob("*.graded.jsonl")))}
    harsh = sum(models[k.split("|")[0]].grades[k.split("|")[1]] == "wrong"
                for k in keys.values())
    compare_graded = _script("compare_graded")
    assert compare_graded.main([str(d), "--baseline", "a", "--human", str(sheet)]) == 0
    sheet_rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig"), delimiter=";"))
    for r in sheet_rows:
        r["chiave"] = keys[r["chiave"]]
    assert human_agreement(sheet_rows, models)["judge_too_harsh"] == harsh


def test_answers_of_the_held_out_exam_are_not_exported(tmp_path):
    d = tmp_path / "norm-exam-answers-v3"
    d.mkdir()
    _write_graded(d, "a", [("norm-contenuto-codice_civile-1", "correct", 10)])
    assert _script("export_grade_review").main([str(d), "--out", str(tmp_path / "r.csv")]) == 2


def test_a_review_subset_holds_exactly_the_answers_on_the_sheet(tmp_path):
    src = tmp_path / "devbig"
    src.mkdir()
    for label in ("a", "b"):
        (src / f"answers-{label}.jsonl").write_text("".join(
            json.dumps({"id": f"norm-contenuto-codice_civile-{i}", "answer": f"{label}{i}"}) + "\n"
            for i in range(10)), encoding="utf-8")
    sheet = tmp_path / "review.csv"
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["n", "chiave", "giudizio"])
        w.writerows([[1, "r1", "corretto"], [2, "r2", "sbagliato"], [3, "r3", ""]])
    (tmp_path / "review.csv.keys.json").write_text(json.dumps({
        "r1": "a|norm-contenuto-codice_civile-3", "r2": "a|norm-contenuto-codice_civile-7",
        "r3": "b|norm-contenuto-codice_civile-0"}), encoding="utf-8")
    out = tmp_path / "devreview"
    subset = _script("review_subset")
    assert subset.main([str(sheet), "--answers", str(src), "--out", str(out)]) == 0
    a = [json.loads(x)["id"] for x in
         (out / "answers-a.jsonl").read_text(encoding="utf-8").splitlines()]
    b = [json.loads(x)["id"] for x in
         (out / "answers-b.jsonl").read_text(encoding="utf-8").splitlines()]
    assert a == ["norm-contenuto-codice_civile-3", "norm-contenuto-codice_civile-7"]
    assert b == ["norm-contenuto-codice_civile-0"]
    # not a development name: refused
    assert subset.main([str(sheet), "--answers", str(src), "--out",
                        str(tmp_path / "final-v3")]) == 2


def test_compare_without_a_baseline_still_measures_the_judge(tmp_path, capsys):
    d = tmp_path / "devreview"
    d.mkdir()
    _write_graded(d, "a", [("norm-contenuto-codice_civile-1", "wrong", 10)])
    sheet = tmp_path / "review.csv"
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["chiave", "giudizio"])
        w.writerow(["r1", "corretto"])
    (tmp_path / "review.csv.keys.json").write_text(
        json.dumps({"r1": "a|norm-contenuto-codice_civile-1"}), encoding="utf-8")
    assert _script("compare_graded").main([str(d), "--human", str(sheet)]) == 0
    out = capsys.readouterr().out
    assert "against" not in out and "judge vs person on 1 answers" in out


def test_deadline_and_absent_questions_are_scored_without_a_judge():
    from eullm_forge.eval.paired import verifiable

    ref = ("La domanda si propone entro sessanta giorni.\n\nTesto integrale dell'articolo: "
           "La domanda si propone entro sessanta giorni dalla notifica.")
    row = {"id": "norm-termine-codice_civile-2", "reference": ref,
           "rubric": "Corretto solo se indica il termine di 60 giorni."}
    assert verifiable({**row, "answer": "Il termine è di sessanta giorni."}) == 1.0
    assert verifiable({**row, "answer": "Entro 60 giorni dalla notifica."}) == 1.0
    assert verifiable({**row, "answer": "Il termine è di trenta giorni."}) == 0.0
    assert verifiable({**row, "answer": "L'articolo non esiste."}) == 0.0
    # an item drawn before the builder skipped two-deadline articles: either counts
    two = {"id": "norm-termine-codice_procedura_penale-554-ter",
           "rubric": "Corretto solo se indica il termine di 60 giorni.",
           "reference": "non oltre sessanta giorni\n\nTesto integrale dell'articolo: fissa "
                        "l'udienza non oltre sessanta giorni; un termine non inferiore a "
                        "venti giorni."}
    assert verifiable({**two, "answer": "Non inferiore a venti giorni."}) == 1.0
    absent = {"id": "norm-inesistente-codice_civile-3000"}
    assert verifiable({**absent, "answer": "L'art. 3000 non esiste nel codice."}) == 1.0
    assert verifiable({**absent, "answer": "Prevede il termine di trenta giorni."}) == 0.0
    assert verifiable({"id": "norm-contenuto-codice_civile-1", "answer": "x"}) is None


def test_the_comparison_carries_the_judge_free_split(tmp_path, capsys):
    d = tmp_path / "devbig"
    d.mkdir()
    ref = "x\n\nTesto integrale dell'articolo: si propone entro sessanta giorni."
    rub = "Corretto solo se indica il termine di 60 giorni."
    for label, answer, grade in (("a", "sessanta giorni", "correct"),
                                 ("b", "trenta giorni", "correct")):
        with (d / f"answers-{label}.graded.jsonl").open("w") as f:
            for i in range(8):
                f.write(json.dumps({"id": f"norm-termine-codice_civile-{i}", "answer": answer,
                                    "reference": ref, "rubric": rub, "grade": grade}) + "\n")
    assert _script("compare_graded").main([str(d), "--baseline", "b"]) == 0
    out = capsys.readouterr().out
    assert "8/8" in out and "0/8" in out
    # the judge saw no difference; the check without a judge sees all of it
    assert "no judge (deadlines, absent articles; n=8): 8:0 p=0.008" in out


def test_questions_of_excluded_rulings_are_left_out_of_every_model(tmp_path, capsys):
    """2026-10-07: 158 development rulings had passages in the OPD prompts."""
    d = tmp_path / "cds-exam"
    d.mkdir()
    for label, grades in (("base", "wwww"), ("opd", "ccww")):
        (d / f"answers-{label}.graded.jsonl").write_text("".join(
            json.dumps({"id": f"cds-20200000{i}-0", "ruling": f"cds/20200000{i}",
                        "answer": "x", "grade": {"c": "correct", "w": "wrong"}[g]}) + "\n"
            for i, g in enumerate(grades)), encoding="utf-8")
    seen = tmp_path / "seen.txt"
    seen.write_text("cds/202000000\ncds/202000001\n", encoding="utf-8")
    compare_graded = _script("compare_graded")
    assert compare_graded.main([str(d), "--baseline", "base"]) == 0
    assert "2:0" in capsys.readouterr().out
    assert compare_graded.main([str(d), "--baseline", "base",
                                "--exclude-rulings", str(seen)]) == 0
    out = capsys.readouterr().out
    assert "leaving out the questions of 2 rulings" in out and "n=2" in out and "0:0" in out


def test_show_prints_the_questions_each_side_won_with_both_answers(tmp_path, capsys):
    d = tmp_path / "devbig"
    d.mkdir()
    ids = [f"norm-inesistente-codice_civile-{i}" for i in range(6)]
    _write_graded(d, "base-open", [(i, "correct" if n < 4 else "wrong", 10)
                                   for n, i in enumerate(ids)])
    _write_graded(d, "new-open", [(i, "wrong" if n < 3 else "correct", 20)
                                  for n, i in enumerate(ids)])
    assert _script("compare_graded").main([str(d), "--baseline", "base-open",
                                           "--show", "2"]) == 0
    printed = capsys.readouterr().out
    assert "only the model right: 2 shown" in printed      # items 4 and 5
    assert "only the baseline right: 2 shown" in printed   # 2 of items 0-2
    assert f"== {ids[4]}" in printed and f"== {ids[0]}" in printed
    assert "new-open [correct]" in printed and "base-open [wrong]" in printed


def test_show_is_refused_on_the_held_out_exam(tmp_path):
    d = tmp_path / "norm-exam-answers"
    d.mkdir()
    _write_graded(d, "a", [("norm-contenuto-x-1", "correct", 5)])
    _write_graded(d, "b", [("norm-contenuto-x-1", "wrong", 5)])
    assert _script("compare_graded").main([str(d), "--baseline", "a", "--show", "1"]) == 2
