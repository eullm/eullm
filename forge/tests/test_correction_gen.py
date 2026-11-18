"""correction_gen: a model's own right and wrong answers, as DPO pairs and exercises."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

from eullm_forge.datasets.correction_gen import (
    build,
    correction_pair,
    diagnosis,
    informative,
    passage_of,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_correction_pairs.py"
PROMPT = ("Testi normativi:\n\n[1] codice civile, art. 7\nAltro.\n\n[2] codice civile, art. 2\n"
          "La domanda si propone entro sessanta giorni.\n\nDomanda: Quale termine prevede "
          "l'art. 2 del codice civile?")
KW = ["60 giorni|sessanta giorni"]


def _row(tipo="termine", answers=(), rid="norm-termine-codice_civile-2"):
    from eullm_forge.rl import score_answer
    return {"id": rid, "prompt": [{"role": "user", "content": PROMPT}], "tipo": tipo,
            "keywords": KW if "termine" in tipo else [], "code": "codice_civile",
            "articolo": "2",
            "samples": [{"answer": a, "score": score_answer(a, tipo, KW if "termine" in tipo
                                                            else [])} for a in answers]}


def test_the_article_is_found_among_the_passages_by_its_label():
    assert passage_of(PROMPT, "codice_civile", "2") == 2
    assert passage_of(PROMPT, "codice_civile", "3") is None


def test_a_bis_article_is_not_taken_for_the_base_article():
    """The label "art. 2-bis" starts with "art. 2": the old boundary matched
    the hyphen and quoted the 30-day neighbour as if it stated 60."""
    prompt = ("Testi normativi:\n\n[1] codice civile, art. 2-bis\n"
              "La domanda si propone entro trenta giorni.\n\n"
              "[2] codice civile, art. 2\n"
              "La domanda si propone entro sessanta giorni.\n\n"
              "Domanda: in materia di termine di prova, qual è il termine?")
    assert passage_of(prompt, "codice_civile", "2") == 2
    assert passage_of(prompt, "codice_civile", "2-bis") == 1


def test_what_is_wrong_is_said_from_what_the_program_checked():
    where = "il testo [2]"
    assert diagnosis("Il termine è di 30 giorni.", "termine", KW, where) == (
        "indica un termine di 30 giorni, ma il testo [2] prevede un termine di 60 giorni")
    assert "non dice che la premessa" in diagnosis("Il termine è di 60 giorni.",
                                                   "premessa_termine", KW, where)
    assert "i testi non rispondono" in diagnosis("Non trovo la disposizione nei testi.",
                                                 "termine", KW, where)
    assert "non è tra i testi" in diagnosis("L'art. 900 prevede la nullità.", "inesistente",
                                            [], where)


def test_a_question_answered_both_ways_gives_a_pair_and_an_exercise():
    row = _row(answers=["Il termine è di 60 giorni [2].", "Il termine è di 30 giorni.",
                        "Il termine è di 30 giorni."])
    corrections, prefs, stats = build([row], random.Random(0), right_share=0.0)
    [p] = prefs
    assert p["chosen"][0]["content"] == "Il termine è di 60 giorni [2]."
    assert p["rejected"][0]["content"] == "Il termine è di 30 giorni."
    assert p["prompt"] == row["prompt"]
    [c] = corrections
    assert c["instruction"].startswith(PROMPT) and "Risposta proposta:\nIl termine è di 30" \
        in c["instruction"]
    assert c["output"] == ("Verifica della risposta proposta: è errata: indica un termine di "
                           "30 giorni, ma il testo [2] prevede un termine di 60 giorni.\n"
                           "Risposta corretta: Il termine è di 60 giorni [2].")
    assert stats == {"pairs": 1, "corrections:wrong": 1}


def test_some_proposed_answers_are_right_and_confirmed():
    row = _row(answers=["Il termine è di 60 giorni [2].", "Sono 60 giorni, art. 2 [2].",
                        "Il termine è di 30 giorni."])
    corrections, _, _ = build([row], random.Random(0), right_share=1.0)
    [c] = corrections
    assert c["output"].startswith("Verifica della risposta proposta: è corretta: indica il "
                                  "termine di 60 giorni che il testo [2] prevede.")


def test_a_question_never_answered_right_gives_nothing_and_judged_kinds_are_left_out():
    wrong = _row(answers=["Il termine è di 30 giorni."] * 3)
    judged = {**_row(answers=["x"]), "tipo": "contenuto"}
    corrections, prefs, stats = build([wrong, judged], random.Random(0))
    assert not corrections and not prefs
    assert stats == {"skipped:never_right": 1, "skipped:not_verifiable": 1}


def test_only_questions_answered_both_ways_are_informative_for_grpo():
    assert informative(_row(answers=["60 giorni", "30 giorni"]))
    assert not informative(_row(answers=["60 giorni", "sessanta giorni"]))
    assert not informative(_row(answers=["30 giorni"]))


def test_the_exercise_keeps_the_id_and_kind():
    c = correction_pair(_row(), "Il termine è di 30 giorni.", "Sono 60 giorni.", False)
    assert c["task"] == "correzione_termine" and c["key"].endswith("-corr")


def test_the_script_prints_counts_only(tmp_path, capsys):
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(_row(answers=["Il termine è di 60 giorni [2].",
                                                 "Il termine è di 30 giorni."])) + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("make_correction_pairs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([str(samples), "--out-sft", str(tmp_path / "c.jsonl"),
                     "--out-dpo", str(tmp_path / "d.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "[corr]" in out and "giorni" not in out
    assert (tmp_path / "d.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_the_script_grades_again_and_writes_the_informative_questions(tmp_path, capsys):
    row = _row(answers=["Il termine è di 60 giorni [2].", "Il termine è di 30 giorni."])
    for s in row["samples"]:
        s["score"] = 0.0                     # as an older reward scored them
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(row) + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("make_correction_pairs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    keep = tmp_path / "informative.jsonl"
    assert mod.main([str(samples), "--informative-out", str(keep)]) == 0
    [k] = [json.loads(ln) for ln in keep.read_text(encoding="utf-8").splitlines()]
    assert "samples" not in k and k["id"] == row["id"]
    assert "pass rate 0.50" in capsys.readouterr().out
