"""The trap test: drawn on articles no premise pair trained on, graded without a judge."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eullm_forge.eval import load_eval_set

FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3


def rec(code, text):
    return {"code": code, "article_num": "", "chunk_index": 0, "text": text}


# As in test_premise_gen.py: two deadlines, a repealed article, two without.
RECORDS = [
    rec("codice_civile", "Art. 1. \n \n (Fonti). \n \n Sono fonti le leggi." + FILLER),
    rec("codice_civile", "Art. 2. \n \n (Termine di prova). \n \n La domanda si propone "
        "entro sessanta giorni dalla notificazione." + FILLER),
    rec("codice_civile", "Art. 3. \n \n (Opposizione). \n \n L'opposizione si propone "
        "entro dieci giorni dalla comunicazione." + FILLER),
    rec("codice_civile", "Art. 4. \n \n (Abrogato). \n \n Articolo abrogato dalla legge "
        "n. 10 del 2000." + FILLER),
    rec("codice_civile", "Art. 5. \n \n (Contratti). \n \n Il contratto è nullo se manca "
        "la causa." + FILLER),
]

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norms(tmp_path):
    norms = tmp_path / "legislazione_test.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS), encoding="utf-8")
    return norms


def test_no_question_is_on_an_article_the_premise_pairs_trained_on(tmp_path, capsys):
    norms = _norms(tmp_path)
    # The real producer: the training pairs as make_premise_pairs.py writes them.
    sft = tmp_path / "sft.jsonl"
    assert _load("make_premise_pairs").main(
        ["--norms", str(norms), "--per-code", "1", "--grpo-share", "0", "--out-sft", str(sft)]) == 0
    trained = {(r["source"], r["key"].split("-", 3)[3])
               for r in map(json.loads, sft.read_text(encoding="utf-8").splitlines())}
    assert trained
    out = tmp_path / "trap-test.jsonl"
    assert _load("make_trap_test").main(
        ["--norms", str(norms), "--per-code", "10", "--exclude-trained", str(sft),
         "--out", str(out)]) == 0
    items = load_eval_set(out)                       # legal_eval.py reads it as it is
    assert items
    assert not {(i.metadata["code"], i.metadata["articolo"]) for i in items} & trained
    assert all(i.rubric and i.reference and i.metadata["tipo"] == i.category for i in items)
    printed = capsys.readouterr().out
    assert "[trap-test]" in printed and "giorni" not in printed and "Verifica" not in printed


def test_answers_are_graded_by_kind_and_a_model_that_refuses_everything_fails_the_true_ones(
        tmp_path, capsys):
    norms = _norms(tmp_path)
    out = tmp_path / "trap-test.jsonl"
    assert _load("make_trap_test").main(["--norms", str(norms), "--per-code", "10",
                                         "--out", str(out)]) == 0
    items = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert {"premessa_termine", "premessa_vera"} <= {i["category"] for i in items}
    # Answers files as legal_eval.py writes them: the reference itself, and a refusal.
    right = tmp_path / "answers-trap-test-right.jsonl"
    right.write_text("".join(json.dumps({"id": i["id"], "answer": i["reference"]}) + "\n"
                             for i in items), encoding="utf-8")
    doubt = tmp_path / "answers-trap-test-doubt.jsonl"
    doubt.write_text("".join(json.dumps({"id": i["id"], "answer": "La premessa è errata."}) + "\n"
                             for i in items), encoding="utf-8")
    assert _load("score_trap_test").main([str(out), str(right), str(doubt)]) == 0
    rows = {ln.split()[0]: ln.split() for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("trap-test-")}
    assert rows["trap-test-right"][1] == "100.0%"
    assert rows["trap-test-doubt"][3] == "0.0%"     # every true premise refused


def test_rewrites_keep_every_number_and_correct_nothing():
    mod = _load("paraphrase_trap_test")
    q = "Mi confermi che l'art. 640 c.p. punisce il fatto con la reclusione da 2 a 8 anni?"
    assert mod.faithful(q, "Un mio cliente rischia per l'art. 640 c.p. da 2 a 8 anni: è così?")
    assert not mod.faithful(q, "L'art. 640 c.p. prevede da due a otto anni?")      # a number lost
    assert not mod.faithful(q, "L'art. 640 c.p. da 2 a 8 anni? In realtà sono meno.")  # answered
    assert mod.clean("«Domanda riscritta: L'art. 2 è così?»") == "L'art. 2 è così?"


def test_the_rewritten_test_is_graded_as_the_original(tmp_path, monkeypatch, capsys):
    norms = _norms(tmp_path)
    items = tmp_path / "trap-test.jsonl"
    assert _load("make_trap_test").main(["--norms", str(norms), "--per-code", "10",
                                         "--out", str(items)]) == 0
    para_mod = _load("paraphrase_trap_test")          # puts forge/scripts on the path
    import generate_instructions

    class Teacher:                     # rewrites in the teacher's way: the question, reworded
        def __init__(self, *a, **k):
            pass

        def generate(self, convs):
            out = []
            for c in convs:
                q = c[0]["content"].rsplit("Domanda: ", 1)[1]
                out.append("Domanda riscritta: Scusi, una cosa: " + q[0].lower() + q[1:])
            return out, 7 * len(convs)

    monkeypatch.setattr(generate_instructions, "EullmGenerator", Teacher)
    para = tmp_path / "trap-test-para.jsonl"
    assert para_mod.main([str(items), "--servers", "http://x",
                                               "--out", str(para)]) == 0
    rewritten = load_eval_set(para)
    originals = {json.loads(ln)["id"]: json.loads(ln)
                 for ln in items.read_text(encoding="utf-8").splitlines()}
    assert len(rewritten) == 3 * len(originals)
    for it in rewritten:
        src = originals[it.metadata["paraphrase_of"]]
        assert it.question.startswith("Scusi") and it.reference == src["reference"]
        assert it.metadata["tipo"] == src["metadata"]["tipo"]
    assert "Scusi" not in capsys.readouterr().out
    # The same answers grade the same on the rewritten test.
    right = tmp_path / "answers-trap-test-para-right.jsonl"
    right.write_text("".join(json.dumps({"id": it.id, "answer": it.reference}) + "\n"
                             for it in rewritten), encoding="utf-8")
    assert _load("score_trap_test").main([str(para), str(right)]) == 0
    assert "100.0%" in capsys.readouterr().out
