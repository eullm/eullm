"""The context gate's examples, its training helpers and its fixed reply."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

from eullm_forge.datasets.context_gate import (
    from_openbook,
    from_question,
    gate_document,
    openbook_article,
)
from eullm_forge.eval import NormIndex
from eullm_forge.eval.abstain import abstained

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rec(code, text):
    return {"code": code, "article_num": "", "chunk_index": 0, "text": text}


RECORDS = [
    rec("codice_civile", "Art. 1. \n \n (Fonti). \n \n Sono fonti del diritto le leggi e i "
        "regolamenti." + FILLER),
    rec("codice_civile", "Art. 2. \n \n (Termine di prova). \n \n La domanda di prova si "
        "propone entro sessanta giorni dalla notificazione." + FILLER),
    rec("codice_civile", "Art. 3. \n \n (Opposizione). \n \n L'opposizione alla domanda si "
        "propone davanti al giudice." + FILLER),
    rec("codice_civile", "Art. 4. \n \n (Spese). \n \n Le spese della domanda sono a carico "
        "di chi la propone." + FILLER),
]


def pair(number="2", question="Entro quanto tempo si propone la domanda di prova?"):
    return {"instruction": f"Testi normativi di riferimento:\n\n[1] x\n\nDomanda: {question}",
            "output": "…", "task": "openbook_grounded", "key": f"ob-g-codice_civile-{number}",
            "named": False}


def test_the_article_of_a_grounded_pair_is_read_from_its_key():
    assert openbook_article(pair()) == ("codice_civile", "2")
    assert openbook_article({**pair(), "key": "ob-g-codice_penale-624-bis-v2"}) == \
        ("codice_penale", "624-bis")
    assert openbook_article({**pair(), "task": "openbook_absent"}) is None


def test_one_question_gives_its_article_in_the_texts_and_taken_out():
    index = NormIndex(RECORDS)
    inside, outside = from_openbook(pair(), index, random.Random(0), k=2)
    assert inside["label"] == 1 and inside["reason"] == "presente"
    assert outside["label"] == 0 and outside["reason"] == "articolo_assente"
    assert inside["question"] == outside["question"]
    assert "sessanta giorni" in inside["document"]
    assert "sessanta giorni" not in outside["document"]
    assert "codice civile, art. 2" in inside["document"]
    # An article the collection does not hold gives nothing.
    assert from_openbook(pair("99"), index, random.Random(0)) == []


def test_the_texts_are_numbered_labelled_and_cut():
    doc = gate_document(RECORDS[:2], max_chars=30)
    assert doc.startswith("[1] codice civile, art. 1\n")
    assert "\n\n[2] codice civile, art. 2\n" in doc
    assert all(len(block.split("\n", 1)[1]) <= 30 for block in doc.split("\n\n"))
    assert gate_document([]) == "(nessun testo trovato)"


def test_off_topic_questions_are_parsed_and_the_ones_naming_our_codes_dropped():
    mod = load("make_offtopic_questions")
    text = ("1. Come si calcola la detrazione IRPEF per le spese mediche sostenute?\n"
            "- Quando scade il pagamento dell'IMU sulla seconda casa?\n"
            "Ecco alcune domande:\n"
            "• Quanti punti si perdono per un sorpasso vietato in curva?\n")
    qs = mod.parse_questions(text)
    assert qs == ["Come si calcola la detrazione IRPEF per le spese mediche sostenute?",
                  "Quando scade il pagamento dell'IMU sulla seconda casa?",
                  "Quanti punti si perdono per un sorpasso vietato in curva?"]
    rows, stats = mod.keep([("fisco", qs[0]), ("fisco", qs[0].upper()),
                            ("x", "Che cosa prevede il codice civile sulle spese mediche?")])
    assert [r["question"] for r in rows] == [qs[0]]
    assert stats == {"duplicate": 1, "names_a_code_we_hold": 1}


def test_the_examples_file_keeps_dev_articles_and_matters_apart(tmp_path):
    norms = tmp_path / "legislazione_cc.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS), encoding="utf-8")
    ob = tmp_path / "openbook.jsonl"
    ob.write_text("".join(json.dumps(pair(n, q)) + "\n" for n, q in (
        ("1", "Quali sono le fonti del diritto?"),
        ("2", "Entro quanto si propone la domanda di prova?"),
        ("3", "Davanti a chi si propone l'opposizione?"),
        ("4", "Chi paga le spese della domanda?"))), encoding="utf-8")
    off = tmp_path / "off.jsonl"
    off.write_text("".join(json.dumps({"question": f"Domanda fiscale numero {i}?",
                                       "topic": f"t{i % 3}", "id": f"off-{i}"}) + "\n"
                           for i in range(6)), encoding="utf-8")
    prem = tmp_path / "premise.jsonl"
    prem.write_text(json.dumps({"id": "pf-i-1", "tipo": "premessa_inesistente",
                                "prompt": [{"role": "user", "content": "Testi…\n\nDomanda: "
                                            "Che cosa prevede l'art. 7-bis del codice civile?"}]})
                    + "\n", encoding="utf-8")
    tr, dv = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    rc = load("make_gate_pairs").main(["--norms", str(norms), "--openbook", str(ob),
                                       "--premise", str(prem), "--offtopic", str(off),
                                       "--dev-share", "0.34", "-k", "2",
                                       "--out-train", str(tr), "--out-dev", str(dv)])
    assert rc == 0
    train = [json.loads(line) for line in tr.open()]
    dev = [json.loads(line) for line in dv.open()]
    reasons = {r["reason"] for r in train + dev}
    assert reasons == {"presente", "articolo_assente", "articolo_inesistente", "fuori_ambito"}
    # No article and no off-topic matter on both sides.
    side = {}
    for name, rows in (("train", train), ("dev", dev)):
        for r in rows:
            if r["key"].startswith("ob-g-"):
                side.setdefault(r["key"].rsplit("-gate", 1)[0], set()).add(name)
    assert all(len(v) == 1 for v in side.values())
    assert sum(r["label"] for r in train + dev) == 4


def test_the_threshold_balances_the_two_errors():
    mod = load("train_context_gate")
    t, bal = mod.best_threshold([0.9, 0.8, 0.3, 0.2, 0.6], [1, 1, 0, 0, 0])
    assert 0.6 < t <= 0.8 and bal == 1.0
    rows = [{"label": 1, "reason": "presente"}, {"label": 0, "reason": "fuori_ambito"}]
    assert mod.report(rows, [0.9, 0.7], 0.5) == {
        "accuracy": 0.5, "by_reason": {"fuori_ambito": 0.0, "presente": 1.0}, "n": 2}


def test_the_gate_input_is_the_rerankers_with_the_texts_cut_to_fit():
    mod = load("train_context_gate")

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    ids = mod.encode(Tok(), {"question": "q", "document": "x" * 500}, [1], [2], 300)
    assert len(ids) == 300 and ids[0] == 1 and ids[-1] == 2
    assert ids.count(ord("x")) < 500


def test_the_gated_reply_reads_as_an_abstention():
    from eullm_forge.eval.gate import GATE_ANSWER
    assert abstained(GATE_ANSWER)


def test_an_off_topic_question_is_labelled_insufficient_whatever_it_retrieves():
    # The texts look close (the expenses of a "domanda"), the matter is not ours.
    ex = from_question("Chi paga le spese della domanda di rimborso IMU?", NormIndex(RECORDS),
                       "fuori_ambito", "k", k=2)
    assert ex["label"] == 0 and ex["document"].startswith("[1] ")


def test_the_threshold_never_gates_more_than_its_share_of_sufficient_questions():
    mod = load("train_context_gate")
    probs = [0.01, 0.02, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.001, 0.003]
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    t = mod.capped_threshold(probs, labels, 0.55, 0.1)   # 1 of 10 may be gated
    assert t == 0.02
    assert sum(1 for p, y in zip(probs, labels) if y and p < t) == 1


def test_off_topic_prompts_are_rewarded_for_abstaining(tmp_path):
    from eullm_forge.rl import score_answer
    norms = tmp_path / "legislazione_cc.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS), encoding="utf-8")
    off = tmp_path / "off.jsonl"
    # texts that look close (a "domanda" and its expenses), on a matter not ours
    off.write_text(json.dumps({"question": "Chi paga le spese della domanda di rimborso IMU?",
                               "topic": "IMU", "id": "off-1"}) + "\n", encoding="utf-8")
    out = tmp_path / "prompts.jsonl"
    assert load("make_offtopic_prompts").main(["--norms", str(norms), "--offtopic", str(off),
                                               "-k", "2", "--out", str(out)]) == 0
    [row] = [json.loads(line) for line in out.open()]
    content = row["prompt"][0]["content"]
    assert row["tipo"] == "assente" and content.startswith("Testi normativi di riferimento")
    assert content.endswith("Domanda: Chi paga le spese della domanda di rimborso IMU?")
    for right in ("I testi riportati non riguardano l'IMU, quindi non posso rispondere.",
                  "Tra i testi a mia disposizione non trovo norme sull'IMU.",
                  "La materia non rientra tra i testi che ho davanti."):
        assert score_answer(right, "assente", []) == 1.0, right
    assert score_answer("L'IMU si paga in due rate, entro 16 giorni dalla scadenza.",
                        "assente", []) == 0.0


def test_off_topic_prompts_can_ask_the_judge_for_the_matter_and_what_is_needed(tmp_path):
    from eullm_forge.rl import JUDGED_TYPES, answer_reward
    mod = load("make_offtopic_prompts")
    norms = tmp_path / "legislazione_cc.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS), encoding="utf-8")
    off = tmp_path / "off.jsonl"
    off.write_text(json.dumps({"question": "Chi paga le spese della domanda di rimborso IMU?",
                               "topic": "IMU, TARI e altri tributi locali", "id": "off-1"}) + "\n",
                      encoding="utf-8")
    first = tmp_path / "prompts.jsonl"
    assert mod.main(["--norms", str(norms), "--offtopic", str(off), "-k", "2",
                     "--out", str(first)]) == 0
    # the rows already built, rewritten: same texts, judged
    out = tmp_path / "judged.jsonl"
    assert mod.main(["--norms", str(norms), "--from-prompts", str(first),
                     "--kind", "fuori_ambito", "--out", str(out)]) == 0
    [old] = [json.loads(line) for line in first.open()]
    [row] = [json.loads(line) for line in out.open()]
    assert row["prompt"] == old["prompt"] and row["tipo"] == "fuori_ambito" in JUDGED_TYPES
    assert row["question"] == "Chi paga le spese della domanda di rimborso IMU?"
    assert "IMU, TARI" in row["reference"] and "IMU, TARI" in row["rubric"]
    assert "partial" in row["rubric"]
    # the verifiable reward leaves it to the judge
    assert answer_reward(["Non trovo la disposizione."], tipo=["fuori_ambito"],
                         keywords=[[]]) == [None]
    # straight from the questions, too
    direct = tmp_path / "direct.jsonl"
    assert mod.main(["--norms", str(norms), "--offtopic", str(off), "-k", "2",
                     "--kind", "fuori_ambito", "--out", str(direct)]) == 0
    assert json.loads(direct.read_text(encoding="utf-8"))["tipo"] == "fuori_ambito"
