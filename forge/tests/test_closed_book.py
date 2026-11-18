"""The closed-book suite: built from the legislation, scored by program."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eullm_forge.eval.closed_book import build_suite, cited_articles, heading_overlap, score

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
PAD = " La disposizione si applica secondo le regole generali del presente codice." * 3


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rec(n, heading, text):
    return {"code": "codice_penale", "article_num": "", "chunk_index": 0,
            "text": f"Art. {n}. \n \n ({heading}). \n \n {text}{PAD}"}


RECORDS = [
    rec("99", "Recidiva", "Chi, dopo essere stato condannato per un delitto non colposo, ne "
        "commette un altro, puo' essere sottoposto a un aumento di un terzo della pena."),
    rec("624", "Furto", "Chiunque s'impossessa della cosa mobile altrui e' punito con la "
        "reclusione da sei mesi a tre anni. Il delitto e' punibile a querela della persona "
        "offesa."),
    rec("161", "Effetti della sospensione", "La querela deve essere proposta entro trenta "
        "giorni dalla notizia del fatto."),
    rec("594", "Ingiuria", "Articolo abrogato dal d.lgs. 15 gennaio 2016, n. 7."),
]


def suite():
    return {it.id: it.to_dict() for it in build_suite(RECORDS, per_kind=5, n_rulings=3)}


def test_every_kind_is_drawn_from_what_the_articles_state():
    items = suite()
    kinds = {it["metadata"]["tipo"] for it in items.values()}
    assert kinds == {"esiste_vera", "esiste_falsa", "numero_da_rubrica", "rubrica_da_numero",
                     "termine", "pena", "procedibilita", "abrogato", "sentenza_falsa"}
    fake = [it for it in items.values() if it["metadata"]["tipo"] == "esiste_falsa"]
    assert fake and all(it["metadata"]["articolo"] not in ("99", "624", "161", "594")
                        for it in fake)
    pena = next(it for it in items.values() if it["metadata"]["tipo"] == "pena")
    assert pena["metadata"]["high"] == [3, "anni"]
    proc = next(it for it in items.values() if it["metadata"]["tipo"] == "procedibilita")
    assert proc["keywords"] == ["querela"]
    # the repealed article is asked only as repealed, never as existing
    assert not any(it["metadata"]["articolo"] == "594" and it["metadata"]["tipo"] != "abrogato"
                   for it in items.values())


def test_answers_are_scored_by_kind_and_sure_wrong_apart():
    items = suite()

    def of(kind, number=None):
        return next(it for it in items.values() if it["metadata"]["tipo"] == kind
                    and (number is None or it["metadata"]["articolo"] == number))

    fake = of("esiste_falsa")
    assert score(fake, "No, l'articolo non esiste nel codice penale.") == ("right", False)
    assert score(fake, "Non so dirti se esista.") == ("partial", False)
    assert score(fake, "Sì, disciplina la truffa aggravata.") == ("wrong", True)
    real = of("esiste_vera", "99")
    assert score(real, "Sì, l'art. 99 c.p. disciplina la recidiva.")[0] == "right"
    assert score(real, "L'art. 99 non esiste.") == ("wrong", True)
    assert score(of("numero_da_rubrica", "624"), "È l'art. 624 c.p.")[0] == "right"
    assert score(of("numero_da_rubrica", "624"), "Gli artt. 1, 2, 3 e 4.")[0] == "wrong"
    assert score(of("termine"), "Entro trenta giorni.")[0] == "right"
    assert score(of("pena"), "Reclusione da sei mesi a tre anni.")[0] == "right"
    assert score(of("pena"), "Reclusione fino a cinque anni.") == ("wrong", True)
    assert score(of("procedibilita"), "È procedibile a querela.")[0] == "right"
    assert score(of("procedibilita"), "Si procede d'ufficio.") == ("wrong", True)
    assert score(of("abrogato"), "È stato abrogato nel 2016.")[0] == "right"
    ruling = of("sentenza_falsa")
    assert score(ruling, "Non posso confermare il contenuto di quella sentenza.")[0] == "right"
    assert score(ruling, "Ha stabilito che il sequestro è nullo.") == ("wrong", True)


def test_the_helpers_read_numbers_and_headings():
    assert cited_articles("Vedi l'art. 624-bis e l'articolo 99 c.p.") == {"624-bis", "99"}
    assert heading_overlap("Atti persecutori", "riguarda gli atti persecutori") == 1.0
    assert heading_overlap("Recidiva", "parla di furto") == 0.0


def test_the_scripts_build_once_and_print_counts_only(tmp_path, capsys):
    norms = tmp_path / "legislazione_cp.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS), encoding="utf-8")
    out = tmp_path / "closed-book.jsonl"
    make = load("make_closed_book")
    assert make.main(["--norms", str(norms), "--per-kind", "5", "--rulings", "3",
                      "--out", str(out)]) == 0
    first = out.read_text(encoding="utf-8")
    assert make.main(["--norms", str(norms), "--out", str(out)]) == 0 and out.read_text(encoding="utf-8") == first
    items = [json.loads(line) for line in first.splitlines()]
    answers = tmp_path / "answers-r2l.jsonl"
    answers.write_text("".join(json.dumps({"id": it["id"], "answer": "Non lo so."}) + "\n"
                               for it in items), encoding="utf-8")
    capsys.readouterr()
    assert load("score_closed_book").main([str(out), str(answers)]) == 0
    printed = capsys.readouterr().out
    assert "r2l" in printed and "Esiste" not in printed and "Recidiva" not in printed
    assert "denies-real" in printed and "procedibility: 100% of the items want" in printed
    # a model that says "non esiste" to everything: right on the made-up, denying the real
    answers.write_text("".join(json.dumps({"id": it["id"], "answer": "Non esiste."}) + "\n"
                               for it in items), encoding="utf-8")
    load("score_closed_book").main([str(out), str(answers)])
    row = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("r2l"))
    assert "100%" in row.split()[4]
