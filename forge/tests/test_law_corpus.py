"""The memory corpus: program-written statements, teacher rewrites checked by program."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eullm_forge.datasets.law_corpus import (
    check_rewrite,
    excluded,
    number_key,
    program_rows,
    rewritable,
    rewrite_messages,
)
from eullm_forge.eval.norm_exam import articles_from_records

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
PAD = " La disposizione si applica secondo le regole generali del presente codice." * 2


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
    rec("624-bis", "Furto in abitazione", "Chiunque si impossessa della cosa mobile altrui "
        "introducendosi in un edificio e' punito con la reclusione da quattro a sette anni."),
    rec("625", "Circostanze aggravanti", "La pena per il fatto previsto dall'articolo 624 e' "
        "aumentata. La querela deve essere proposta entro trenta giorni dalla notizia."),
    rec("594", "Ingiuria", "Articolo abrogato dal d.lgs. 15 gennaio 2016, n. 7."),
]


def articles():
    return articles_from_records(RECORDS)


def test_the_program_states_what_each_article_says_and_what_the_code_lacks():
    rows = program_rows(articles(), absent_share=1.0)
    by = {(r["forma"], r["articolo"]): r["text"] for r in rows}
    assert by[("testo", "624")].startswith("Codice penale, art. 624 (Furto).")
    assert "«Furto»" in by[("rubrica", "624")] and "art. 624" in by[("rubrica", "624")]
    assert "reclusione da 6 mesi a 3 anni" in by[("pena", "624")]
    assert "a querela della persona offesa" in by[("procedibilita", "624")]
    assert "30 giorni" in by[("termine", "625")]
    assert ("termine", "624") not in by                     # a penalty's length is no deadline
    assert "non è più in vigore" in by[("abrogato", "594")]
    assert ("rubrica", "594") not in by                     # repealed: stated as repealed only
    # after 624-bis the next is 625, so 624-ter does not exist
    absent = by[("inesistente", "624-ter")]
    assert absent.startswith("Nel codice penale non esiste un art. 624-ter")
    assert "art. 624-bis (Furto in abitazione)" in absent and "art. 625" in absent
    index = next(r["text"] for r in rows if r["forma"] == "indice")
    assert index.index("art. 99") < index.index("art. 594") < index.index("art. 624 ") \
        < index.index("art. 624-bis") < index.index("art. 625")
    assert "Ingiuria (abrogato)" in index
    assert not any("?" in r["text"] for r in rows)          # statements, never questions


def test_numbers_asked_about_by_an_evaluation_are_never_stated_absent(tmp_path):
    suite = tmp_path / "closed-book.jsonl"
    suite.write_text(json.dumps({"id": "cb-esiste_falsa-codice_penale-624-ter",
                                 "metadata": {"tipo": "esiste_falsa", "code": "codice_penale",
                                              "articolo": "624-ter"}}) + "\n"
                     + json.dumps({"id": "x", "metadata": {"tipo": "pena",
                                                           "code": "codice_penale",
                                                           "articolo": "624"}}) + "\n")
    fakes, asked = excluded([suite])
    assert fakes == {("codice_penale", "624-ter")} and asked == {("codice_penale", "624")}
    rows = program_rows(articles(), fakes=fakes, absent_share=1.0)
    assert not any(r["articolo"] == "624-ter" for r in rows)
    work = load("law_rewrites").work_list(articles(), ["parafrasi", "domande"], 1, asked)
    keys = {w[0] for w in work}
    assert "codice_penale|624|parafrasi|0" in keys and "codice_penale|624|domande|0" not in keys
    assert "codice_penale|99|domande|0" in keys
    assert not any("|594|" in k for k in keys)               # repealed: not rewritten


def test_a_rewrite_is_kept_only_if_it_adds_nothing_to_the_article():
    a = articles()[("codice_penale", "624")]
    good = ("L'art. 624 del codice penale punisce chi si appropria di una cosa mobile che "
            "appartiene ad altri: la reclusione va da sei mesi a tre anni e si procede solo se "
            "la persona offesa presenta querela.")
    assert check_rewrite(a, "parafrasi", good) is None
    assert check_rewrite(a, "parafrasi", good.replace("art. 624", "norma")) == "no_article"
    assert check_rewrite(a, "parafrasi", good + " Si veda anche l'art. 640 c.p.") \
        == "other_article"
    assert check_rewrite(a, "parafrasi", good.replace("tre anni", "cinque anni")) == "deadline"
    assert check_rewrite(a, "caso", good + " Tizio sottrae beni per 1.500 euro.") == "number"
    assert check_rewrite(a, "parafrasi", "L'art. 624 c.p.") == "length"
    t = articles()[("codice_penale", "625")]
    plain = ("L'art. 625 del codice penale aumenta la pena del furto previsto dall'articolo 624 "
             "quando ricorrono le circostanze aggravanti e regola la querela della persona "
             "offesa, con le regole generali del codice.")
    assert check_rewrite(t, "spiegazione", plain) is None
    assert check_rewrite(t, "parafrasi", plain) == "lost_deadline"


def test_the_teacher_is_shown_the_article_and_asked_one_style():
    a = articles()[("codice_penale", "624")]
    assert rewritable(a) and not rewritable(articles()[("codice_penale", "594")])
    msgs = rewrite_messages(a, "domande")
    assert "Art. 624 del codice penale (Furto)" in msgs[1]["content"]
    assert "cosa mobile altrui" in msgs[1]["content"] and "«D:»" in msgs[1]["content"]
    assert number_key("99") < number_key("99-bis") < number_key("99-ter") < number_key("100")


def test_the_scripts_write_the_corpus_and_resume(tmp_path, capsys, monkeypatch):
    norms = tmp_path / "legislazione_cp.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    out = tmp_path / "program.jsonl"
    assert load("make_law_corpus").main(["--norms", str(norms), "--out", str(out)]) == 0
    assert "rows" in capsys.readouterr().out and out.stat().st_size > 0

    mod = load("law_rewrites")
    replies = iter(["L'art. 99 del codice penale prevede che chi, già condannato per un "
                    "delitto non colposo, ne commetta un altro possa subire un aumento di un "
                    "terzo della pena.", "Testo che non nomina l'articolo." * 10])

    def fake_ask(url, msgs, max_tokens, temperature, seed, timeout=600.0):
        return {"choices": [{"message": {"content": next(replies, "x")},
                             "finish_reason": "stop"}], "usage": {}}

    monkeypatch.setattr(mod, "ask", fake_ask)
    rw = tmp_path / "rewrites.jsonl"
    args = ["--norms", str(norms), "--out", str(rw), "--url", "http://x", "--parallel", "1",
            "--styles", "parafrasi", "--codes", "codice_penale", "--limit", "2"]
    assert mod.main(args) == 0
    kept = [json.loads(line) for line in rw.open()]
    rejected = [json.loads(line) for line in (tmp_path / "rewrites.rejects.jsonl").open()]
    assert len(kept) == 1 and kept[0]["forma"] == "parafrasi" and kept[0]["articolo"] == "99"
    assert len(rejected) == 1 and rejected[0]["reason"] == "no_article"
    capsys.readouterr()
    mod.main(args)                                           # the two done are not asked again
    printed = capsys.readouterr().out
    assert "2 done already" in printed and "L'art. 99" not in printed


def test_a_chain_started_halfway_asks_the_middle_of_the_list(tmp_path, monkeypatch):
    norms = tmp_path / "legislazione_cp.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    mod = load("law_rewrites")
    asked = []

    def fake_ask(url, msgs, max_tokens, temperature, seed, timeout=600.0):
        asked.append(msgs[1]["content"].split("\n")[0])
        return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}], "usage": {}}

    monkeypatch.setattr(mod, "ask", fake_ask)
    base = ["--norms", str(norms), "--url", "http://x", "--parallel", "1",
            "--styles", "parafrasi", "--limit", "1"]
    mod.main(base + ["--out", str(tmp_path / "a" / "rewrites.jsonl")])
    mod.main(base + ["--out", str(tmp_path / "b" / "rewrites.jsonl"), "--start", "0.5"])
    # four articles in force, 99 624 624-bis 625: halfway is the third
    assert asked == ["Art. 99 del codice penale (Recidiva)",
                     "Art. 624-bis del codice penale (Furto in abitazione)"]


def test_eu_acts_are_named_from_their_celex_number_and_rewritten():
    from eullm_forge.datasets.law_corpus import by_code, labels

    assert labels("eurlex_32016R0679") == ("del regolamento (UE) 2016/679",
                                           "nel regolamento (UE) 2016/679")
    assert labels("eurlex_32012R1215")[0] == "del regolamento (UE) n. 1215/2012"
    assert labels("eurlex_32003R0001")[0] == "del regolamento (CE) n. 1/2003"
    assert labels("eurlex_32011L0083")[0] == "della direttiva 2011/83/UE"
    assert labels("eurlex_32019L1937")[0] == "della direttiva (UE) 2019/1937"
    assert labels("eurlex_52020PC0825") is None          # a proposal: left out
    assert labels("codice_penale")[0] == "del codice penale"
    # A chunk as prepare_eurlex.py writes it: the act's title, then the article.
    title = "REGOLAMENTO (UE) 2016/679 DEL PARLAMENTO EUROPEO E DEL CONSIGLIO"
    body = ("Articolo 5\nPrincipi applicabili al trattamento di dati personali\n"
            "1. I dati personali sono trattati in modo lecito, corretto e trasparente nei "
            "confronti dell'interessato." + PAD)
    arts = articles_from_records([{"code": "eurlex_32016R0679", "chunk_index": 0,
                                   "article_num": "5", "text": f"{title}\n{body}"}])
    assert [a.number for a in by_code(arts)["eurlex_32016R0679"]] == ["5"]
    msgs = rewrite_messages(arts[("eurlex_32016R0679", "5")], "parafrasi")
    assert "del regolamento (UE) 2016/679" in json.dumps(msgs, ensure_ascii=False)
