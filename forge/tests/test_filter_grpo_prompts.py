"""GRPO prompts made on Leonardo, kept off the articles of the tests drawn on JUPITER."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3


def rec(code, text):
    return {"code": code, "article_num": "", "chunk_index": 0, "text": text}


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


def _rows(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_no_prompt_is_on_an_article_of_the_trap_test(tmp_path, capsys):
    norms = tmp_path / "legislazione_test.chunks.jsonl"
    norms.write_text("".join(json.dumps(r) + "\n" for r in RECORDS), encoding="utf-8")
    # The real producers: GRPO prompts as make_premise_pairs.py writes them on
    # Leonardo, and a trap test drawn on the same articles.
    grpo = tmp_path / "prompts-premise.jsonl"
    assert _load("make_premise_pairs").main(
        ["--norms", str(norms), "--per-code", "10", "--grpo-share", "1",
         "--out-grpo", str(grpo)]) == 0
    test = tmp_path / "trap-test.jsonl"
    assert _load("make_trap_test").main(
        ["--norms", str(norms), "--per-code", "10", "--out", str(test)]) == 0
    tested = {(it["metadata"]["code"], str(it["metadata"]["articolo"])) for it in _rows(test)}
    before = _rows(grpo)
    assert {(r["code"], str(r["articolo"])) for r in before} & tested

    # A second file repeating the first: each prompt is kept once.
    again = tmp_path / "prompts-again.jsonl"
    again.write_text(grpo.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "prompts.jsonl"
    assert _load("filter_grpo_prompts").main(
        [str(grpo), str(again), "--exclude-exam", str(test), "--out", str(out)]) == 0
    kept = _rows(out)
    assert kept and not {(r["code"], str(r["articolo"])) for r in kept} & tested
    assert len({r["id"] for r in kept}) == len(kept)
    assert len(kept) == sum((r["code"], str(r["articolo"])) not in tested for r in before)
    printed = capsys.readouterr().out
    assert "[grpo-prompts]" in printed and "giorni" not in printed

    # A test on every prompt's article leaves nothing to train on: refused.
    every = tmp_path / "every.jsonl"
    every.write_text("".join(json.dumps(
        {"id": f"t{i}", "domain": "legal", "lang": "it", "question": "?", "reference": "",
         "metadata": {"code": r["code"], "articolo": str(r["articolo"])}}) + "\n"
        for i, r in enumerate(before)), encoding="utf-8")
    gone = tmp_path / "gone.jsonl"
    assert _load("filter_grpo_prompts").main(
        [str(grpo), "--exclude-exam", str(every), "--out", str(gone)]) == 1
    assert not gone.exists() and "no prompt left" in capsys.readouterr().err


def test_the_note_is_taken_out_as_on_leonardo(tmp_path):
    head = "Testi normativi di riferimento:\n[1] art. 2 ...\n\nDomanda: entro quanto?"
    rows = [{"id": f"p{i}", "prompt": [{"role": "user", "content":
             "Nota: nella raccolta normativa non è presente art. 9.\n\n" + head}],
             "tipo": "assente", "keywords": [], "code": "codice_civile", "articolo": "9"}
            for i in range(40)]
    src = tmp_path / "noted.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ours, theirs = tmp_path / "ours.jsonl", tmp_path / "theirs.jsonl"
    assert _load("filter_grpo_prompts").main([str(src), "--out", str(ours)]) == 0
    assert _load("drop_prompt_notes").main([str(src), "--out", str(theirs)]) == 0
    by_id = lambda rows: sorted(rows, key=lambda r: r["id"])  # noqa: E731
    assert by_id(_rows(ours)) == by_id(_rows(theirs))
    bare = sum(r["prompt"][0]["content"] == head for r in _rows(ours))
    assert 0 < bare < len(rows)
