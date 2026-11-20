"""Consiglio di Stato step 0 and 1: rulings back from chunks, the split, the cards.

Pinned: chunks rejoin into the ruling with their overlap dropped; the dev
split keeps an appeal's rulings on one side, leaves 2025-2026 out of both
lists and is the same on a second run; a card that carries a pseudonym
placeholder, a tax code or malformed fields is refused; cds_schede.py,
against a stand-in teacher speaking the chat API, writes one card per
ruling, refuses the bad ones, skips both on a second run and never prints
a ruling.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from eullm_forge.caselaw import join_chunks, load_rulings, ruling_view
from eullm_forge.caselaw.schede import CardRejected, parse_card, prefix

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOOD = {
    "principi": ["Il termine per impugnare l'aggiudicazione decorre dalla pubblicazione "
                 "dell'atto sul profilo del committente, salvo conoscenza anteriore."],
    "norme": ["art. 120 c.p.a.", "art. 29, d.lgs. n. 50/2016"],
    "esito": "rigetto",
    "materia": "appalti pubblici - termine di impugnazione",
    "domande_ricerca": ["Da quando decorre il termine per impugnare l'aggiudicazione?",
                        "La conoscenza anteriore dell'atto anticipa il termine?",
                        "Che rilievo ha la pubblicazione sul profilo del committente?"],
    "domande_esame": [
        {"domanda": "Da quando decorre il termine per impugnare un'aggiudicazione?",
         "risposta": "Dalla pubblicazione sul profilo del committente.",
         "rubrica": "Deve indicare la pubblicazione sul profilo del committente."},
        {"domanda": "La conoscenza aliunde anticipa il termine?",
         "risposta": "Sì, se piena.", "rubrica": "Deve dire che la conoscenza piena lo anticipa."}],
}


def _chunks(text: str, size: int = 300, overlap: int = 60) -> list[str]:
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return out


def test_chunks_rejoin_without_their_overlap():
    text = " ".join(f"parola{i}" for i in range(400))
    assert join_chunks(_chunks(text)) == text


def test_a_long_ruling_keeps_its_head_and_its_reasons():
    text = "EPIGRAFE " * 2000 + "\nDIRITTO\n" + "MOTIVAZIONE " * 3000
    view = ruling_view(text, max_chars=8000)
    assert len(view) <= 8100 and view.startswith("EPIGRAFE") and "\nDIRITTO\n" in view


def test_a_card_is_parsed_and_the_case_cannot_leak_into_it():
    card = parse_card("```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```")
    assert card["esito"] == "rigetto" and len(card["domande_esame"]) == 2
    assert prefix(card, {"sezione": "Sezione Quinta", "numero": "1234", "data": "2023-03-01"}) \
        .startswith("[Cons. Stato, Sezione Quinta, n. 1234, 2023-03-01 - appalti")
    for bad, reason in [
        ({**GOOD, "principi": ["La società [PERSONA_1] non poteva essere esclusa dalla gara "
                               "per il motivo dedotto."]}, "placeholder"),
        # the teacher varies capitalisation ([persona_1]), and instruct_gen
        # documents the same for its own check: only uppercase was refused.
        ({**GOOD, "principi": ["La società [persona_1] non poteva essere esclusa."]},
         "placeholder"),
        ({**GOOD, "principi": ["La società [Persona_1] non poteva essere esclusa."]},
         "placeholder"),
        ({**GOOD, "materia": "RSSMRA80A01H501U"}, "structured_pii"),
        ({**GOOD, "domande_esame": GOOD["domande_esame"][:1]}, "count"),
        ({**GOOD, "principi": []}, "count"),
    ]:
        with pytest.raises(CardRejected) as e:
            parse_card(json.dumps(bad, ensure_ascii=False))
        assert e.value.reason == reason
    with pytest.raises(CardRejected):
        parse_card("non è JSON")


@pytest.fixture
def corpus(tmp_path):
    """Twelve rulings in chunks: 2019-2024 in pairs on one appeal, and one of 2025."""
    rows, index = [], []
    for n in range(12):
        year = 2019 + n // 2
        num = f"{year}{n:06d}"
        text = f"Sentenza {num}. " + " ".join(f"fatto{n}_{i}" for i in range(200)) + \
            "\nDIRITTO\n" + " ".join(f"motivo{n}_{i}" for i in range(200))
        for i, c in enumerate(_chunks(text)):
            rows.append({"text": c, "source_id": f"cds/{num}", "sentence_id": f"cds/{num}",
                         "year": year, "kind": "cds", "chunk_index": i})
        index.append({"NUMERO_PROVVEDIMENTO": num, "NUMERO_RICORSO": f"NRG{n // 2}",
                      "NOME_SEZIONE": f"Sezione {['Quarta', 'Quinta'][n % 2]}",
                      "DATA_PUBBLICAZIONE": f"{year}-01-01", "ESITO_PROVVEDIMENTO": "Respinto"})
    rows.append({"text": "Art. 1 della legge.", "source_id": "legge/1", "sentence_id": "legge/1",
                 "kind": "legislazione", "chunk_index": 0})
    late = {"text": "Sentenza 2025.", "source_id": "cds/2025000001",
            "sentence_id": "cds/2025000001", "year": 2025, "kind": "cds", "chunk_index": 0}
    rows.append(late)
    chunks = tmp_path / "train.jsonl"
    chunks.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    og = tmp_path / "cds-sentenze.csv"
    with og.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]), delimiter=";")
        w.writeheader()
        w.writerows(index)
    return chunks, og


def test_rulings_load_whole_and_only_the_court(corpus):
    chunks, _ = corpus
    rulings = load_rulings([chunks])
    assert len(rulings) == 13 and "legge/1" not in rulings
    r = rulings["cds/2019000000"]
    assert r.text.startswith("Sentenza 2019000000.") and r.text.endswith("motivo0_199")


def test_the_split_keeps_appeals_together_and_the_test_years_out(corpus, tmp_path, capsys):
    chunks, og = corpus
    mod = _load("cds_split")
    out = tmp_path / "split"
    assert mod.main(["--chunks", str(chunks), "--openga", str(og), "--dev", "4",
                     "--out-dir", str(out)]) == 0
    dev = (out / "cds-dev-ids.txt").read_text(encoding="utf-8").split()
    train = (out / "cds-train-ids.txt").read_text(encoding="utf-8").split()
    assert len(dev) >= 4 and not set(dev) & set(train) and len(dev) + len(train) == 12
    assert "cds/2025000001" not in dev + train
    pairs = {f"cds/{2019 + n // 2}{n:06d}": n // 2 for n in range(12)}
    assert {pairs[i] for i in dev}.isdisjoint({pairs[i] for i in train})   # whole appeals
    first = json.loads((out / "cds-split.json").read_text(encoding="utf-8"))
    assert mod.main(["--chunks", str(chunks), "--openga", str(og), "--dev", "4",
                     "--out-dir", str(out)]) == 0
    assert json.loads((out / "cds-split.json").read_text(encoding="utf-8"))["dev_sha256"] == first["dev_sha256"]
    printed = capsys.readouterr()
    assert "fatto0_" not in printed.out + printed.err
    assert "after 2024" in printed.err


@pytest.fixture
def teacher():
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            user = body["messages"][-1]["content"]
            card = dict(GOOD)
            if "fatto3_" in user:      # the teacher copies a placeholder into this one
                card = {**GOOD, "principi": ["Il ricorso di [PERSONA_2] è infondato per le "
                                             "ragioni esposte in motivazione."]}
            data = json.dumps({"choices": [{"message": {"content": json.dumps(card)}}],
                               "usage": {"prompt_tokens": 1000, "completion_tokens": 300}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()


def test_cards_are_written_refused_and_not_asked_twice(corpus, teacher, tmp_path, capsys):
    chunks, og = corpus
    url, seen = teacher
    ids = tmp_path / "ids.txt"
    ids.write_text("cds/2019000000\ncds/2020000003\ncds/2021000004\ncds/nonexistent\n", encoding="utf-8")
    out = tmp_path / "cds" / "schede.jsonl"
    mod = _load("cds_schede")
    args = ["--chunks", str(chunks), "--openga", str(og), "--ids", str(ids), "--out", str(out),
            "--url", url, "--parallel", "2", "--teacher", "qwen3-30b-q8"]
    assert mod.main(args) == 0
    cards = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(c["id"] for c in cards) == ["cds/2019000000", "cds/2021000004"]
    c = next(c for c in cards if c["id"] == "cds/2019000000")
    assert c["nrg"] == "NRG0" and c["sezione"] == "Sezione Quarta" and c["numero"] == "2019000000"
    assert c["teacher"] == "qwen3-30b-q8" and c["principi"] == GOOD["principi"]
    rejects = [json.loads(line) for line in
               (out.parent / "schede.rejects.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rejects == [{"id": "cds/2020000003", "reason": "placeholder", "v": 2}]
    assert seen[0]["response_format"] == {"type": "json_object"} and seen[0]["temperature"] == 0
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    n = len(seen)
    assert mod.main(args) == 0 and len(seen) == n            # nothing asked again
    printed = capsys.readouterr()
    assert "fatto0_" not in printed.out + printed.err
    assert "rulings/min" in printed.out and "ids not in the chunk files" in printed.err


def test_the_cards_job_serves_one_teacher_per_gpu(tmp_path):
    import os
    import shutil
    import subprocess
    import sys

    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("needs POSIX bash")
    repo = Path(__file__).resolve().parents[2]
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, body in {"nvidia-smi": "printf 'GPU 0\\nGPU 1\\nGPU 2\\n'",
                       "python": f'echo "$@" > {tmp_path}/args'}.items():
        (bin_ / name).write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        (bin_ / name).chmod(0o755)
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    server.chmod(0o755)
    teacher = tmp_path / "t.gguf"
    teacher.write_bytes(b"GGUF")
    chunks = tmp_path / "train.jsonl"
    chunks.write_text("{}\n", encoding="utf-8")
    ids = tmp_path / "ids.txt"
    ids.write_text("cds/1\n", encoding="utf-8")
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(tmp_path),
           "CS_REPO": str(repo), "CS_IDS": str(ids), "CS_OUT": str(tmp_path / "o.jsonl"),
           "CS_TEACHER": str(teacher), "CS_CHUNKS": str(chunks), "LCPP_SERVER": str(server),
           "CUDA_VISIBLE_DEVICES": "1,2,3", "EULLM_VENV": str(tmp_path / "nov"),
           "HEARTBEAT_INTERVAL": "1"}
    env.pop("SLURM_JOB_ID", None)
    script = repo / "forge" / "scripts" / "leonardo" / "sbatch_cds_schede.slurm"
    r = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    args = (tmp_path / "args").read_text(encoding="utf-8").split()
    assert args[args.index("--gpus") + 1] == "1,2,3"
    assert args[args.index("--serve") + 1] == str(teacher)
    assert args[args.index("--stop-after") + 1] == "6300"
    assert "--openga" not in args and "no OpenGA CSV" in r.stderr
    (tmp_path / "args").unlink()
    r = subprocess.run(["bash", str(script)], cwd=tmp_path,
                       env={**env, "CS_TEACHER": str(tmp_path / "none.gguf")},
                       capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 1 and "no teacher GGUF" in r.stderr and not (tmp_path / "args").exists()


def test_a_second_chain_skips_what_the_first_one_carded(corpus, teacher, tmp_path):
    chunks, og = corpus
    url, seen = teacher
    ids = tmp_path / "ids.txt"
    ids.write_text("cds/2019000000\ncds/2021000004\n", encoding="utf-8")
    out = tmp_path / "cds" / "schede.jsonl"
    mod = _load("cds_schede")
    base = ["--chunks", str(chunks), "--openga", str(og), "--ids", str(ids), "--url", url]
    assert mod.main(base + ["--out", str(out), "--limit", "1"]) == 0
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])["id"]
    n = len(seen)
    other = out.with_name("schede-b.jsonl")
    assert mod.main(base + ["--out", str(other)]) == 0
    assert len(seen) == n + 1                                  # only the one not done
    assert first not in other.read_text(encoding="utf-8")


def test_odd_norms_are_dropped_and_extra_items_cut_not_the_card_refused():
    card = parse_card(json.dumps({**GOOD, "norme": [
        "art. 120 c.p.a.", "Direttiva 2014/24/UE", "R.D. n. 1265/1934", "la regola generale",
        "d.l. n. 34/2020"], "principi": GOOD["principi"] * 6}, ensure_ascii=False))
    assert card["norme"] == ["art. 120 c.p.a.", "Direttiva 2014/24/UE", "R.D. n. 1265/1934",
                             "d.l. n. 34/2020"]
    assert len(card["principi"]) == 4
    assert prefix(card, {"esito_openga": "ACCOGLIE"}).endswith("- accoglie - art. 120 c.p.a.; "
                                                               "Direttiva 2014/24/UE; "
                                                               "R.D. n. 1265/1934]")


def test_refusals_under_older_checks_are_asked_again_once(corpus, teacher, tmp_path):
    chunks, og = corpus
    url, seen = teacher
    ids = tmp_path / "ids.txt"
    ids.write_text("cds/2019000000\ncds/2021000004\n", encoding="utf-8")
    out = tmp_path / "cds" / "schede.jsonl"
    out.parent.mkdir(parents=True)
    (out.parent / "schede.rejects.jsonl").write_text(
        '{"id": "cds/2019000000", "reason": "bad_norm"}\n'
        '{"id": "cds/2021000004", "reason": "placeholder", "v": 2}\n', encoding="utf-8")
    mod = _load("cds_schede")
    assert mod.main(["--chunks", str(chunks), "--openga", str(og), "--ids", str(ids),
                     "--out", str(out), "--url", url]) == 0
    assert [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()] == ["cds/2019000000"]
    assert len(seen) == 1


def test_sparse_bm25_ranks_like_bm25_and_returns_rulings_once():
    from eullm_forge.caselaw.index import RulingIndex, SparseBM25, Unit

    units = [Unit("cds/1", "aggiudicazione termine impugnazione profilo committente"),
             Unit("cds/1", "spese di giudizio compensate"),
             Unit("cds/2", "paesaggio strutture balneari vincolo"),
             Unit("cds/3", "termine impugnazione bando di gara")]
    bm = SparseBM25([u.text for u in units])
    assert bm.ranking("impugnazione aggiudicazione", 10)[0] == 0
    assert bm.ranking("parola assente", 10) == []
    index = RulingIndex(units, bm25=bm)
    # both words in both rulings: the shorter unit first, cds/1 once despite two units
    assert index.search("termine impugnazione", k=10) == ["cds/3", "cds/1"]
    assert index.search("strutture balneari", k=1) == ["cds/2"]


def test_units_carry_the_card_prefix_and_cards_become_units(corpus):
    from eullm_forge.caselaw import attach_meta, load_openga
    from eullm_forge.caselaw.index import build_units

    chunks_path, og = corpus
    rulings = load_rulings([chunks_path])
    attach_meta(rulings, load_openga([og]))
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()
              if json.loads(line).get("kind") == "cds"]
    cards = {"cds/2019000000": {**GOOD, "esito_openga": "RESPINGE"}}
    plain = build_units(rulings, chunks)
    pre = build_units(rulings, chunks, cards=cards, prefix_chunks=True, card_units=True)
    assert len(pre) == len(plain) + 1
    first = next(u for u in pre if u.ruling == "cds/2019000000")
    assert first.text.startswith("[Cons. Stato, Sezione Quarta, n. 2019000000")
    assert "respinge" in first.text.split("\n", 1)[0]
    assert pre[-1].ruling == "cds/2019000000" and "termine per impugnare" in pre[-1].text


def test_the_retrieval_check_finds_the_rulings_its_questions_are_about(corpus, tmp_path, capsys):
    chunks, og = corpus
    questions = tmp_path / "dev-cards.jsonl"
    questions.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [
        {"id": "cds/2019000000", "domande_ricerca": ["fatto0_3 fatto0_17 motivo0_9"],
         "domande_esame": [{"domanda": "motivo0_100 motivo0_101", "risposta": "x",
                            "rubrica": "y"}]},
        {"id": "cds/2021000004", "domande_ricerca": ["fatto4_8 motivo4_150"],
         "domande_esame": []}]), encoding="utf-8")
    dev = tmp_path / "dev.txt"
    dev.write_text("cds/2019000000\ncds/2021000004\n", encoding="utf-8")
    cards = tmp_path / "schede.jsonl"
    cards.write_text(json.dumps({"id": "cds/2019000000", **GOOD}, ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "ret.csv"
    mod = _load("cds_retrieval")
    assert mod.main(["--chunks", str(chunks), "--openga", str(og), "--cards", str(cards),
                     "--questions", str(questions), "--dev-ids", str(dev),
                     "--setting", "chunks", "prefix+cards", "--csv", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert {(r["setting"], r["kind"]) for r in rows} == {
        ("chunks", "ricerca"), ("chunks", "esame"),
        ("prefix+cards", "ricerca"), ("prefix+cards", "esame")}
    assert all(float(r["recall3"]) == 1.0 for r in rows)
    # A CSV a killed link left at 0 bytes still gets its header: existence was
    # the test, so the first recall row was read *as* the header and the whole
    # table came out one row short and mislabelled. judge_answers.py and
    # legal_eval.py test size for this reason.
    fresh = tmp_path / "ret-empty.csv"
    fresh.touch()
    assert mod.main(["--chunks", str(chunks), "--openga", str(og), "--cards", str(cards),
                     "--questions", str(questions), "--dev-ids", str(dev),
                     "--setting", "chunks", "--csv", str(fresh)]) == 0
    rows = list(csv.DictReader(fresh.open(encoding="utf-8")))
    assert {(r["setting"], r["kind"]) for r in rows} == {
        ("chunks", "ricerca"), ("chunks", "esame")}
    assert all(float(r["recall3"]) == 1.0 for r in rows)
    printed = capsys.readouterr().out
    assert "fatto0_3" not in printed and "recall@3 1.000" in printed
    # a later link skips what is measured and measures only what is not
    assert mod.main(["--chunks", str(chunks), "--openga", str(og), "--cards", str(cards),
                     "--questions", str(questions), "--dev-ids", str(dev), "--limit", "2",
                     "--setting", "chunks", "prefix", "--csv", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "chunks: already in" in printed and "prefix: already" not in printed
    rows = list(csv.DictReader(out.open()))
    assert [r["setting"] for r in rows].count("chunks") == 2
    assert sum(int(r["n"]) for r in rows if r["setting"] == "prefix") == 2


def test_a_reasoning_block_before_the_card_is_skipped():
    card = parse_card("<think>\nLa sentenza riguarda...\n</think>\n" + json.dumps(GOOD))
    assert card["esito"] == "rigetto"


@pytest.fixture
def cass_corpus(tmp_path):
    """Corte di cassazione chunks as format_pretraining.py wrote them: no OpenGA."""
    path = tmp_path / "legal_it.jsonl"
    rows = []
    for n in range(10):
        kind = "snciv" if n < 8 else "snpen"
        year = 2021 + n % 4
        sid = f"{kind}/{year}/{10000 + n}"
        for i in range(2):
            rows.append({"text": f"Cassazione {n} parte{i} fatto{n}_ motivo{i}", "source_id": sid,
                         "sentence_id": sid, "chunk_index": i, "chunk_total": 2, "year": year,
                         "kind": kind})
    rows.append({"text": "Art. 1 legge", "source_id": "legge", "sentence_id": None,
                 "chunk_index": 0, "chunk_total": 1, "year": None, "kind": "legge"})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_the_cassazione_split_is_by_kind_and_named_after_it(cass_corpus, tmp_path, capsys):
    mod = _load("cds_split")
    out = tmp_path / "cass"
    assert mod.main(["--kind", "snciv", "--chunks", str(cass_corpus), "--dev", "2",
                     "--out-dir", str(out)]) == 0
    dev = (out / "snciv-dev-ids.txt").read_text(encoding="utf-8").split()
    train = (out / "snciv-train-ids.txt").read_text(encoding="utf-8").split()
    assert len(dev) >= 2 and not set(dev) & set(train) and len(dev) + len(train) == 8
    assert all(i.startswith("snciv/") for i in dev + train)
    assert json.loads((out / "snciv-split.json").read_text(encoding="utf-8"))["kind"] == "snciv"
    assert not (out / "cds-dev-ids.txt").exists()
    assert "fatto" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        mod.main(["--kind", "tar", "--chunks", str(cass_corpus), "--out-dir", str(out)])


def test_a_cassazione_card_is_asked_as_the_cassazione(cass_corpus, teacher, tmp_path):
    url, seen = teacher
    ids = tmp_path / "ids.txt"
    ids.write_text("snciv/2021/10000\nsnciv/2022/10001\n", encoding="utf-8")
    out = tmp_path / "cass" / "schede.jsonl"
    mod = _load("cds_schede")
    assert mod.main(["--kind", "snciv", "--chunks", str(cass_corpus), "--ids", str(ids),
                     "--out", str(out), "--url", url, "--parallel", "2"]) == 0
    cards = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(c["id"] for c in cards) == ["snciv/2021/10000", "snciv/2022/10001"]
    assert all(c["court"] == "snciv" and c["numero"] in ("10000", "10001") for c in cards)
    system, user = seen[0]["messages"][0]["content"], seen[0]["messages"][-1]["content"]
    assert "Massimario della Corte di cassazione" in system
    assert user.startswith("Sentenza della Corte di cassazione, sezioni civili, 202")
    assert "cassazione con rinvio" in user and "Consiglio di Stato" not in user


def test_cassazione_outcomes_and_citations():
    from eullm_forge.caselaw.courts import SNCIV
    from eullm_forge.caselaw.prompts import caselaw_prompt, ruling_label
    card = parse_card(json.dumps({**GOOD, "esito": "Cassazione con rinvio"}), SNCIV)
    assert card["esito"] == "cassazione con rinvio"
    assert parse_card(json.dumps({**GOOD, "esito": "cassazione con rinvio"}))["esito"] == "altro"
    assert prefix(card, {"numero": "12345", "year": 2023}, SNCIV).startswith(
        "[Cass. civ., n. 12345, 2023 - ")
    assert ruling_label({"numero": "12345"}, SNCIV) == "Cass. civ., n. 12345"
    assert "(sentenze della Corte di cassazione)" in caselaw_prompt("D?", [("x", "t")], SNCIV)
