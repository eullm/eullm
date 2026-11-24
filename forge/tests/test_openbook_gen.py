"""Open-book stage-3 pairs: grounded answers, absent articles, exam kept out."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from eullm_forge.datasets.instruct_gen import Rejected
from eullm_forge.datasets.openbook_gen import (
    OpenBookJob,
    exam_exclusions,
    make_openbook_jobs,
    missing_pair,
    parse_openbook,
)
from eullm_forge.eval import EvalItem, NormIndex

FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3


def rec(code, num, body):
    return {"code": code, "article_num": "", "chunk_index": 0,
            "text": f"Art. {num}. \n \n (Rubrica {num}). \n \n {body}{FILLER}"}


RECORDS = [rec("codice_civile", n, f"Disciplina numero {n} della materia civile.")
           for n in range(1, 41)]
RECORDS.append(rec("codice_civile", 41, "Il creditore agisce entro sessanta giorni."))


@pytest.fixture
def index():
    return NormIndex(RECORDS)


def exam_item(code, art, kind="contenuto"):
    return EvalItem(id=f"norm-{kind}-{code}-{art}", domain="legal", lang="it",
                    question="q", metadata={"code": code, "articolo": art, "tipo": kind})


def test_the_exams_articles_are_never_drawn(index):
    exclude = exam_exclusions([exam_item("codice_civile", str(n)) for n in range(1, 21)])
    jobs = make_openbook_jobs(index, 30, seed=1, exclude=exclude)
    grounded = {j.number for j in jobs if j.kind == "grounded"}
    assert grounded and not grounded & {str(n) for n in range(1, 21)}


def test_a_share_of_jobs_asks_about_absent_articles_past_the_end(index):
    jobs = make_openbook_jobs(index, 20, seed=2, missing_share=0.25)
    missing = [j for j in jobs if j.kind == "missing"]
    assert len(missing) == 5
    assert all(int(j.number) > 41 for j in missing)


def test_an_absent_article_is_answered_by_saying_so(index):
    job = OpenBookJob(key="m1", kind="missing", code="codice_civile", number="3500")
    pair = missing_pair(job, index)
    assert "non è presente art. 3500" in pair["instruction"]
    assert "non compare l'art. 3500" in pair["output"]
    assert pair["instruction"].endswith("?")


def test_an_existing_article_cannot_become_a_missing_pair(index):
    job = OpenBookJob(key="m2", kind="missing", code="codice_civile", number="5")
    with pytest.raises(Rejected):
        missing_pair(job, index)


def grounded_job(named=True, number="41"):
    return OpenBookJob(key=f"g-{number}", kind="grounded", code="codice_civile",
                       number=number, text=RECORDS[int(number) - 1]["text"], named=named)


def gen(q, a):
    return json.dumps({"domanda": q, "risposta": a}, ensure_ascii=False)


ANSWER = ("Secondo l'art. 41 del codice civile, il creditore deve agire entro "
          "sessanta giorni; il termine è stabilito espressamente dal testo della norma.")


def test_a_grounded_pair_carries_the_article_in_the_prompt(index):
    pair = parse_openbook(gen("Entro quanto agisce il creditore secondo l'art. 41 del "
                              "codice civile?", ANSWER), grounded_job(), index)
    assert "entro sessanta giorni" in pair["instruction"]
    assert pair["output"] == ANSWER and pair["task"] == "openbook_grounded"
    assert pair["instruction"].startswith("Testi normativi di riferimento")


def test_the_article_is_put_in_when_retrieval_misses_it(index):
    # Asked by topic in words that match other articles better, the target
    # may not be retrieved; the training prompt must still hold it.
    pair = parse_openbook(gen("Nel codice civile, quale disciplina della materia civile "
                              "vale per il creditore?", ANSWER),
                          grounded_job(named=False), index)
    assert "entro sessanta giorni" in pair["instruction"]


@pytest.mark.parametrize("q, a, named, reason", [
    ("Che cosa dice il codice civile sul creditore?", ANSWER, True, "question_not_naming"),
    ("Cosa prevede l'art. 41 del codice civile?", ANSWER, False, "question_names_article"),
    ("Cosa prevede l'art. 41 del codice civile?",
     "Il creditore deve agire entro sessanta giorni, come prevede la norma del codice.",
     True, "answer_not_citing"),
    ("Cosa prevede l'art. 41 del codice civile?", "Troppo breve.", True, "answer_length"),
])
def test_generations_that_break_the_format_are_rejected(index, q, a, named, reason):
    with pytest.raises(Rejected) as exc:
        parse_openbook(gen(q, a), grounded_job(named=named), index)
    assert exc.value.reason == reason


@pytest.fixture
def suffix_index():
    return NormIndex(RECORDS + [
        rec("codice_civile", "12", "La norma base disciplina la materia."),
        rec("codice_civile", "12-bis", "La norma bis aggiunge un caso."),
        rec("codice_civile", "12-sexies", "La norma sexies disciplina il caso speciale."),
    ])


def _suffix_job():
    return OpenBookJob(key="g-12-sexies", kind="grounded", code="codice_civile",
                       number="12-sexies", text="La norma sexies disciplina il caso speciale.")


@pytest.mark.parametrize("q, reason", [
    ("Che cosa prevede l'art. 12 del codice civile?", "question_not_naming"),
    ("Che cosa prevede l'art. 12-bis del codice civile?", "question_not_naming"),
])
def test_a_suffixed_job_is_not_answered_by_its_plain_or_other_sibling(suffix_index, q, reason):
    with pytest.raises(Rejected) as exc:
        parse_openbook(gen(q, ANSWER), _suffix_job(), suffix_index)
    assert exc.value.reason == reason


def test_a_suffixed_job_reads_the_exact_suffix(suffix_index):
    q = "Che cosa prevede l'art. 12-sexies del codice civile?"
    a = ("Secondo l'art. 12-sexies del codice civile, la norma disciplina il caso "
         "speciale e si applica alle fattispecie indicate dal testo.")
    pair = parse_openbook(gen(q, a), _suffix_job(), suffix_index)
    assert pair["key"] == "g-12-sexies"
    assert "caso speciale" in pair["instruction"]
    # the same job does not accept an answer citing only the plain article
    with pytest.raises(Rejected) as exc:
        parse_openbook(gen(q, ANSWER), _suffix_job(), suffix_index)
    assert exc.value.reason == "answer_not_citing"


def test_a_plain_job_does_not_accept_a_suffixed_citation(suffix_index):
    job = OpenBookJob(key="g-12", kind="grounded", code="codice_civile", number="12",
                      text="La norma base disciplina la materia.")
    q = "Che cosa prevede l'art. 12 del codice civile?"
    a = ("Secondo l'art. 12-bis del codice civile, la norma aggiunge un caso ulteriore "
         "e si applica alle fattispecie indicate dal testo.")
    with pytest.raises(Rejected) as exc:
        parse_openbook(gen(q, a), job, suffix_index)
    assert exc.value.reason == "answer_not_citing"


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_openbook_pairs.py"


def load_script():
    spec = importlib.util.spec_from_file_location("generate_openbook_pairs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_driver_refuses_to_run_without_an_exam_to_exclude(tmp_path):
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    assert load_script().main(["--norms", str(norms), "--out", str(tmp_path / "o.jsonl"),
                               "--limit", "5", "--dry-run"]) == 2


def test_a_dry_run_writes_the_teacher_free_pairs_and_says_how_many_it_excluded(tmp_path,
                                                                              capsys):
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    exam = tmp_path / "exam.jsonl"
    exam.write_text(json.dumps(exam_item("codice_civile", "3").to_dict()) + "\n", encoding="utf-8")
    out = tmp_path / "o.jsonl"
    mod = load_script()
    assert mod.main(["--norms", str(norms), "--exclude-exam", str(exam), "--out", str(out),
                     "--limit", "20", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "1 exam articles left out" in printed
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert lines and {p["task"] for p in lines} == {"openbook_missing"}
    # grounded jobs are still to do: a dry run never claims the data is complete
    assert not out.with_name("o.jsonl.done").exists()


def test_the_done_marker_appears_only_when_no_job_is_left(tmp_path):
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    out = tmp_path / "o.jsonl"
    mod = load_script()
    args = ["--norms", str(norms), "--no-exam", "--out", str(out), "--limit", "10"]
    jobs = make_openbook_jobs(NormIndex(RECORDS), 10, seed=0)
    # every grounded job already answered by an earlier link
    with open(out.with_name("o.jsonl.rejected.jsonl"), "w", encoding="utf-8") as f:
        for j in jobs:
            if j.kind == "grounded":
                f.write(json.dumps({"key": j.key, "reason": "x"}) + "\n")
    assert mod.main(args) == 0
    # submit_when_ready.sh tests with -s: an empty marker counts as missing
    assert out.with_name("o.jsonl.done").stat().st_size > 0


# --- more than one question per article, and jobs in parallel ----------------

def test_one_pair_per_article_is_the_original_draw(index):
    old = make_openbook_jobs(index, 20, seed=5)
    same = make_openbook_jobs(index, 20, seed=5, per_article=1)
    assert [j.key for j in old] == [j.key for j in same]


def test_several_questions_per_article_have_their_own_keys_and_focus(index):
    from eullm_forge.datasets.openbook_gen import build_messages
    jobs = [j for j in make_openbook_jobs(index, 24, seed=1, per_article=3,
                                          missing_share=0) if j.kind == "grounded"]
    keys = [j.key for j in jobs]
    assert len(keys) == len(set(keys)) == 24
    by_article = {}
    for j in jobs:
        by_article.setdefault(j.number, []).append(j)
    assert all(len(v) == 3 for v in by_article.values())
    prompts = {build_messages(j)[1]["content"] for j in by_article[next(iter(by_article))]}
    assert len(prompts) == 3                     # greedy teacher, three different asks


def test_shards_split_one_draw_without_overlap(index):
    whole = make_openbook_jobs(index, 30, seed=2, per_article=2)
    parts = [make_openbook_jobs(index, 30, seed=2, per_article=2, shard=(k, 3))
             for k in range(3)]
    keys = [j.key for p in parts for j in p]
    assert sorted(keys) == sorted(j.key for j in whole) and len(keys) == len(set(keys))


def test_a_second_question_still_counts_as_the_same_trained_article():
    from eullm_forge.eval.norm_exam import trained_articles
    assert trained_articles([{"key": "ob-g-codice_civile-2-bis-v2"},
                             {"key": "ob-g-codice_civile-41"}]) == {
        ("codice_civile", "2-bis"), ("codice_civile", "41")}


def test_the_driver_excludes_every_exam_it_is_given(tmp_path, capsys):
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    a, b = tmp_path / "v3.jsonl", tmp_path / "dev.jsonl"
    a.write_text(json.dumps(exam_item("codice_civile", "3").to_dict()) + "\n", encoding="utf-8")
    b.write_text(json.dumps(exam_item("codice_civile", "4").to_dict()) + "\n", encoding="utf-8")
    assert load_script().main(["--norms", str(norms), "--exclude-exam", str(a), str(b),
                               "--out", str(tmp_path / "o.jsonl"), "--limit", "10",
                               "--per-article", "2", "--shard", "1/2", "--dry-run"]) == 0
    assert "2 exam articles left out" in capsys.readouterr().out


# --- RAFT: the same topic question without its article ------------------------

def _topic_pair(index, number="41", named=False):
    q = ("Entro quanto agisce il creditore secondo l'art. 41 del codice civile?" if named
         else "Nel codice civile, entro quanto deve agire il creditore?")
    pair = parse_openbook(gen(q, ANSWER), grounded_job(named=named, number=number), index)
    return pair | {"key": f"ob-g-codice_civile-{number}"}


def test_an_absent_context_pair_leaves_out_the_article_and_says_so(index):
    from eullm_forge.datasets.openbook_gen import absent_context_pair
    pair = _topic_pair(index)
    assert "entro sessanta giorni" in pair["instruction"]
    absent = absent_context_pair(pair, index)
    assert absent["task"] == "openbook_absent" and absent["key"].endswith("-absent")
    assert "entro sessanta giorni" not in absent["instruction"]
    assert "Domanda: Nel codice civile, entro quanto deve agire il creditore?" \
        in absent["instruction"]
    assert "non trovo la disposizione" in absent["output"]
    assert "del codice civile" in absent["output"]


def test_a_question_that_names_its_article_is_never_turned(index):
    from eullm_forge.datasets.openbook_gen import absent_context_pair
    assert absent_context_pair(_topic_pair(index, named=True), index) is None


def test_an_absent_pair_removes_the_continuation_chunks_too():
    """The pair teaches "the texts do not contain the answer".

    A chunk that continues the removed article carries no header of its own,
    so record_articles reported no article at all for it and it stayed in the
    context -- with most of a long article's text, the answer included. The
    index resolves continuations in articles_of, and says itself that any
    caller removing an article has to use it.
    """
    from eullm_forge.datasets.openbook_gen import absent_context_pair
    from eullm_forge.eval.retrieval import NormIndex

    records = [
        {"code": "codice_civile", "article_num": "", "chunk_index": 0,
         "text": "Art. 41. \n \n (Rubrica quarantuno). \n \n Il creditore e il debitore."},
        {"code": "codice_civile", "article_num": "", "chunk_index": 1,
         "text": "Il creditore agisce entro sessanta giorni dalla notifica."},
        {"code": "codice_civile", "article_num": "", "chunk_index": 0,
         "text": "Art. 42. \n \n (Rubrica quarantadue). \n \n "
                 "Il debitore deve dare preavviso al creditore nel codice civile."},
    ]
    index = NormIndex(records)
    pair = {"task": "openbook_grounded", "named": False, "key": "ob-g-codice_civile-41",
            "instruction": "Domanda: Nel codice civile, entro quanto deve agire il creditore?"}
    absent = absent_context_pair(pair, index)
    assert absent["task"] == "openbook_absent"
    assert "sessanta giorni" not in absent["instruction"]
    assert "Art. 42" in absent["instruction"]  # the other article stays


def test_the_raft_script_keeps_every_original_and_adds_the_share(tmp_path, index):
    import importlib.util
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    pairs = [_topic_pair(index) | {"key": f"ob-g-codice_civile-41-v{i}" if i else
                                   "ob-g-codice_civile-41"} for i in range(4)]
    pairs.append(_topic_pair(index, named=True))
    src = tmp_path / "v04.jsonl"
    src.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
                   encoding="utf-8")
    out = tmp_path / "v05.jsonl"
    spec = importlib.util.spec_from_file_location(
        "make_raft_absent", Path(__file__).resolve().parents[1] / "scripts" / "make_raft_absent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([str(src), "--norms", str(norms), "--share", "0.5", "--out", str(out)]) == 0
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5 + 2
    assert sum(r["task"] == "openbook_absent" for r in rows) == 2
    assert mod.main([str(src), "--norms", str(norms), "--out", str(src)]) == 2


def test_a_grounded_context_keeps_the_continuation_retrieval_found():
    """The answer of a long article is in its continuation chunk.

    _context asked record_articles whether the article was among the
    passages; a continuation has no header of its own, so the article looked
    absent, and the header chunk was put in place of the last passage -- the
    continuation itself, here -- leaving a grounded pair without its answer.
    """
    import random

    from eullm_forge.datasets.openbook_gen import OpenBookJob, _context
    from eullm_forge.eval.retrieval import NormIndex

    records = [
        {"code": "codice_civile", "article_num": "", "chunk_index": 0,
         "text": "Art. 41. \n \n (Rubrica quarantuno). \n \n Il creditore e il debitore."},
        {"code": "codice_civile", "article_num": "", "chunk_index": 1,
         "text": "Il creditore agisce entro sessanta giorni dalla notifica."},
        {"code": "codice_civile", "article_num": "", "chunk_index": 0,
         "text": "Art. 42. \n \n (Rubrica quarantadue). \n \n Il debitore paga."},
    ]
    index = NormIndex(records)
    job = OpenBookJob(key="ob-g-codice_civile-41", kind="grounded", code="codice_civile",
                      number="41", named=False)
    found, _ = _context(index, "Entro quanti giorni agisce il creditore dalla notifica?", job,
                        1, random.Random(0))
    assert any("sessanta giorni" in r["text"] for r in found)


def test_abstain_prompts_teach_the_difference_and_hide_the_note_from_the_student(tmp_path):
    """2026-10-08: with no texts, legal-it-8b invented articles."""
    import importlib.util

    from eullm_forge.eval.retrieval import open_book_prompt

    records = [
        {"code": "codice_civile", "article_num": "", "chunk_index": 0,
         "text": "Art. 41. \n \n (Rubrica quarantuno). \n \n Il creditore e il debitore."},
        {"code": "codice_civile", "article_num": "", "chunk_index": 1,
         "text": "Il creditore agisce entro sessanta giorni dalla notifica."},
        {"code": "codice_civile", "article_num": "", "chunk_index": 0,
         "text": "Art. 42. \n \n (Rubrica quarantadue). \n \n "
                 "Il debitore deve dare preavviso al creditore nel codice civile."},
    ]
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                     encoding="utf-8")
    question = "Nel codice civile, entro quanto deve agire il creditore?"
    pair = {"task": "openbook_grounded", "named": False, "key": "ob-g-codice_civile-41",
            "instruction": open_book_prompt(question, records[:2]), "output": "x"}
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(pair, ensure_ascii=False) + "\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_abstain_prompts.py"
    spec = importlib.util.spec_from_file_location("make_abstain_prompts", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "abstain.jsonl"
    assert mod.main([str(pairs), "--norms", str(norms), "--absent", "1", "--grounded", "1",
                     "--no-text", "1", "--out", str(out)]) == 0
    rows = {r["kind"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    assert set(rows) == {"absent", "statute", "no_text"}
    absent = rows["absent"]
    assert "sessanta giorni" not in absent["student"][0]["content"]   # the article is gone
    assert absent["teacher"][0]["content"].startswith(mod.ABSENT_NOTE)
    assert "Nota riservata" not in absent["student"][0]["content"]
    assert "sessanta giorni" in rows["statute"]["student"][0]["content"]
    assert rows["no_text"]["student"][0]["content"] == question
    assert rows["no_text"]["teacher"][0]["content"].startswith(mod.NO_TEXT_NOTE)


# --- the engine as teacher (JUPITER) -------------------------------------------

def test_the_engine_backend_writes_grounded_pairs(tmp_path):
    import re
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(self.path)
            n = re.search(r"art\. (\S+) ", body["messages"][-1]["content"]).group(1)
            answer = (f"Secondo l'art. {n} la disciplina numero {n} regola la materia "
                      "civile con disposizioni di dettaglio, come dice il testo.")
            content = json.dumps({"domanda": f"Che cosa prevede l'art. {n} del codice civile?",
                                  "risposta": answer})
            out = json.dumps({"message": {"role": "assistant", "content": content},
                              "eval_count": 9, "done": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        norms = tmp_path / "legislazione_x.chunks.jsonl"
        norms.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
        exam = tmp_path / "exam.jsonl"
        exam.write_text(json.dumps(exam_item("codice_civile", "3").to_dict()) + "\n",
                        encoding="utf-8")
        out = tmp_path / "o.jsonl"
        assert load_script().main([
            "--norms", str(norms), "--exclude-exam", str(exam), "--out", str(out),
            "--limit", "20", "--backend", "eullm", "--model", "m",
            "--servers", f"http://127.0.0.1:{srv.server_address[1]}", "--concurrency", "4",
        ]) == 0
    finally:
        srv.shutdown()
    pairs = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    grounded = [p for p in pairs if p["task"] == "openbook_grounded"]
    assert seen and set(seen) == {"/api/chat"}
    # the stub always names the article: kept when the job asked for it, rejected otherwise
    assert grounded and all(p["named"] for p in grounded)
    assert all(p["key"].split("-v")[0] != "ob-g-codice_civile-3" for p in pairs)
    assert out.with_name("o.jsonl.done").exists()
