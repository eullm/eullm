"""The LEGAL_PACK's statutes, and the end-to-end exam through RAG Enterprise's API.

The pack is built from records in the shapes Normattiva's export comes in
(an article in one chunk, an article continued in the next chunk); the exam
runs against a stub of RAG Enterprise's three endpoints, served locally, so
what is pinned is the script's side of the API as RAG Enterprise defines it
(src/api/auth.rs, documents.rs, query.rs of I3K-IT/RAG-Enterprise).
"""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norms(tmp_path: Path) -> Path:
    recs = [
        {"code": "codice_penale", "article_num": "", "chunk_index": 0,
         "text": "Art. 54. \n \n (Stato di necessità). \n \n Non è punibile chi ha commesso il "
                 "fatto per esservi stato costretto dalla necessità di salvare sé od altri."
                 + FILLER},
        {"code": "codice_penale", "article_num": "", "chunk_index": 1,
         "text": "Questa disposizione non si applica a chi ha un particolare dovere giuridico."},
        {"code": "codice_penale", "article_num": "", "chunk_index": 0,
         "text": "Art. 624. \n \n (Furto). \n \n Chiunque s'impossessa della cosa mobile altrui "
                 "è punito con la reclusione." + FILLER},
        {"code": "legge_procedimento_amministrativo", "article_num": "Art. 3.", "chunk_index": 0,
         "text": "Art. Art. 3. ((Motivazione del provvedimento))\nOgni provvedimento "
                 "amministrativo deve essere motivato." + FILLER},
    ]
    p = tmp_path / "legislazione_x.chunks.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                 encoding="utf-8")
    return p


def test_one_file_per_article_named_as_it_is_cited(tmp_path, capsys):
    out = tmp_path / "pack"
    assert _load("build_legal_pack").main(["--norms", str(_norms(tmp_path)), "--vigente",
                                           "2026-09-15", "--out", str(out)]) == 0
    files = sorted(p.name for p in (out / "articoli").glob("*.txt"))
    assert files == ["Codice penale, art. 54 - Stato di necessità.txt",
                     "Codice penale, art. 624 - Furto.txt",
                     "Legge n. 241-1990, art. 3 - Motivazione del provvedimento.txt"]
    text = (out / "articoli" / files[0]).read_text(encoding="utf-8")
    assert text.startswith("Codice penale, art. 54 (Stato di necessità)\n\n")
    assert "particolare dovere giuridico" in text          # the continuation chunk too
    rows = [json.loads(ln) for ln in (out / "legal-pack.jsonl").read_text().splitlines()]
    assert {r["id"] for r in rows} == {"codice_penale-54", "codice_penale-624",
                                       "legge_procedimento_amministrativo-3"}
    assert all(r["vigente_al"] == "2026-09-15" for r in rows)
    assert "3 articles in force at 2026-09-15" in capsys.readouterr().out


def test_file_names_are_safe_everywhere_and_never_collide():
    mod = _load("build_legal_pack")
    assert mod.file_name('Codice civile, art. 1 - "Fonti": a/b?') == \
        "Codice civile, art. 1 - Fonti a b.txt"
    assert len(mod.file_name("Codice civile, art. 1 - " + "x" * 400)) <= mod.MAX_NAME + 4


def test_citations_outside_the_sources_and_abstentions_are_told_apart():
    mod = _load("rag_enterprise_eval")
    sources = ["Codice penale, art. 54 - Stato di necessità.txt", "Codice penale, art. 626.txt"]
    ok = mod.check_sources("Lo stato di necessità [Codice penale, art. 54 - Stato di necessità] "
                           "esclude la punibilità (art. 54).", sources)
    assert ok["sources_ok"] and ok["cited_articles"] == ["54"]
    bad = mod.check_sources("Secondo l'art. 49 c.p. il furto resta punibile.", sources)
    assert not bad["sources_ok"] and bad["outside_sources"] == ["art. 49"]
    # the article asked about, named to say it is missing, is not cited from memory
    asked = mod.check_sources("L'art. 2875 del codice civile non è presente nei dati forniti.",
                              sources, "Che cosa prevede l'art. 2875 del codice civile?")
    assert asked["sources_ok"] and asked["outside_sources"] == []
    assert mod.abstained("Nei testi disponibili non ho trovato la norma che risponde.")
    assert not mod.abstained("Secondo l'art. 54 c.p. non è punibile.")


class _Stub(BaseHTTPRequestHandler):
    docs: list[str] = []
    asked: list[str] = []

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.headers["Authorization"] == "Bearer tok"
        self._send({"documents": [{"id": "1", "filename": n} for n in self.docs]})

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/api/auth/login":
            creds = json.loads(body)
            return self._send({"access_token": "tok", "token_type": "bearer", "user": {}}
                              if creds["password"] == "segreta" else {"error": "no"},
                              200 if creds["password"] == "segreta" else 401)
        assert self.headers["Authorization"] == "Bearer tok"
        if self.path == "/api/documents/upload":
            name = body.split(b'filename="')[1].split(b'"')[0].decode()
            self.docs.append(name)
            return self._send({"ok": True})
        q = json.loads(body)["query"]
        self.asked.append(q)
        if "inesistente" in q:
            return self._send({"answer": "Nei testi disponibili non ho trovato la norma.",
                               "sources": [{"filename": "Codice penale, art. 54.txt"}]})
        return self._send({"answer": "Secondo l'art. 54 c.p. [Codice penale, art. 54] non è "
                                     "punibile.",
                           "sources": [{"filename": "Codice penale, art. 54.txt"}]})


@pytest.fixture
def server():
    _Stub.docs, _Stub.asked = ["Codice penale, art. 54.txt"], []
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_the_exam_goes_through_the_api_and_resumes(server, tmp_path, monkeypatch, capsys):
    mod = _load("rag_enterprise_eval")
    monkeypatch.setenv("RAG_PASSWORD", "segreta")
    pack = tmp_path / "articoli"
    pack.mkdir()
    for n in ("Codice penale, art. 54.txt", "Codice penale, art. 624.txt"):
        (pack / n).write_text("testo", encoding="utf-8")
    assert mod.main(["upload", "--url", server, "--pack", str(pack)]) == 0
    assert _Stub.docs == ["Codice penale, art. 54.txt", "Codice penale, art. 624.txt"]

    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps(it) + "\n" for it in [
        {"id": "norm-contenuto-codice_penale-54", "question": "Cosa prevede l'art. 54?",
         "reference": "r", "rubric": "",
         "metadata": {"tipo": "contenuto", "code": "codice_penale", "articolo": "54"}},
        {"id": "norm-inesistente-codice_penale-9999", "question": "art. inesistente 9999?",
         "reference": "r", "rubric": "", "metadata": {"tipo": "inesistente"}}]))
    out = tmp_path / "answers-x.jsonl"
    base = ["ask", "--url", server, "--label", "x", "--items", str(items), "--out", str(out)]
    assert mod.main(base + ["--limit", "1"]) == 0
    assert mod.main(base) == 0                               # carries on, asks only the second
    assert len(_Stub.asked) == 2
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert [r["abstained"] for r in rows] == [False, True]
    assert rows[0]["sources_ok"] and rows[0]["question"] == "Cosa prevede l'art. 54?"

    graded = tmp_path / "answers-x.graded.jsonl"
    graded.write_text("".join(json.dumps({**r, "grade": "correct"}) + "\n" for r in rows))
    capsys.readouterr()
    assert mod.main(["summary", str(graded)]) == 0
    printed = capsys.readouterr().out
    assert "precision 1.000 (1/1 answered)" in printed
    assert "absent articles abstained 1/1" in printed
    # the stub's sources are the article asked, named as the pack names it
    assert "article asked among the sources 1/1" in printed


def test_the_summary_reads_an_ungraded_file_and_names_the_code_as_the_pack_does(tmp_path,
                                                                             capsys):
    mod = _load("rag_enterprise_eval")
    assert mod.CODE_NAMES == _load("build_legal_pack").CODE_NAMES
    p = tmp_path / "answers-x.jsonl"
    p.write_text(json.dumps({
        "id": "norm-contenuto-codice_civile-2875", "question": "Che cosa prevede l'art. 2875 c.c.?",
        "metadata": {"tipo": "contenuto", "code": "codice_civile", "articolo": "2875"},
        "answer": "L'art. 2875 del codice civile non è presente nei dati forniti.",
        "sources": ["Codice civile, art. 2475 - Amministrazione della societa'.txt",
                    "Codice penale, art. 2875.txt"],
        "abstained": False, "sources_ok": False}) + "\n")
    assert mod.main(["summary", str(p)]) == 0
    out = capsys.readouterr().out
    assert "precision n/a (not graded; 0 answered)" in out
    assert "citing outside the sources 0" in out and "abstained 1" in out
    assert "article asked among the sources 0/1" in out


def test_the_password_never_comes_from_the_command_line(monkeypatch, tmp_path):
    mod = _load("rag_enterprise_eval")
    monkeypatch.delenv("RAG_PASSWORD", raising=False)
    with pytest.raises(SystemExit, match="RAG_PASSWORD"):
        mod.main(["upload", "--pack", str(tmp_path)])
