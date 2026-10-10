#!/usr/bin/env python3
"""Ask an exam through RAG Enterprise's API, as a user would: the product end to end.

    # 1. once: load the LEGAL_PACK's article files into RAG Enterprise
    RAG_PASSWORD=... python forge/scripts/rag_enterprise_eval.py upload \\
        --url http://localhost:8000 --user admin --pack legal-pack/articoli

    # 2. per configuration (model, prompt, version): ask every question
    RAG_PASSWORD=... python forge/scripts/rag_enterprise_eval.py ask \\
        --url http://localhost:8000 --user admin --label community-qwen3-14b \\
        --items norm-exam-devbig.jsonl --out answers-community-qwen3-14b.jsonl

    # 3. after judge_answers.py has graded the answers (on Leonardo)
    python forge/scripts/rag_enterprise_eval.py summary answers-*.graded.jsonl \\
        --pack legal-pack/articoli

The models were measured until 2026-10-08 only outside the product, and the
product was never measured at all: asked in a plain chat with no texts,
legal-it-8b invented articles. This measures what a user gets, with the
release thresholds of the LEGAL_PACK (eullm-priv docs/legal-pack-spec.md, §2):

* **precision**: correct answers over answers given (abstentions left out);
* **coverage**: answers given over questions, on questions about articles
  that exist;
* **sources**: answers citing a file that was not among the retrieved
  sources -- with the pack's one-article-per-file names, a citation outside
  them is an invented article;
* **abstention** where the article does not exist (the exam's
  ``inesistente`` items).

Standard library only, so it runs on the machine RAG Enterprise runs on,
Windows included. The password comes from ``RAG_PASSWORD``, never the
command line. ``ask`` appends one line per answer and skips the ids already
in ``--out``, so an interrupted run carries on. It prints counts only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# The same pattern as eullm_forge.eval.abstain.ABSTAIN (a test keeps them
# equal): this file imports nothing from the package, so it runs alone.
ABSTAIN = re.compile(
    r"non (?:ho trovato|trovo)|non contengono|non dispongo|"
    r"non (?:è|sono) present[ei] (?:nei|tra i|nella raccolta)|"
    r"non risulta(?:no)? (?:nei|dai|tra i) (?:testi|documenti)|"
    r"non (?:è|sono) (?:indicat|previst|riportat)[oaie] nei (?:testi|documenti)|"
    r"nei (?:testi|documenti) (?:forniti|disponibili|riportati) non|"
    r"non posso (?:quindi )?(?:dar\w*|fornir\w*|rispondert?\w*)[^.]{0,40}"
    r"(?:certezza|sicur[oa])|"
    r"senza (?:il|i) test[oi] dell[ae] norm|"
    r"no relevant information|informazioni rilevanti", re.IGNORECASE)
_BRACKETS = re.compile(r"\[([^\[\]]{3,200})\]")
_ART = re.compile(r"\bart(?:icolo|\.)?\s*(\d+(?:[- ]?(?:bis|ter|quater|quinquies|sexies|"
                  r"septies|octies|novies|decies))?)", re.IGNORECASE)


# As build_legal_pack.CODE_NAMES names each code in a file name (a test keeps
# the two equal): this file imports nothing from the package.
CODE_NAMES = {
    "codice_civile": "Codice civile",
    "codice_penale": "Codice penale",
    "codice_procedura_civile": "Codice di procedura civile",
    "codice_procedura_penale": "Codice di procedura penale",
    "codice_consumo": "Codice del consumo",
    "costituzione": "Costituzione",
    "codice_processo_amministrativo": "Codice del processo amministrativo",
    "legge_procedimento_amministrativo": "Legge n. 241-1990",
    "ricorsi_amministrativi": "D.P.R. n. 1199-1971",
}


def _norm_number(n: str) -> str:
    return re.sub(r"[- ]+", "-", n.strip().lower())


def abstained(answer: str) -> bool:
    return bool(ABSTAIN.search(answer or ""))


def _head(name: str) -> str:
    """A pack file's citation without its rubrica: "codice civile, art. 743"."""
    name = re.sub(r"\s+", " ", name.strip().lower().removesuffix(".txt"))
    return name.split(" - ", 1)[0]


_NEXT_PREV = re.compile(r"\b(?:art(?:icolo|\.)?)\s+(precedente|seguente|successivo)\b",
                        re.IGNORECASE)


def _neighbours(text: str) -> set[str]:
    """Articles a text refers to by position: "a norma dell'articolo
    precedente" in art. 1048 c.c. is art. 1047. The text's own number is its
    first citation, the heading the pack writes ("Codice civile, art. 1048")."""
    own = _ART.search(text or "")
    if not own:
        return set()
    base = int(re.match(r"\d+", own.group(1)).group())
    out = set()
    for m in _NEXT_PREV.finditer(text):
        n = base - 1 if m.group(1).lower() == "precedente" else base + 1
        if n > 0:
            out.add(str(n))
    return out


def check_sources(answer: str, sources: list[str], question: str = "",
                  source_texts: list[str] = ()) -> dict:
    """Which files the answer cites, and whether each is among the sources.

    Two kinds of citation: ``[file name]``, RAG Enterprise's own, which must
    be one of the retrieved files -- the same code and article, since a model
    copying "Societa' contratta con l'erede" as "contrattacon" still cites
    art. 743; and ``art. N`` in the text, whose number must be the article of
    some retrieved file (the pack's file names carry it, "..., art. 54 - ...")
    or an article the retrieved texts themselves refer to: art. 1484 c.c.
    sends the reader to art. 1480, and citing it is reading, not memory.
    """
    names = {s.lower().removesuffix(".txt") for s in sources}
    heads = {_head(s) for s in sources if _ART.search(_head(s))}
    numbers = set()
    for s in list(sources) + list(source_texts):
        numbers.update(_norm_number(m) for m in _ART.findall(s))
    for t in source_texts:
        numbers.update(_neighbours(t))
    cited = [c.strip() for c in _BRACKETS.findall(answer or "")]
    bad = [c for c in cited
           if c.lower().removesuffix(".txt") not in names and _head(c) not in heads]
    arts = sorted({_norm_number(m) for m in _ART.findall(_BRACKETS.sub(" ", answer or ""))})
    # The article the question asks about is not cited from memory: "l'art.
    # 2875 non è presente nei dati forniti" names it in order to abstain.
    asked = {_norm_number(m) for m in _ART.findall(question or "")}
    bad_arts = [a for a in arts if a not in numbers and a not in asked]
    return {"cited_files": cited, "cited_articles": arts,
            "sources_ok": not bad and not bad_arts,
            "outside_sources": bad + [f"art. {a}" for a in bad_arts]}


class Api:
    """The few RAG Enterprise endpoints this needs, with a bearer token."""

    def __init__(self, url: str, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.token = ""

    def _call(self, method: str, path: str, body: bytes | None = None,
              ctype: str = "application/json") -> dict | list:
        req = urllib.request.Request(self.url + path, data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", ctype)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                # 503: a document is being ingested and the model is unloaded
                if e.code == 503 and attempt < 2:
                    time.sleep(10)
                    continue
                raise SystemExit(f"[rag-eval] {method} {path}: HTTP {e.code} "
                                 f"{e.read()[:200].decode('utf-8', 'replace')}") from None
        raise SystemExit(f"[rag-eval] {method} {path}: still busy")

    def login(self, user: str, password: str) -> None:
        r = self._call("POST", "/api/auth/login",
                       json.dumps({"username": user, "password": password}).encode())
        self.token = r["access_token"]

    def documents(self) -> list[str]:
        r = self._call("GET", "/api/documents")
        rows = r.get("documents", r) if isinstance(r, dict) else r
        return [d.get("filename") or d.get("name") or "" for d in rows]

    def upload(self, path: Path) -> None:
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{path.name}\"\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
                ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        self._call("POST", "/api/documents/upload", body,
                   f"multipart/form-data; boundary={boundary}")

    def query(self, question: str, top_k: int | None) -> dict:
        payload = {"query": question}
        if top_k:
            payload["top_k"] = top_k
        return self._call("POST", "/api/query", json.dumps(payload).encode())


def _api(args) -> Api:
    password = os.environ.get("RAG_PASSWORD")
    if not password:
        raise SystemExit("[rag-eval] set RAG_PASSWORD (the password is never taken from "
                         "the command line)")
    api = Api(args.url)
    api.login(args.user, password)
    return api


def cmd_upload(args) -> int:
    api = _api(args)
    have = {n.lower() for n in api.documents()}
    files = sorted(p for p in args.pack.glob("*.txt"))
    todo = [p for p in files if p.name.lower() not in have]
    print(f"[rag-eval] {len(files):,} article files, {len(files) - len(todo):,} already loaded",
          flush=True)
    t0 = time.monotonic()
    for i, p in enumerate(todo, 1):
        api.upload(p)
        if i % 100 == 0 or i == len(todo):
            print(f"[rag-eval] uploaded {i:,}/{len(todo):,} "
                  f"({(time.monotonic() - t0) / i:.1f} s each)", flush=True)
    return 0


def cmd_ask(args) -> int:
    items = [json.loads(ln) for ln in args.items.open(encoding="utf-8") if ln.strip()]
    if args.limit:
        items = items[:args.limit]
    done = set()
    if args.out.exists():
        done = {json.loads(ln)["id"] for ln in args.out.open(encoding="utf-8") if ln.strip()}
    todo = [it for it in items if it["id"] not in done]
    print(f"[rag-eval] {args.label}: {len(items):,} questions, {len(done):,} already answered",
          flush=True)
    api = _api(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with args.out.open("a", encoding="utf-8") as f:
        for i, it in enumerate(todo, 1):
            r = api.query(it["question"], args.top_k)
            answer = r.get("answer", "")
            sources = sorted({s.get("filename", "") for s in r.get("sources", [])})
            row = {**it, "label": args.label, "answer": answer, "sources": sources,
                   "abstained": abstained(answer),
                   **check_sources(answer, sources, it["question"])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"[rag-eval] {i:,}/{len(todo):,} ({(time.monotonic() - t0) / i:.1f} s each)",
                      flush=True)
    return 0


def asked_retrieved(row: dict) -> bool | None:
    """Whether the article the item is about is among the sources, or None for
    an item about no article in the pack (``inesistente``, or no metadata).

    A file holds one article and is named "<code>, art. <N> - <rubrica>.txt".
    """
    meta = row.get("metadata") or {}
    code, number = meta.get("code"), meta.get("articolo")
    if not code or not number or meta.get("tipo") == "inesistente":
        return None
    want = _norm_number(str(number))
    for s in row.get("sources", []):
        head = s.split(" - ", 1)[0]
        if head.startswith(CODE_NAMES.get(code, code) + ",") and \
                {_norm_number(m) for m in _ART.findall(head)} == {want}:
            return True
    return False


def pack_texts(pack: Path | None):
    """A reader of the pack's article files by name, or of nothing without one."""
    cache: dict[str, str] = {}

    def text(name: str) -> str:
        if pack is None:
            return ""
        if name not in cache:
            p = pack / name
            cache[name] = p.read_text(encoding="utf-8") if p.is_file() else ""
        return cache[name]
    return text


def summarize(rows: list[dict], pack: Path | None = None) -> dict:
    """The LEGAL_PACK's release figures from graded answers (judge_answers.py output).

    Abstention and sources are read again from the answer with the current
    checks, so answers collected before a fix of the checks still count right.
    With ``pack`` (the ``articoli/`` folder that was uploaded), the articles
    the retrieved files refer to count as in hand too.
    """
    text = pack_texts(pack)
    rows = [{**r, "abstained": abstained(r.get("answer", "")),
             **check_sources(r.get("answer", ""), r.get("sources", []), r.get("question", ""),
                             [text(n) for n in r.get("sources", [])])}
            if "sources" in r else r for r in rows]
    absent = [r for r in rows if (r.get("metadata") or {}).get("tipo") == "inesistente"]
    real = [r for r in rows if r not in absent]
    answered = [r for r in real if not r.get("abstained")]
    correct = [r for r in answered if r.get("grade") == "correct"]
    return {
        "questions": len(rows),
        "coverage": len(answered) / len(real) if real else 0.0,
        "precision": len(correct) / len(answered) if answered else 0.0,
        "answered": len(answered), "correct": len(correct),
        "outside_sources": sum(not r.get("sources_ok", True) for r in rows),
        "absent_items": len(absent),
        "abstained": sum(bool(r.get("abstained")) for r in rows),
        "graded": any("grade" in r for r in rows),
        "asked_retrieved": sum(asked_retrieved(r) is True for r in rows),
        "asked_items": sum(asked_retrieved(r) is not None for r in rows),
        "absent_abstained": sum(bool(r.get("abstained")) for r in absent),
    }


def cmd_summary(args) -> int:
    for p in args.graded:
        rows = [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]
        s = summarize(rows, args.pack)
        precision = (f"precision {s['precision']:.3f} ({s['correct']}/{s['answered']} answered)"
                     if s["graded"] else f"precision n/a (not graded; {s['answered']} answered)")
        print(f"{p.name}: {s['questions']} questions | {precision} | "
              f"coverage {s['coverage']:.3f} | "
              f"citing outside the sources {s['outside_sources']} | "
              f"absent articles abstained {s['absent_abstained']}/{s['absent_items']} | "
              f"article asked among the sources {s['asked_retrieved']}/{s['asked_items']} | "
              f"abstained {s['abstained']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("upload", "ask"):
        p = sub.add_parser(name)
        p.add_argument("--url", default="http://localhost:8000")
        p.add_argument("--user", default="admin")
    sub.choices["upload"].add_argument("--pack", type=Path, required=True,
                                       help="the pack's articoli/ folder")
    a = sub.choices["ask"]
    a.add_argument("--label", required=True, help="the configuration: model, prompt, version")
    a.add_argument("--items", type=Path, required=True, help="exam items (EvalItem JSONL)")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--top-k", type=int, default=0, help="0: RAG Enterprise's own default")
    a.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("summary")
    s.add_argument("graded", nargs="+", type=Path)
    s.add_argument("--pack", type=Path,
                   help="the pack's articoli/ folder: articles the sources refer to count "
                        "as in hand, not as cited from memory")
    args = ap.parse_args(argv)
    return {"upload": cmd_upload, "ask": cmd_ask, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
