"""The legislation written so that a model holds it: the memory corpus.

The legal-memory plan (``docs/research/legal-memory-plan.md``, H1) trains a
model to recall the law with no text in front of it. Work on knowledge
capacity finds a fact becomes recallable only after many exposures in varied
forms, not after one pass over the raw text, so every article is written
several ways:

* by program (`program_rows`), from what an article states and nothing else:
  the text itself, a table of contents of each code, heading to number and
  back, its deadlines, penalty, procedibility, repeal, and the numbers a code
  does NOT have ("there is no art. 640-septies: after 640-sexies comes 641");
* by a teacher (`rewrite_messages`, forge/scripts/law_rewrites.py): the rule
  in other words, explained, applied to a case, as questions and answers, as
  a list of elements. Every rewrite is checked against its article
  (`check_rewrite`) before it is kept: it names the article, and it brings no
  other article, no number and no deadline or penalty the article does not
  have. A teacher that adds what it remembers is refused, not believed.

What is held out (`excluded`), and why that is not the articles. The
closed-book suite (``eval/closed_book.py``) asks about articles, and a model
can only know an article it was trained on; holding the articles out would
measure nothing. What is held out is the *question*: the made-up numbers the
suite asks about are never said not to exist, and an article the suite or
another evaluation asks about gets no question-and-answer rewrite. The
knowledge goes in as statements; whether it comes out as answers to
questions the model never trained on is what the suite measures, as in the
knowledge-capacity work (biographies trained, questions held out).
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from ..eval.closed_book import _concept, cited_articles
from ..eval.norm_exam import CODE_LABELS, Article, _deadlines, all_deadlines, split_sentences
from ..rl.rewards import mentioned_deadlines
from .premise_gen import (
    SUFFIXES,
    _repealed_sentence,
    penalty_of,
    procedibility_of,
    say,
    unused_number,
)

FORMS = ("testo", "indice", "rubrica", "termine", "pena", "procedibilita", "abrogato",
         "inesistente")
STYLES = ("parafrasi", "spiegazione", "caso", "domande", "elementi")
# What is asked unless a run names its styles. Not the case: applying a rule
# to facts is where the teacher errs, and no check by program sees it. Of six
# cases read on 2026-10-29 three were wrong in law (a false attribution called
# an error on the nature of the object, art. 1429 c.c.; art. 540-bis c.p.c.,
# on movable property, applied to land; two rules of art. 50-bis c.p.c.
# mixed), while the paraphrases and the questions read were right.
DEFAULT_STYLES = ("parafrasi", "spiegazione", "domande", "elementi")
# Raised whenever check_rewrite changes: refusals of an older check are asked again.
CHECKS_VERSION = 1
# Raised whenever a request changes, and written in every rewrite ("pv"), so the
# rewrites of an older request can be told apart. 2 (2026-10-29): the
# explanation may not place the article in an institute the text does not
# name; one of three read had art. 2553 c.c., of the associazione in
# partecipazione, about the partners of a company.
PROMPT_VERSION = 2
INDEX_SPAN = 15             # articles per table-of-contents block
MAX_ARTICLE_CHARS = 6000    # longer articles are written by program only

RUBRICA_FWD = [
    "L'art. {n} {of} è intitolato «{h}».",
    "La rubrica dell'art. {n} {of} è «{h}».",
    "L'articolo {n} {of} porta la rubrica «{h}».",
]
RUBRICA_BACK = [
    "{In} la disciplina di «{h}» si trova all'art. {n}.",
    "«{h}» è la materia dell'art. {n} {of}.",
    "{In} l'articolo dedicato a «{h}» è il {n}.",
]


def code_name(code: str) -> str:
    """"Codice penale", "Costituzione", "Legge n. 241/1990": the code as a title."""
    of = CODE_LABELS[code][0]
    name = re.sub(r"^(?:del|della|dello|dell['’])\s*", "", of)
    return name[:1].upper() + name[1:]


def _plain(text: str) -> str:
    return " ".join(text.replace("((", " ").replace("))", " ").split())


def _body(article: Article) -> str:
    """The article's text after its number and heading."""
    flat = _plain(article.text)
    return re.sub(r"^\s*Art\.?\s*[\w-]+\.?\s*(?:\([^)]*\)\.?\s*)?", "", flat).strip()


def number_key(number: str) -> tuple[int, int]:
    """Order of articles in a code: 99, 99-bis, 99-ter, 100."""
    base, _, suf = number.partition("-")
    m = re.match(r"\d+", base)
    rank = SUFFIXES.index(suf) + 1 if suf in SUFFIXES else (0 if not suf else len(SUFFIXES) + 1)
    return (int(m.group()) if m else 10**6, rank)


def by_code(articles: dict[tuple[str, str], Article]) -> dict[str, list[Article]]:
    """The articles of each code we have a label for, in the code's order."""
    out: dict[str, list[Article]] = defaultdict(list)
    for a in articles.values():
        if a.code in CODE_LABELS:
            out[a.code].append(a)
    return {c: sorted(v, key=lambda a: number_key(a.number)) for c, v in sorted(out.items())}


def excluded(paths: list[Path]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """(made-up numbers asked about, articles asked about) of evaluation sets.

    Reads any JSONL whose rows name ``code`` and ``articolo``, at the top or
    in ``metadata`` (the closed-book suite, the exams, the outside test). A
    row of kind ``esiste_falsa`` names a number that does not exist.
    """
    fakes: set[tuple[str, str]] = set()
    asked: set[tuple[str, str]] = set()
    for p in paths:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                meta = {**row, **(row.get("metadata") or {})}
                code, number = meta.get("code"), meta.get("articolo")
                if not code or not number:
                    continue
                key = (str(code), str(number).lower())
                (fakes if meta.get("tipo") == "esiste_falsa" else asked).add(key)
    return fakes, asked


def program_rows(articles: dict[tuple[str, str], Article], *,
                 fakes: set[tuple[str, str]] = frozenset(), seed: int = 0,
                 absent_share: float = 0.3) -> list[dict]:
    """Every program-written form of every article: statements, never questions."""
    rng = random.Random(seed)
    rows: list[dict] = []

    def add(form, code, number, text):
        rows.append({"id": f"{form}|{code}|{number}", "code": code, "articolo": number,
                     "forma": form, "source": "program", "text": text})

    for code, arts in by_code(articles).items():
        of, in_ = CODE_LABELS[code]
        name = code_name(code)
        In = in_[:1].upper() + in_[1:]
        taken = {a.number for a in arts}
        for a in arts:
            h = a.heading.strip(" .")
            repealed = _repealed_sentence(a)
            body = _body(a)
            if body:
                add("testo", code, a.number, f"{name}, art. {a.number}"
                    + (f" ({h})" if h else "") + f".\n{body}")
            if repealed:
                add("abrogato", code, a.number,
                    f"L'art. {a.number} {of}" + (f" ({h})" if h else "")
                    + f" non è più in vigore. {_plain(repealed)}")
                continue
            if _concept(h):
                i = rng.randrange(len(RUBRICA_FWD))
                add("rubrica", code, a.number,
                    RUBRICA_FWD[i].format(n=a.number, of=of, h=h) + " "
                    + RUBRICA_BACK[(i + 1) % len(RUBRICA_BACK)].format(
                        n=a.number, of=of, h=h, In=In))
            # deadlines ("entro trenta giorni"), not the length of a penalty
            timed = [s for _, s in split_sentences(_plain(a.text)) if _deadlines(s)]
            if timed:
                terms = sorted({(u, n) for s in timed for n, u in _deadlines(s)})
                add("termine", code, a.number,
                    f"I termini dell'art. {a.number} {of}"
                    + (f" ({h})" if h else "") + ": "
                    + "; ".join(say(n, u) for u, n in terms)
                    + ". " + " ".join(_plain(s) for s in timed))
            pena = penalty_of(a.text)
            if pena:
                kind, lo, lo_unit, hi, unit, sentence, comma = pena
                add("pena", code, a.number,
                    f"Il reato dell'art. {a.number} {of}" + (f" ({h})" if h else "")
                    + f"{comma} è punito con la {kind} da {say(lo, lo_unit)} "
                    f"a {say(hi, unit)}. {_plain(sentence)}")
            proc = procedibility_of(a.text)
            if proc:
                regime, sentence = proc
                how = "a querela della persona offesa" if regime == "querela" else "d'ufficio"
                add("procedibilita", code, a.number,
                    f"Il reato previsto dall'art. {a.number} {of}" + (f" ({h})" if h else "")
                    + f" è procedibile {how}. {_plain(sentence)}")

        for start in range(0, len(arts), INDEX_SPAN):
            block = arts[start:start + INDEX_SPAN]
            lines = [f"art. {a.number}"
                     + (f" — {a.heading.strip(' .')}" if a.heading.strip(" .") else "")
                     + (" (abrogato)" if _repealed_sentence(a) else "") for a in block]
            add("indice", code, f"{block[0].number}..{block[-1].number}",
                f"{name}, articoli da {block[0].number} a {block[-1].number}:\n"
                + "\n".join(lines))

        position = {a.number: i for i, a in enumerate(arts)}
        bases = {a.number.split("-")[0] for a in arts}
        for base in sorted(bases, key=lambda b: number_key(b)):
            if base not in position:
                continue
            suffixed = any(n.startswith(base + "-") for n in taken)
            if not suffixed and rng.random() >= absent_share:
                continue
            fake = unused_number(base, taken)
            if not fake or (code, fake) in fakes:
                continue
            last = max((a for a in arts if a.number.split("-")[0] == base),
                       key=lambda a: number_key(a.number))
            nxt = position[last.number] + 1
            after = arts[nxt] if nxt < len(arts) else None

            def named(a):
                h = a.heading.strip(" .")
                return f"l'art. {a.number}" + (f" ({h})" if h else "")

            text = (f"{In} non esiste un art. {fake}: dopo {named(last)} "
                    + (f"viene {named(after)}." if after else "il codice termina."))
            add("inesistente", code, fake, text[:1].upper() + text[1:])
        last = arts[-1]
        add("inesistente", code, f"{last.number}+",
            f"{In} l'ultimo articolo è il {last.number}: non esistono articoli con un numero "
            f"più alto.")
    return rows


SYSTEM = (
    "Sei un docente di diritto italiano. Riscrivi gli articoli di legge che ti vengono dati "
    "senza aggiungere nulla che il testo non dica: niente altri articoli o leggi, niente "
    "numeri, termini, pene o importi che non vi compaiano, niente giurisprudenza. Scrivi in "
    "italiano giuridico corretto e rispondi solo con il testo richiesto."
)

TASKS = {
    "parafrasi": "Riscrivi con parole tue, in un paragrafo, la regola che l'articolo stabilisce, "
                 "conservando ogni condizione, termine, pena ed eccezione. Comincia nominando "
                 "l'articolo («L'art. {n} {of} ...»).",
    "spiegazione": "Spiega a uno studente di giurisprudenza, in due o tre paragrafi, che cosa "
                   "stabilisce l'articolo, a che cosa serve e quali sono i suoi elementi. "
                   "Nomina l'art. {n} {of} nella prima frase. Non collocare l'articolo in un "
                   "istituto, un contratto o un tipo di società che il testo non nomina.",
    "caso": "Scrivi un breve esempio pratico, di cinque-otto frasi, in cui si applica "
            "l'articolo, con le persone indicate solo come Tizio, Caio e Sempronio e senza "
            "date, luoghi o importi; poi spiega come l'art. {n} {of} lo risolve.",
    "domande": "Scrivi da tre a cinque coppie di domanda e risposta sul contenuto "
               "dell'articolo, ciascuna su due righe che cominciano con «D:» e «R:». Ogni "
               "risposta nomina l'art. {n} {of}.",
    "elementi": "Scrivi una frase che nomina l'articolo («L'art. {n} {of} ...») e poi elenca, "
                "uno per riga e preceduti da un trattino, i presupposti, gli effetti e le "
                "eccezioni che l'articolo stabilisce.",
}


def rewritable(article: Article) -> bool:
    """In force, with a body, and short enough to be rewritten whole."""
    body = _body(article)
    return (bool(body) and 80 <= len(body) <= MAX_ARTICLE_CHARS
            and not _repealed_sentence(article))


def rewrite_messages(article: Article, style: str) -> list[dict]:
    """The teacher request for one rewrite of one article."""
    of = CODE_LABELS[article.code][0]
    h = article.heading.strip(" .")
    head = f"Art. {article.number} {of}" + (f" ({h})" if h else "")
    task = TASKS[style].format(n=article.number, of=of)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{head}\n--- TESTO ---\n{_body(article)}\n--- FINE ---\n\n"
                                        f"{task}"}]


_DIGITS = re.compile(r"\d+(?:[.,]\d+)*")
# deadlines and penalties a rewrite of these styles must keep: the ones
# meant to restate the whole rule
_COMPLETE = ("parafrasi", "elementi")


def clean_reply(raw: str) -> str:
    text = re.sub(r"(?s)<think>.*?</think>", "", raw or "").strip()
    return re.sub(r"^```\w*\s*|\s*```$", "", text).strip()


def check_rewrite(article: Article, style: str, text: str) -> str | None:
    """Why a rewrite is refused, or None when it says only what the article says."""
    if not 120 <= len(text) <= 5000:
        return "length"
    own = article.number.lower()
    cited = cited_articles(text)
    if own not in cited:
        return "no_article"
    if cited - cited_articles(article.text) - {own}:
        return "other_article"
    source = _plain(article.text)
    allowed = {d.replace(".", "").replace(",", "") for d in _DIGITS.findall(source)}
    allowed |= set(re.findall(r"\d+", own))
    for d in _DIGITS.findall(text):
        d = d.replace(".", "").replace(",", "")
        if int(d) > 10 and d not in allowed:
            return "number"
    stated = all_deadlines(source) | mentioned_deadlines(source)
    said = mentioned_deadlines(text)
    if said - stated:
        return "deadline"
    if style in _COMPLETE and all_deadlines(source) - said:
        return "lost_deadline"
    if "abrogat" in text.lower() and "abrogat" not in source.lower():
        return "repeal"
    return None
