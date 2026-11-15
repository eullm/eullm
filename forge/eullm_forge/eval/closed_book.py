"""The closed-book suite: what a model knows of the law with no text in front of it.

Every exam since September hands the model its texts; this one asks the bare
question, as a plain chat does, to measure the knowledge the legal-memory
plan (``docs/research/legal-memory-plan.md``) sets out to put in. Built by
program from the legislation, so every answer key is a fact read off the
article, and scored by program (`score`), so a judge's noise is not in it:

* ``esiste_vera`` / ``esiste_falsa``: does art. N exist, and what is it
  about. Half real, half made up (an unused "-bis", or a number past the
  code's last article); a made-up one is right only when denied.
* ``numero_da_rubrica`` / ``rubrica_da_numero``: heading to number and back.
* ``termine``, ``pena``, ``procedibilita``: the number or the regime an
  article states (the extractors of the premise generator).
* ``abrogato``: an article repealed, asked as if in force.
* ``sentenza_falsa``: a ruling cited by a number made up; right only when
  the model says it cannot confirm it.

A wrong answer given without any doubt is counted apart ("confident"):
a model that says "I am not sure" is worth more than one that invents.

The items name their article (``metadata``), so a training corpus can be
split against them; the question templates here are the suite's own and
must not be used to write training data.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from ..datasets.premise_gen import (
    _repealed_sentence,
    defines_offence,
    penalty_of,
    procedibility_of,
)
from ..rl.rewards import abstains, denies_existence, mentioned_deadlines, score_answer
from .dataset import EvalItem
from .norm_exam import (
    CODE_LABELS,
    _deadline_keyword,
    _deadlines,
    _flat,
    all_deadlines,
    articles_from_records,
)

KINDS = ("esiste_vera", "esiste_falsa", "numero_da_rubrica", "rubrica_da_numero", "termine",
         "pena", "procedibilita", "abrogato", "sentenza_falsa")

Q_ESISTE = "Esiste l'art. {n} {of}? Se esiste, di che cosa tratta?"
Q_NUMERO = "Quale articolo {in_} disciplina «{h}»? Indica il numero."
Q_RUBRICA = "Di che cosa tratta l'art. {n} {of}? Rispondi in una frase."
Q_TERMINE = "Quale termine prevede l'art. {n} {of}?"
Q_PENA = "Quale pena prevede l'art. {n} {of}{comma}?"
Q_PROC = "Il reato previsto dall'art. {n} {of} è procedibile d'ufficio o a querela?"
Q_ABROGATO = "Che cosa prevede oggi l'art. {n} {of}?"
Q_SENTENZA = "Che cosa ha stabilito la Corte di cassazione ({sez}) con la sentenza n. {num}/{year}?"

_GENERIC = re.compile(
    r"^(?:disposizion|norm[ae]\b|ambito|definizion|abrogat|oggetto|finalit|entrata in vigore|"
    r"modific|articol|casi\b|altr[ie]\b|rinvio|applicazion|clausola|regime transitorio)",
    re.IGNORECASE)
_STOP = {"della", "delle", "degli", "dello", "dalla", "dalle", "dagli", "nella", "nelle",
         "negli", "sulla", "sulle", "sugli", "alla", "alle", "agli", "questo", "questa",
         "essere", "sono", "come", "anche", "ogni"}


def _words(text: str) -> set[str]:
    """Content words, cut to five letters so that number and gender do not matter."""
    return {w[:5] for w in re.findall(r"[a-zàèéìòù]{4,}", text.lower()) if w not in _STOP}


def heading_overlap(heading: str, answer: str) -> float:
    """Share of the heading's content words the answer says."""
    h = _words(heading)
    return len(h & _words(answer)) / len(h) if h else 0.0


_ART = re.compile(r"\bart(?:icol[oi]|t)?\.?\s*(\d+(?:\s*-\s*[a-z]+)?)", re.IGNORECASE)


def cited_articles(answer: str) -> set[str]:
    """The article numbers an answer names ("art. 99", "articolo 624-bis")."""
    return {re.sub(r"\s+", "", m).lower() for m in _ART.findall(answer)}


# Saying one does not know, in the words models use; no content is claimed.
_UNSURE = re.compile(
    r"non (?:so|conosco|ricordo|sono (?:sicur|in grado)|ho (?:informazioni|dati|elementi)"
    r"|dispongo|posso (?:confermar|verificar|dire|indicar|rispondere))"
    r"|non mi risulta|non (?:è|e') possibile (?:verificar|confermar)", re.IGNORECASE)


def unsure(answer: str) -> bool:
    """Whether the answer says it does not know or cannot confirm."""
    return bool(_UNSURE.search(answer)) or abstains(answer)


def _concept(heading: str) -> str | None:
    h = heading.strip(" .;:()")
    if not 4 <= len(h) <= 70 or _GENERIC.match(h) or not h[:1].isalpha():
        return None
    return h


def build_suite(records: list[dict], per_kind: int = 15, n_rulings: int = 40,
                seed: int = 0) -> list[EvalItem]:
    """Draw the suite: up to ``per_kind`` items of each kind for every code."""
    rng = random.Random(seed)
    arts = articles_from_records(records)
    by_code: dict[str, list] = defaultdict(list)
    for a in arts.values():
        if a.code in CODE_LABELS:
            by_code[a.code].append(a)
    items: list[EvalItem] = []

    def add(kind, code, number, question, reference, keywords=(), **meta):
        items.append(EvalItem(
            id=f"cb-{kind}-{code}-{number}", domain="legal", lang="it", question=question,
            reference=reference, keywords=list(keywords), category=code,
            metadata={"tipo": kind, "code": code, "articolo": number, **meta}))

    for code in sorted(by_code):
        of, in_ = CODE_LABELS[code]
        pool = sorted(by_code[code], key=lambda a: a.number)
        live = [a for a in pool if not _repealed_sentence(a)]
        taken = {a.number for a in pool}

        def draw(cands, _n=per_kind):
            return rng.sample(cands, min(_n, len(cands)))

        named = [a for a in live if _concept(a.heading)]
        for a in draw(named):
            add("esiste_vera", code, a.number, Q_ESISTE.format(n=a.number, of=of),
                f"Sì: l'art. {a.number} {of} riguarda «{a.heading}».", [a.heading])
        last = max(int(re.match(r"\d+", n).group()) for n in taken if re.match(r"\d+", n))
        plain = [a.number for a in live if a.number.isdigit() and f"{a.number}-bis" not in taken]
        fakes: set[str] = set()
        for _ in range(per_kind * 3):
            if len(fakes) >= per_kind:
                break
            fake = (f"{rng.choice(plain)}-bis" if plain and rng.random() < 0.6
                    else str(last + rng.randint(30, 900)))
            if fake not in taken:
                fakes.add(fake)
        for fake in sorted(fakes):
            add("esiste_falsa", code, fake, Q_ESISTE.format(n=fake, of=of),
                f"No: l'art. {fake} {of} non esiste.")
        counts = defaultdict(int)
        for a in named:
            counts[a.heading.lower()] += 1
        unique = [a for a in named if counts[a.heading.lower()] == 1]
        for a in draw(unique):
            add("numero_da_rubrica", code, a.number,
                Q_NUMERO.format(in_=in_, h=_concept(a.heading)),
                f"L'art. {a.number} {of}.", [a.number])
        for a in draw(unique):
            add("rubrica_da_numero", code, a.number, Q_RUBRICA.format(n=a.number, of=of),
                f"L'art. {a.number} {of} riguarda «{a.heading}».", [a.heading])
        timed = [(a, next(iter(d))) for a in live
                 if len(d := _deadlines(a.text)) == 1 and all_deadlines(a.text) == d]
        for a, (n, unit) in draw(timed):
            add("termine", code, a.number, Q_TERMINE.format(n=a.number, of=of),
                _flat(a.text), [_deadline_keyword(n, unit)])
        punished = [(a, p) for a in live if defines_offence(a.text) and (p := penalty_of(a.text))]
        for a, (kind_, lo, lo_unit, hi, unit, sentence, comma) in draw(punished):
            add("pena", code, a.number, Q_PENA.format(n=a.number, of=of, comma=comma),
                sentence, [f"{hi} {unit}"], high=[hi, unit], low=[lo, lo_unit])
        regimes = [(a, r) for a in live
                   if defines_offence(a.text) and (r := procedibility_of(a.text))]
        for a, (regime, sentence) in draw(regimes):
            add("procedibilita", code, a.number, Q_PROC.format(n=a.number, of=of),
                sentence, [regime])
        repealed = [(a, s) for a in pool if (s := _repealed_sentence(a))]
        for a, sentence in draw(repealed):
            add("abrogato", code, a.number, Q_ABROGATO.format(n=a.number, of=of), sentence,
                ["abrogat"])

    sections = ("Sez. I civile", "Sez. II civile", "Sez. III civile", "Sez. lavoro",
                "Sez. I penale", "Sez. II penale", "Sez. V penale", "Sez. VI penale")
    for _ in range(n_rulings):
        num, year = rng.randint(10000, 59999), rng.randint(2019, 2025)
        add("sentenza_falsa", "cassazione", f"{num}-{year}",
            Q_SENTENZA.format(sez=rng.choice(sections), num=num, year=year),
            "Non è possibile confermare il contenuto di questa sentenza.")
    return items


def score(item: dict, answer: str) -> tuple[str, bool]:
    """("right" | "partial" | "wrong", confident) of one answer.

    ``confident`` marks a wrong answer given without any doubt expressed.
    """
    kind = item["metadata"]["tipo"]
    kw = item.get("keywords") or []
    doubt = unsure(answer)
    if kind == "esiste_falsa":
        if denies_existence(answer):
            return "right", False
        return ("partial", False) if doubt else ("wrong", True)
    if kind == "sentenza_falsa":
        return ("right", False) if doubt else ("wrong", True)
    if kind in ("esiste_vera", "rubrica_da_numero"):
        if denies_existence(answer):
            return "wrong", True
        if heading_overlap(kw[0], answer) >= 0.5:
            return "right", False
        return ("partial", False) if doubt else ("wrong", True)
    if kind == "numero_da_rubrica":
        cited = cited_articles(answer)
        if kw[0].lower() in cited and len(cited) <= 3:
            return "right", False
        return ("partial", False) if doubt else ("wrong", True)
    if kind == "termine":
        if score_answer(answer, "termine", kw) == 1.0:
            return "right", False
        return ("partial", False) if doubt else ("wrong", True)
    if kind == "pena":
        hi = tuple(item["metadata"]["high"])
        named = mentioned_deadlines(answer)
        if hi in named and len(named) <= 4:
            return "right", False
        return ("partial", False) if doubt else ("wrong", True)
    if kind == "procedibilita":
        low = answer.lower().replace("’", "'")
        q, u = "querela" in low, "d'ufficio" in low
        right = kw[0]
        says, other = (q, u) if right == "querela" else (u, q)
        if says and not other:
            return "right", False
        if says and other:
            return "partial", False
        return ("partial", False) if doubt else ("wrong", True)
    if kind == "abrogato":
        if "abrogat" in answer.lower():
            return "right", False
        return ("partial", False) if doubt else ("wrong", True)
    raise ValueError(f"no closed-book check for {kind!r}")
