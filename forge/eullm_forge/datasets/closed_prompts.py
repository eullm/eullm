"""GRPO prompts with no texts, graded by program: what a model says when nothing is in front of it.

E1 put the articles into the backbone and GRPO gave back the open-book
behaviour, but on the closed book the made-up rulings went from 90% refused
(g9r-r2) to 0-8% after every GRPO round (10-11-2026): GRPO's prompts all
carry their texts, so nothing in them asks what a ruling held with nothing to
check it against. These prompts do, as a plain chat asks, with four kinds the
rewards check (`eullm_forge.rl.closed_rewards`):

* ``libero_sentenza``: what a made-up ruling held; right only when the
  answer says it cannot verify it and states no holding;
* ``libero_esiste_falsa``: an article the code does not have; right only
  when it is denied;
* ``libero_esiste_vera``: a real article; right when its heading is named and
  it is not denied, so that refusing everything earns nothing;
* ``libero_numero``: the article of a heading, the direction E1 did not
  learn (0-8% on the closed book against 50% the other way).

The question templates are this module's own: the closed-book suite's must
not be used to write training data (`eval.closed_book`). No article of an
evaluation set is drawn (``excluded``), nor a made-up number the suite asks
about.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from ..eval.closed_book import _concept
from ..eval.norm_exam import CODE_LABELS
from .premise_gen import _repealed_sentence

Q_SENTENZA = [
    "Un collega mi ha citato la sentenza della Cassazione n. {num} del {year} ({sez}). "
    "Che principio afferma?",
    "Mi riassumi la pronuncia Cass. {sez} {num}/{year}?",
    "Devo citare in un atto Cass., {sez}, n. {num}/{year}: cosa ha deciso la Corte?",
]
Q_ARTICOLO = [
    "Cosa dispone l'art. {n} {of}?",
    "Riassumimi in due righe il contenuto dell'art. {n} {of}.",
    "A cosa si riferisce l'art. {n} {of}?",
]
Q_NUMERO = [
    "Sai indicarmi il numero dell'articolo {in_} su «{h}»?",
    "Per citare «{h}», {in_}, quale articolo devo indicare?",
]
SECTIONS = ("Sez. I civile", "Sez. II civile", "Sez. III civile", "Sez. lavoro",
            "Sez. I penale", "Sez. II penale", "Sez. V penale", "Sez. VI penale")


def _row(kind, code, number, question, keywords=()):
    return {"id": f"cl-{kind}-{code}-{number}", "prompt": [{"role": "user", "content": question}],
            "tipo": kind, "keywords": list(keywords), "code": code, "articolo": number}


def closed_prompts(articles: dict, *, per_code: int, rulings: int, rng: random.Random,
                   excluded: set[tuple[str, str]] = frozenset(),
                   fakes: set[tuple[str, str]] = frozenset()) -> list[dict]:
    """``per_code`` prompts of each article kind for every code, and ``rulings`` made up.

    ``excluded``: (code, article) an evaluation set asks about, never drawn;
    ``fakes``: (code, number) the suite asks about as made up, never used.
    """
    by_code: dict[str, list] = defaultdict(list)
    for a in articles.values():
        if a.code in CODE_LABELS:
            by_code[a.code].append(a)
    rows: list[dict] = []
    for code in sorted(by_code):
        of, in_ = CODE_LABELS[code]
        pool = sorted(by_code[code], key=lambda a: a.number)
        taken = {a.number for a in pool}
        live = [a for a in pool if not _repealed_sentence(a)
                and (code, a.number.lower()) not in excluded]
        named = [a for a in live if _concept(a.heading)]
        for a in rng.sample(named, min(per_code, len(named))):
            rows.append(_row("libero_esiste_vera", code, a.number,
                             rng.choice(Q_ARTICOLO).format(n=a.number, of=of), [a.heading]))
        counts = defaultdict(int)
        for a in named:
            counts[a.heading.lower()] += 1
        unique = [a for a in named if counts[a.heading.lower()] == 1]
        for a in rng.sample(unique, min(per_code, len(unique))):
            rows.append(_row("libero_numero", code, a.number,
                             rng.choice(Q_NUMERO).format(in_=in_, h=_concept(a.heading)),
                             [a.number]))
        plain = [a.number for a in live if a.number.isdigit() and f"{a.number}-bis" not in taken]
        last = max((int(m.group()) for n in taken if (m := re.match(r"\d+", n))), default=0)
        made: set[str] = set()
        for _ in range(per_code * 4):
            if len(made) >= per_code or not plain:
                break
            fake = (f"{rng.choice(plain)}-bis" if rng.random() < 0.6
                    else str(last + rng.randint(30, 900)))
            if fake not in taken and (code, fake) not in fakes:
                made.add(fake)
        for fake in sorted(made):
            rows.append(_row("libero_esiste_falsa", code, fake,
                             rng.choice(Q_ARTICOLO).format(n=fake, of=of)))
    for i in range(rulings):
        num, year = rng.randint(10000, 59999), rng.randint(2016, 2025)
        rows.append(_row("libero_sentenza", "", "",
                         rng.choice(Q_SENTENZA).format(sez=rng.choice(SECTIONS), num=num,
                                                       year=year))
                    | {"id": f"cl-libero_sentenza-{i}"})
    rng.shuffle(rows)
    return rows
