"""Rewards for the prompts with no texts (`eullm_forge.datasets.closed_prompts`).

Written against the ways a policy games them, as `rewards` is:

* **Hedging, then answering.** "Non posso verificarla, ma ha stabilito che..."
  is not a refusal: a ruling answer that states a holding earns nothing.
* **Denying everything.** Real and made-up articles come in equal numbers,
  and a real one denied earns nothing.
* **Naming many articles.** A number answer is right only with at most three
  numbers in it: "artt. 640, 641, 642 e 643" names four, though
  `cited_articles` reads only the first after "artt.".
"""

from __future__ import annotations

import re

from ..eval.closed_book import cited_articles, heading_overlap, unsure
from .rewards import denies_existence, unverified_ruling

LIBERO_TYPES = frozenset({"libero_sentenza", "libero_esiste_falsa", "libero_esiste_vera",
                          "libero_numero"})

# A number standing alone (an article's), not part of a date or a ruling's n./year.
_NUMBER = re.compile(r"(?<![\d/.,])\b\d{1,4}(?:\s*-\s*[a-z]+)?\b(?![\d/.,]\d)", re.IGNORECASE)
# What a ruling "held": an answer saying it, however hedged, claims to know.
_HOLDING = re.compile(
    r"\bha (?:stabilito|affermato|statuito|ribadito|chiarito|ritenuto|deciso|precisato|"
    r"enunciato|escluso|riconosciuto)\b|\b(?:la|tale|questa|detta) (?:sentenza|pronuncia) "
    r"(?:afferma|stabilisce|chiarisce|ritiene|precisa)\b|\bil principio (?:affermato|"
    r"enunciato|di diritto)\b", re.IGNORECASE)


def score_libero(answer: str, tipo: str, keywords: list[str] | None = None) -> float:
    """1.0 for a right answer to a prompt with no texts, 0.0 otherwise."""
    kw = keywords or []
    if tipo == "libero_sentenza":
        doubts = unverified_ruling(answer) or unsure(answer)
        return 1.0 if doubts and not _HOLDING.search(answer) else 0.0
    if tipo == "libero_esiste_falsa":
        return 1.0 if denies_existence(answer) else 0.0
    if tipo == "libero_esiste_vera":
        if not kw or denies_existence(answer):
            return 0.0
        return 1.0 if heading_overlap(kw[0], answer) >= 0.5 else 0.0
    if tipo == "libero_numero":
        cited = cited_articles(answer)
        named = {re.sub(r"\s+", "", m).lower() for m in _NUMBER.findall(answer)}
        return 1.0 if kw and kw[0].lower() in cited and len(named) <= 3 else 0.0
    raise ValueError(f"no closed-book reward for {tipo!r}")
