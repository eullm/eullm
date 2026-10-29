"""The principles of law of the ruling cards, rewritten for the memory corpus.

A card (`schede`) holds one to four principles of a ruling, stated in the
abstract. For the model to hold them each must be seen in more than one
form, as the articles are (`datasets.law_corpus`): the teacher writes every
principle again in other words, and as a lawyer's question with its answer.

Nothing is added: a rewrite is refused (`check`) if it names an article the
principle and the card's norms do not, a number over ten they do not have,
a pseudonym placeholder, a tax code or an IBAN. The ruling's number is not
in the request and not in the rewrite: whether real numbers are attached is
decided apart, by program, when the corpus is put together.
"""

from __future__ import annotations

import json
import re

from ..eval.closed_book import cited_articles
from .schede import _leaks

# Raised whenever the request or the checks change; written in every row ("pv").
PROMPT_VERSION = 1

SYSTEM = (
    "Sei un magistrato dell'Ufficio del massimario. Riformuli principi di diritto già "
    "estratti da sentenze, senza aggiungere nulla: niente norme, numeri, nomi, luoghi o "
    "fatti che non compaiano nel principio. Scrivi in italiano giuridico corretto. Rispondi "
    "solo con un oggetto JSON."
)

TASK = (
    "Principi di diritto affermati {of}{materia}:\n{principles}\n{norms}\n"
    "Per ciascun principio, nello stesso ordine, restituisci un oggetto JSON "
    "{{\"principi\": [{{\"riformulazione\": \"...\", \"domanda\": \"...\", "
    "\"risposta\": \"...\"}}]}}: "
    "la riformulazione dice lo stesso principio con parole diverse, senza toglierne "
    "condizioni o eccezioni; la domanda è generale, come la porrebbe un avvocato, e la "
    "risposta (una-tre frasi) espone il principio."
)

COURT_OF = {"cds": "dal Consiglio di Stato", "snciv": "dalla Corte di cassazione civile",
            "snpen": "dalla Corte di cassazione penale"}


def messages(card: dict) -> list[dict]:
    """The request for one card's principles."""
    principles = "\n".join(f"{i}. {p}" for i, p in enumerate(card["principi"], 1))
    norms = ("Norme applicate: " + "; ".join(card["norme"]) + "\n") if card.get("norme") else ""
    materia = f" (materia: {card['materia']})" if card.get("materia") else ""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASK.format(
                of=COURT_OF.get(card.get("court", ""), "in una sentenza"), materia=materia,
                principles=principles, norms=norms)}]


_DIGITS = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    return {d.replace(".", "").replace(",", "") for d in _DIGITS.findall(text)}


def check(source: str, text: str) -> str | None:
    """Why a rewrite of a principle is refused, or None when it adds nothing."""
    if not 40 <= len(text) <= 1500:
        return "length"
    leak = _leaks(text)
    if leak:
        return leak
    if cited_articles(text) - cited_articles(source):
        return "other_article"
    allowed = _numbers(source)
    if any(int(d) > 10 and d not in allowed for d in _numbers(text) if d.isdigit()):
        return "number"
    return None


def parse(card: dict, raw: str) -> tuple[list[tuple[int, str, str]], list[tuple[int, str]]]:
    """(kept, refused) of one reply: kept rows are (principle, form, text),
    refused ones (principle, reason); principle -1 refuses the whole reply."""
    text = re.sub(r"(?s)<think>.*?</think>", "", raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        items = json.loads(text[start:end + 1])["principi"] if 0 <= start < end else None
    except (json.JSONDecodeError, KeyError, TypeError):
        items = None
    if not isinstance(items, list):
        return [], [(-1, "bad_json")]
    if len(items) != len(card["principi"]):
        return [], [(-1, "count")]
    norms = " ".join(card.get("norme") or [])
    kept, refused = [], []
    for i, (item, principle) in enumerate(zip(items, card["principi"])):
        if not isinstance(item, dict):
            refused.append((i, "bad_field"))
            continue
        source = f"{principle} {norms}"
        forms = (("riformulazione", str(item.get("riformulazione") or "").strip()),
                 ("domanda", "D: " + str(item.get("domanda") or "").strip()
                  + "\nR: " + str(item.get("risposta") or "").strip()))
        for form, body in forms:
            reason = check(source, body)
            if reason:
                refused.append((i, f"{form}:{reason}"))
            else:
                kept.append((i, form, body))
    return kept, refused
