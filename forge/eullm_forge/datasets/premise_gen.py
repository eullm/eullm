"""Questions built on a false premise, and answers that correct it first.

The LexIA test of 2026-10-18 (a lawyer's 36 trap questions, a report
received on reddit.com) found what our exams had not: legal-it-8b corrected
6 of 18 false premises, and no teacher we measured did much better -- the
best, Qwen3.6-27B, corrected 6 too. Distilling a teacher cannot teach what
the teacher does not do. These need no teacher: the premise is made false by
construction, from the statute itself, so the right answer is known and a
program can check it.

Three kinds, each in the exact prompt the exam and the engine use
(`open_book_prompt`, same retrieval), and with no note saying an article is
missing, as RAG Enterprise asks:

* ``premessa_termine`` -- the question states a deadline the article does
  not: "L'art. 2 della legge n. 241/1990 prevede un termine di 60 giorni:
  ...". The article is among the texts; the answer says the premise is
  wrong, quotes the sentence that states the real deadline and gives it.
* ``premessa_inesistente`` -- the question takes for granted an article that
  is not in the collection: "-bis" of an article that has none, or a number
  past the last. The answer says it is not there and attributes it nothing.
* ``premessa_abrogato`` -- the question treats a repealed article as in
  force. Its text, among the others, says it was repealed; the answer says
  so, quoting it.
* ``premessa_pena`` -- the question states a maximum penalty the criminal
  code's article does not: "l'art. N c.p. punisce con la reclusione fino a
  cinque anni?" where it says three. The LexIA traps are mostly of this kind
  and of the next; the deadlines above were already learnt (0.94-0.99 on
  2026-10-21) and moved the traps by one.
* ``premessa_procedibilita`` -- the question takes an offence prosecuted on
  complaint ("a querela della persona offesa") for one prosecuted ex officio,
  or the other way round; only articles that state one of the two, and only
  one, are used.
* ``premessa_vera`` -- the same deadline question with the RIGHT deadline in
  it. The answer confirms it, quoting the text. Without these a model taught
  only false premises learns to correct everything: on 2026-10-21 one that
  had been through the false premises answered 1 of the 9 LexIA questions
  whose premise was sound. Asked of the same article as its false twin, it
  is the article that decides, not the shape of the question.

The answer reasons in fixed steps before it answers (``fmt="steps"``):
"Verifica della premessa", "Norma applicabile", "Risposta". Each step can be
checked against the texts shown -- the passage number ``[n]``, the quoted
words, the deadline -- which a free reasoning cannot: on 2026-10-19 thinking
cost Qwen3.6-27B 6 of its 16 right trap answers. ``fmt="plain"`` is the same
content in one paragraph, for the ablation that decides between the two.

Every quotation is a literal span of the text the prompt shows, checked when
the pair is built: a pair whose quote is not there is not written.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..eval.metrics import DEADLINE_UNITS, number_of
from ..eval.norm_exam import (
    CODE_LABELS,
    _deadline_keyword,
    _deadlines,
    _sentence_with,
    _sentences,
    all_deadlines,
    articles_from_records,
    split_sentences,
)
from ..eval.retrieval import named_articles, named_code, open_book_prompt

KINDS = ("premessa_termine", "premessa_inesistente", "premessa_abrogato", "premessa_pena",
         "premessa_procedibilita", "premessa_vera", "premessa_sentenza", "premessa_sentenza_vera",
         "premessa_numero")
FORMATS = ("steps", "plain")

# Deadlines a person might wrongly believe, by unit: the usual ones of
# Italian procedure. The wrong one is drawn from these, never from the
# deadlines the article itself states.
USUAL = {"giorni": [3, 5, 8, 10, 15, 20, 30, 40, 45, 60, 90, 120, 180],
         "mesi": [1, 2, 3, 4, 6, 12, 18, 24, 36],
         "anni": [1, 2, 3, 5, 10, 20],
         "ore": [12, 24, 48, 72]}
_SINGULAR = {"giorni": "giorno", "mesi": "mese", "anni": "anno", "ore": "ora"}

# A quotation longer than this is cut to the words around what it is quoted for.
MAX_QUOTE_CHARS = 300

TERMINE_NAMED = [
    "L'art. {n} {of} prevede un termine di {wrong}: entro quando devo quindi agire?",
    "Mi confermi che il termine previsto dall'art. {n} {of} è di {wrong}?",
    "Il mio avvocato dice che l'art. {n} {of} dà {wrong} di tempo. È così?",
]
# A deadline attributed to a ruling. The ruling is made up, so it is never
# among the texts: the answer cannot confirm what it held, says so, and
# checks the claim against the statute it does have. The outside traps the
# numbered premises did not move include principles attributed to rulings
# that do not state them (2026-10-22).
SENTENZA_Q = [
    "La Cassazione ({sez}), con la sentenza n. {num}/{year}, ha affermato che il termine previsto "
    "dall'art. {n} {of} è di {claim}. Mi confermi che è così?",
    "Secondo Cass. {sez} n. {num}/{year}, per l'art. {n} {of} il termine è di {claim}: posso fare "
    "affidamento su questa sentenza?",
]
TERMINE_TOPIC = [
    "{In}, in materia di «{heading}», il termine è di {wrong}, giusto?",
    "So che {in_}, in materia di «{heading}», ci sono {wrong} di tempo: me lo confermi?",
]
INESISTENTE = [
    "Quali obblighi impone l'art. {n} {of}?",
    "Ho letto che l'art. {n} {of} è stato introdotto di recente: che cosa prevede?",
    "Secondo l'art. {n} {of}, chi è competente a decidere?",
    "Come si applica in concreto l'art. {n} {of}?",
]
PENA_Q = [
    "Mi confermi che l'art. {n} {of}{comma} punisce il fatto con la {pena} da {lo} a {hi}?",
    "L'art. {n} {of}{comma} prevede la {pena} fino a {hi}: è questo il massimo che si rischia?",
]
# The same, naming the offence by its heading instead of the article: people
# ask about "il furto", not "l'art. 624", and so do the outside tests the
# numbered questions did not move (2026-10-22).
PENA_NAMED_Q = [
    "Mi confermi che per il reato di {name}{base} la pena è la {pena} da {lo} a {hi}?",
    "Per il reato di {name}{base} è prevista la {pena} fino a {hi}: è questo il massimo che "
    "si rischia?",
]
# An offence named right with an article number that does not exist ("la frode
# informatica dell'art. 640-septies c.p."): the answer says so, names the
# article that provides for the offence, and answers from it. Until
# 2026-10-27 every nonexistent article was answered "non posso dirti che cosa
# prevede ... se mi indichi l'argomento, posso cercare", and the 8B, asked the
# outside test's 640-septies, corrected the number and stopped there where the
# answer was one article away (640-ter) among its own texts.
NUMERO_Q = [
    "Quale pena prevede l'art. {fake} {of} per il reato di {name}?",
    "Per il reato di {name}, previsto dall'art. {fake} {of}, qual è la pena massima?",
    "Il reato di {name} dell'art. {fake} {of}: quale pena rischia chi lo commette?",
]
SUFFIXES = ("bis", "ter", "quater", "quinquies", "sexies", "septies", "octies", "novies",
            "decies")


def unused_number(number: str, taken: set[str]) -> str | None:
    """The next suffix of ``number``'s base that the code does not use ("640-septies")."""
    base = number.split("-")[0]
    used = [i for i, suf in enumerate(SUFFIXES) if f"{base}-{suf}" in taken]
    nxt = (max(used) + 1) if used else 0
    return f"{base}-{SUFFIXES[nxt]}" if nxt < len(SUFFIXES) else None


# The procedibility as the article states it, and the question that gets it
# wrong (false) or right (true). {reato} is "il reato previsto dall'art. ..."
# or "il reato di <heading>".
PROC = {
    "querela": ("a querela della persona offesa",
                ["{Reato} è procedibile d'ufficio, giusto?",
                 "Per {reato} non serve la querela della persona offesa, vero?"],
                ["{Reato} è procedibile a querela della persona offesa?"]),
    "d'ufficio": ("d'ufficio",
                  ["{Reato} è procedibile solo a querela della persona offesa, giusto?"],
                  ["{Reato} è procedibile d'ufficio?"]),
}
ABROGATO = [
    "Che cosa prevede oggi l'art. {n} {of}?",
    "Posso invocare l'art. {n} {of} nel mio caso?",
    "Come si applica l'art. {n} {of}?",
]


@dataclass
class PremisePair:
    """One question on a false premise, its prompt and its answer."""

    key: str
    kind: str
    code: str
    number: str
    question: str
    instruction: str
    output: str
    keywords: list[str] = field(default_factory=list)

    def sft_row(self) -> dict:
        return {"instruction": self.instruction, "output": self.output, "task": self.kind,
                "source": self.code, "key": self.key}

    def grpo_row(self) -> dict:
        """A GRPO prompt, in make_grpo_prompts.py's shape; graded by
        `eullm_forge.rl.rewards` on ``tipo`` and ``keywords``."""
        return {"id": self.key, "prompt": [{"role": "user", "content": self.instruction}],
                "tipo": self.kind, "keywords": self.keywords, "code": self.code,
                "articolo": self.number}


def say(n: int, unit: str) -> str:
    """A deadline as a sentence writes it: "1 anno", "30 giorni"."""
    return f"{n} {_SINGULAR.get(unit, unit) if n == 1 else unit}"


def _flat(text: str) -> str:
    return " ".join(text.split())


def quote_span(sentence: str, anchor: str = "", max_chars: int = MAX_QUOTE_CHARS) -> str:
    """The sentence, or the words around ``anchor`` in it when it is too long."""
    sentence = _flat(sentence).strip(" .;")
    if len(sentence) <= max_chars:
        return sentence
    words = sentence.split()
    at = next((i for i, w in enumerate(words) if anchor and anchor in w), 0)
    lo, hi = max(0, at - 12), min(len(words), at + 13)
    return " ".join(words[lo:hi]).strip(" .;,")


def shown(records: list[dict], max_chars: int = 3000) -> list[str]:
    """The texts the prompt shows, flattened, cut where `open_book_prompt` cuts them."""
    return [_flat(r.get("text", "")[:max_chars]) for r in records]


def retrieved_articles(index, items, k: int = 3) -> set[tuple[str, str]]:
    """(code, article) of what retrieval returns for each item without our metadata.

    An outside set's question may name the offence and no article ("le
    lesioni aggravate"): the articles it is about are then the ones its
    question retrieves, and those are left out too. Our own exams record
    theirs, and retrieving for thousands of them would leave out far more
    than they are about.
    """
    out = set()
    for it in items:
        md = it.metadata if hasattr(it, "metadata") else it.get("metadata", {})
        if (md.get("code") and md.get("articolo")) or md.get("gruppo"):
            continue
        question = it.question if hasattr(it, "question") else it.get("question", "")
        for r in index.search(question, k):
            out.update((r.get("code"), n) for n in index.articles_of(r))
    return out


def wrong_deadline(n: int, unit: str, stated: set[tuple[int, str]],
                   rng: random.Random) -> int | None:
    """A usual deadline of the same unit that the article does not state."""
    pool = [m for m in USUAL.get(unit, []) if m != n and (m, unit) not in stated]
    return rng.choice(pool) if pool else None


def excluded_articles(items) -> set[tuple[str, str]]:
    """(code, article) of every exam item, by metadata and by what its question names.

    The metadata is how our exams record it (``code``/``articolo``, and the
    group of a practical item); the question is the only record an outside
    set has (the LexIA traps name "art. 640 c.p." in the text), so the
    articles a question names in the code it names are left out too.
    """
    out = set()
    for it in items:
        md = it.metadata if hasattr(it, "metadata") else it.get("metadata", {})
        if md.get("code") and md.get("articolo"):
            out.add((md["code"], str(md["articolo"])))
        # An outside set may say the article only where it records its source.
        for said in (md.get("source"), md.get("false_element")):
            if isinstance(said, str) and (c := named_code(said)):
                out.update((c, n) for n in named_articles(said))
        for ref in md.get("gruppo") or []:
            c, _, a = str(ref).partition(":")
            if c and a:
                out.add((c, a))
        question = it.question if hasattr(it, "question") else it.get("question", "")
        code = named_code(question)
        if code:
            out.update((code, n) for n in named_articles(question))
    return out


def _with_article(index, question: str, code: str, number: str, k: int,
                  rng: random.Random) -> tuple[list[dict], int | None]:
    """What retrieval returns for the question, the article put in if it missed
    it, and the 1-based position of the article's first text."""
    found = index.search(question, k)
    mine = [r for r in index.records if r.get("code") == code
            and number in index.articles_of(r)]
    if mine and not any(r in found for r in mine):
        found = found[:k - 1]
        found.insert(rng.randrange(len(found) + 1), mine[0])
    pos = next((i for i, r in enumerate(found, 1) if r.get("code") == code
                and number in index.articles_of(r)), None)
    return found, pos


def _termine_answer(fmt: str, n: str, of: str, right: str, wrong: str, pos: int,
                    quote: str) -> str:
    if fmt == "plain":
        return (f"La premessa non è corretta: l'art. {n} {of} prevede un termine di {right}, "
                f"non di {wrong} [{pos}]. Il testo dice: «{quote}».")
    return (f"Verifica della premessa: il testo [{pos}] è l'art. {n} {of}, e prevede un "
            f"termine di {right}, non di {wrong}. La premessa della domanda è errata.\n"
            f"Norma applicabile: art. {n} {of} [{pos}]: «{quote}».\n"
            f"Risposta: il termine previsto è di {right}.")


def _sentenza_answer(fmt: str, n: str, of: str, right: str, wrong: str | None, pos: int,
                     quote: str, ruling: str) -> str:
    """A claim attributed to a ruling not in the texts: wrong (``wrong``) or right."""
    unverified = (f"tra i testi riportati non c'è la sentenza {ruling}, quindi non posso "
                  "verificare che cosa abbia stabilito")
    if fmt == "plain":
        verdict = (f"la premessa è errata: l'art. {n} {of} prevede un termine di {right}, non "
                   f"di {wrong}" if wrong
                   else f"l'art. {n} {of} prevede proprio un termine di {right}")
        return (f"{unverified[0].upper()}{unverified[1:]}. Quanto al termine, {verdict} [{pos}]. "
                f"Il testo dice: «{quote}». La sentenza citata va verificata sulla fonte.")
    check = (f"prevede un termine di {right}, non di {wrong}: la premessa della domanda è errata"
             if wrong else f"prevede proprio un termine di {right}: su questo la premessa è "
             "corretta")
    answer = (f"il termine previsto è di {right}" if wrong
              else f"sì, il termine previsto è di {right}")
    return (f"Verifica della premessa: {unverified}. Il testo [{pos}] è l'art. {n} {of}, e "
            f"{check}.\n"
            f"Norma applicabile: art. {n} {of} [{pos}]: «{quote}».\n"
            f"Risposta: {answer}; la sentenza citata va verificata sulla fonte.")


def _vera_answer(fmt: str, n: str, of: str, right: str, pos: int, quote: str) -> str:
    if fmt == "plain":
        return (f"Sì: l'art. {n} {of} prevede proprio un termine di {right} [{pos}]. "
                f"Il testo dice: «{quote}».")
    return (f"Verifica della premessa: il testo [{pos}] è l'art. {n} {of}, e prevede proprio "
            f"un termine di {right}. La premessa della domanda è corretta.\n"
            f"Norma applicabile: art. {n} {of} [{pos}]: «{quote}».\n"
            f"Risposta: sì, il termine previsto è di {right}.")


# One penalty range: "reclusione da sei mesi a tre anni", "arresto da uno a sei mesi"
# (the lower bound's unit left to the upper one's). Normattiva's (( )) marks
# are taken out first.
_PENA = re.compile(r"\b(reclusione|arresto)\s+da\s+([a-z0-9]+)(?:\s+(giorni|mesi|anni))?\s+a\s+"
                   r"([a-z0-9]+)\s+(giorni|mesi|anni)\b", re.IGNORECASE)
# Words may sit in between: art. 624 c.p. reads "Il delitto è punibile a
# querela della persona offesa. Si procede, tuttavia, d'ufficio [se ...]",
# and the first version, knowing only "si procede d'ufficio", took it for
# querela alone. An article naming the other procedibility anywhere is left out.
_QUERELA = re.compile(r"\b(?:si procede|punibil[ei]|procedibil[ei]|perseguibil[ei])\b[^.;]{0,40}?"
                      r"\ba querela\b|\ba querela della persona offesa\b", re.IGNORECASE)
_UFFICIO = re.compile(r"\b(?:si procede|procedibil[ei]|perseguibil[ei])\b[^.;]{0,40}?"
                      r"\bd['’]ufficio\b", re.IGNORECASE)
_ANY_UFFICIO = re.compile(r"\bd['’]ufficio\b", re.IGNORECASE)
# "nei reati punibili a querela" (art. 158 c.p., on the prescription), "le
# ipotesi procedibili d'ufficio" (art. 282-bis c.p.p., a precautionary
# measure): the procedibility of other offences, named in general. Read as
# the article's own, both became "the offence of art. N is procedibile ..."
# in the memory corpus, found by the checkers on 2026-11-15.
_GENERIC = re.compile(r"\b(?:reati|delitti|contravvenzioni|ipotesi|casi|fattispecie)"
                      r"(?:\s+\w+){0,2}\s+$", re.IGNORECASE)
# An article that defines an offence says it punishes: "è punito", "è
# punibile". Not one that names penalties of others (art. 600-septies c.p.,
# confiscation, "la condanna alla reclusione da tre a cinque anni importa
# l'interdizione"), nor a general or procedural one.
_PUNISHES = re.compile(r"\b(?:è|e'|sono)\s+punit[oaie]\b|\b(?:è|e')\s+punibile\b",
                       re.IGNORECASE)
_PUNISH_WORD = re.compile(r"\bpunit[oaie]\b|\bpunibil[ei]\b|\bpena\b|\bsi applica\b",
                          re.IGNORECASE)


def defines_offence(text: str) -> bool:
    """Whether the article punishes a conduct ("è punito"), the only kind whose
    penalty and procedibility a statement may call "the offence's"."""
    return bool(_PUNISHES.search(_flat(text.replace("((", " ").replace("))", " "))))


def _opening(flat: str) -> str:
    """The article's first sentence after its number and heading."""
    body = re.sub(r"^\s*Art\.?\s*\d+[\w-]*\.?\s*(?:\([^)]*\)\.?\s*)?", "", flat)
    return next((st.strip() for _, st in _sentences(body) if st.strip()), "")


def penalty_of(text: str):
    """(kind, low, low unit, high, unit, sentence, comma) of the penalty range
    an article states, or None.

    One range only, or the first of several when the article's first sentence
    states it alone: the base offence of the first comma ("Chiunque ... è
    punito con la reclusione da sei mesi a tre anni"), the later ranges being
    its aggravated or lesser forms. That one is asked about "al primo comma"
    (``comma``), the only form the question can name without ambiguity. One
    range alone gave 44 questions over the whole penal code.
    """
    flat = _flat(text.replace("((", " ").replace("))", " "))
    found = list(_PENA.finditer(flat))
    if not found:
        return None
    m, comma = found[0], ""
    opening = _opening(flat)
    # The range must be the base offence's, in the first sentence beside the
    # word that punishes. One range alone is not enough: art. 507 c.p. punishes
    # "fino a tre anni" and states its only range, "da due a sei anni", for the
    # aggravated form, and the corpus called that the penalty (2026-11-15).
    if m.group(0) not in opening or not _PUNISH_WORD.search(opening):
        return None
    if len(found) > 1:
        if len(_PENA.findall(opening)) != 1:
            return None
        comma = ", al primo comma,"
    lo, hi = number_of(m.group(2).lower()), number_of(m.group(4).lower())
    if not lo or not hi:
        return None
    unit = DEADLINE_UNITS[m.group(5).lower()]
    lo_unit = DEADLINE_UNITS[m.group(3).lower()] if m.group(3) else unit
    sentence = next((st for off, st in _sentences(flat) if off <= m.start() < off + len(st)),
                    flat[max(0, m.start() - 150):m.end() + 50])
    return m.group(1).lower(), lo, lo_unit, hi, unit, sentence, comma


# The heading after the number: "Art. 624. (Furto).", Normattiva's (( )) marks
# made plain. Not every heading names an offence: the general and procedural
# ones are left out.
_HEADING = re.compile(r"^\s*Art\.?\s*[\w-]+\.?\s*\(([^()]{3,70})\)")
_NOT_AN_OFFENCE = re.compile(
    r"^(?:circostanz|disposizion|pen[ae]\b|casi\b|effett|applicazion|norm[ae]\b|definizion|"
    r"caus[ae]\b|misur|confisca|estinzion|querela|procedibilit|aument|diminu|concorso|"
    r"reati?\b|delitti?\b|contravvenzion|sanzion|pubblicazion|interdizion|sospension|"
    r"limit|ambito|abrogat|modific|articol)", re.IGNORECASE)


def offence_name(text: str) -> str | None:
    """The offence the article's heading names, as a question says it, or None."""
    m = _HEADING.match(_flat(text.replace("((", "(").replace("))", ")")))
    if not m:
        return None
    name = m.group(1).strip(" .")
    if _NOT_AN_OFFENCE.match(name) or not name[:1].isalpha():
        return None
    return name[0].lower() + name[1:]


def procedibility_of(text: str):
    """("querela" | "d'ufficio", sentence) of an article that states exactly one, or None."""
    flat = _flat(text.replace("((", " ").replace("))", " "))

    def own(pattern):   # the first mention that is not about offences in general
        return next((m for m in pattern.finditer(flat)
                     if not _GENERIC.search(flat[max(0, m.start() - 60):m.start()])), None)

    q, u = own(_QUERELA), own(_UFFICIO)
    if bool(q) == bool(u) or (q and _ANY_UFFICIO.search(flat)) or (u and "querel" in flat.lower()):
        return None
    m = q or u
    sentence = next((st for off, st in _sentences(flat) if off <= m.start() < off + len(st)),
                    flat[max(0, m.start() - 150):m.end() + 50])
    return ("querela" if q else "d'ufficio"), sentence


def _steps(fmt: str, false: bool, pos: str, label: str, states: str, wrong: str, quote: str,
           answer: str) -> str:
    """The checked steps (or one paragraph) for a premise the text decides."""
    if fmt == "plain":
        lead = "La premessa non è corretta" if false else "Sì"
        tail = f", non {wrong}" if false else ""
        return f"{lead}: {states}{tail} {pos}. Il testo dice: «{quote}»."
    verdict = ("La premessa della domanda è errata." if false
               else "La premessa della domanda è corretta.")
    tail = f", non {wrong}" if false else ""
    return (f"Verifica della premessa: {states}{tail} {pos}. {verdict}\n"
            f"Norma applicabile: {label} {pos}: «{quote}».\n"
            f"Risposta: {answer}")


def _inesistente_answer(fmt: str, n: str, of: str) -> str:
    if fmt == "plain":
        return (f"La domanda parte da un riferimento errato: tra i testi riportati non c'è "
                f"l'art. {n} {of}, e non mi risulta che esista, quindi non posso dirti che "
                "cosa prevede. Verifica il numero dell'articolo e la fonte; se mi indichi "
                "l'argomento, posso cercare la norma pertinente.")
    return (f"Verifica della premessa: tra i testi riportati non c'è l'art. {n} {of}, e non "
            "mi risulta che esista: la domanda parte da un riferimento errato.\n"
            "Norma applicabile: nessuno dei testi riportati è l'articolo citato, quindi non "
            "posso attribuirgli un contenuto.\n"
            f"Risposta: non posso dirti che cosa prevede l'art. {n} {of}. Verifica il numero "
            "dell'articolo e la fonte; se mi indichi l'argomento, posso cercare la norma "
            "pertinente.")


def _numero_answer(fmt: str, fake: str, n: str, of: str, name: str, pos: int, quote: str,
                   penalty: str) -> str:
    if fmt == "plain":
        return (f"La premessa non è corretta: l'art. {fake} {of} non mi risulta che esista; il "
                f"reato di {name} è previsto dall'art. {n} {of} [{pos}], che prevede {penalty}. "
                f"Il testo dice: «{quote}».")
    return (f"Verifica della premessa: tra i testi riportati non c'è l'art. {fake} {of}, e non "
            f"mi risulta che esista: il reato di {name} è previsto dall'art. {n} {of} [{pos}]. "
            "La premessa della domanda è errata.\n"
            f"Norma applicabile: art. {n} {of} [{pos}]: «{quote}».\n"
            f"Risposta: per il reato di {name} l'art. {n} {of} prevede {penalty}.")


def _abrogato_answer(fmt: str, n: str, of: str, pos: int, quote: str) -> str:
    if fmt == "plain":
        return (f"La premessa non è corretta: l'art. {n} {of} è stato abrogato, come riporta "
                f"il testo [{pos}]: «{quote}». Non si applica più; per la disciplina vigente "
                "occorre la norma che lo ha sostituito, se esiste.")
    return (f"Verifica della premessa: il testo [{pos}] riporta che l'art. {n} {of} è stato "
            f"abrogato: «{quote}». La domanda lo presuppone in vigore.\n"
            f"Norma applicabile: l'art. {n} {of} non è più in vigore; la disciplina vigente è "
            "quella della norma che lo ha sostituito, se esiste.\n"
            f"Risposta: l'art. {n} {of} è stato abrogato, quindi non si applica. Se mi indichi "
            "l'argomento, posso cercare la norma oggi in vigore.")


# A part of the article repealed, the rest in force: Normattiva's "COMMA
# ABROGATO DALLA L. 2 LUGLIO 2010, N. 108" in art. 600 c.p., "3) NUMERO
# ABROGATO DALLA L. 28 GIUGNO 2024, N. 90" in art. 617-quater. The corpus
# said both articles were no longer in force (2026-11-15).
_PART_REPEALED = re.compile(
    r"\b(?:comm[ai]|numer[oi]|letter[ae]|period[oi]|parol[ae]|capovers[oi]|alinea|punt[oi])\b"
    r"(?:\s+[\w'’-]+){0,3}?\s+(?:(?:è|e'|sono)\s+stat[oaie]\s+|(?:è|e'|sono)\s+)?abrogat",
    re.IGNORECASE)


def _repealed_sentence(article) -> str:
    """The sentence that says THIS article is repealed, or "".

    "Abrogat" alone is not enough: a codifying decree's final article
    ("Sono abrogati: a) ...") is in force and repeals other acts. The
    sentence must name the article, the norm or a disposition as the thing
    repealed, or say it was repealed by a later act.
    """
    pattern = re.compile(
        r"\b(?:art(?:icol[oi])?\.?|norm[ae]|disposizion[ei])\s+"
        r"(?:\d+[\w-]*\s+)?(?:(?:è|e'|sono)\s+stat[oaie]\s+|è|e'|sono|risulta(?:no)?|viene|vengono)?"
        r"\s*abrogat[oaie]\b"
        r"|\babrogat[oaie]\s+(?:dalla\b|dal\b|dall['’]|con\b|ai sensi\b)",
        re.IGNORECASE,
    )
    for _, s in split_sentences(article.text):
        if pattern.search(s) and not _PART_REPEALED.search(s):
            return s
    if article.heading.strip("()[] ").lower() in ("abrogato", "abrogata"):
        return next((s for _, s in split_sentences(article.text)
                     if "abrogat" in s.lower()), "")
    return ""


def _quoted_in(quote: str, records: list[dict], pos: int) -> bool:
    return bool(quote) and quote in shown(records)[pos - 1]


def build_pairs(index, *, per_code: int, rng: random.Random, fmt: str = "steps",
                exclude: set[tuple[str, str]] = frozenset(), k: int = 3,
                bis_share: float = 0.5, true_share: float = 0.5, named_share: float = 0.5,
                stats: dict | None = None) -> list[PremisePair]:
    """Up to ``per_code`` pairs of each kind for each code, none on an excluded article.

    ``true_share`` of the wrong-deadline questions also get their twin with
    the right deadline (``premessa_vera``). ``named_share`` of the penalty
    and procedibility questions name the offence by its heading instead of
    the article, when the heading names one offence of the code alone.
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}")
    stats = stats if stats is not None else {}

    def skip(reason: str) -> None:
        stats[reason] = stats.get(reason, 0) + 1

    arts = articles_from_records(index.records)
    by_code: dict[str, list] = {}
    for (code, num), a in sorted(arts.items()):
        if code in CODE_LABELS and (code, num) not in exclude:
            by_code.setdefault(code, []).append(a)
    pairs: list[PremisePair] = []
    for code in sorted(by_code):
        of, in_ = CODE_LABELS[code]
        pool = by_code[code]
        live = [a for a in pool if not _repealed_sentence(a) and len(a.text) >= 150]

        timed = [(a, next(iter(d))) for a in live
                 if len(d := _deadlines(a.text)) == 1 and all_deadlines(a.text) == d]
        for a, (n, unit) in rng.sample(timed, min(per_code, len(timed))):
            wrong = wrong_deadline(n, unit, all_deadlines(a.text), rng)
            if wrong is None:
                skip("termine:no_wrong_deadline")
                continue
            values = {"n": a.number, "of": of, "wrong": say(wrong, unit)}
            if a.heading and rng.random() < 0.5:
                q = rng.choice(TERMINE_TOPIC).format(
                    In=in_[0].upper() + in_[1:], in_=in_, heading=a.heading, **values)
            else:
                q = rng.choice(TERMINE_NAMED).format(**values)
            found, pos = _with_article(index, q, code, a.number, k, rng)
            quote = quote_span(_sentence_with(a.text, n, unit), anchor=str(n))
            if pos is None or not _quoted_in(quote, found, pos):
                skip("termine:quote_not_shown")
                continue
            pairs.append(PremisePair(
                key=f"pf-t-{code}-{a.number}", kind="premessa_termine", code=code,
                number=a.number, question=q, instruction=open_book_prompt(q, found),
                output=_termine_answer(fmt, a.number, of, say(n, unit), say(wrong, unit),
                                       pos, quote),
                keywords=[_deadline_keyword(n, unit)]))
            if rng.random() < true_share:
                # The twin, with the deadline the article states: the same
                # template, so only the number tells the two apart.
                tq = q.replace(say(wrong, unit), say(n, unit))
                tfound, tpos = _with_article(index, tq, code, a.number, k, rng)
                if tpos is None or not _quoted_in(quote, tfound, tpos):
                    skip("vera:quote_not_shown")
                    continue
                pairs.append(PremisePair(
                    key=f"pf-v-{code}-{a.number}", kind="premessa_vera", code=code,
                    number=a.number, question=tq, instruction=open_book_prompt(tq, tfound),
                    output=_vera_answer(fmt, a.number, of, say(n, unit), tpos, quote),
                    keywords=[_deadline_keyword(n, unit)]))

        # The same deadlines, attributed to a made-up ruling: half of them
        # wrongly, half rightly (the twin, so that "I cannot verify the
        # ruling" is not read as "the premise is wrong").
        sez = "pen." if "penal" in code else "civ."
        for a, (n, unit) in rng.sample(timed, min(per_code // 2, len(timed))):
            wrong = wrong_deadline(n, unit, all_deadlines(a.text), rng)
            num, year = rng.randrange(1000, 40000), rng.randrange(2015, 2026)
            ruling = f"n. {num}/{year}"
            template = rng.choice(SENTENZA_Q)
            quote = quote_span(_sentence_with(a.text, n, unit), anchor=str(n))
            for false in ([True] if wrong else []) + [False]:
                q = template.format(sez=sez, num=num, year=year, n=a.number, of=of,
                                    claim=say(wrong if false else n, unit))
                found, pos = _with_article(index, q, code, a.number, k, rng)
                if pos is None or not _quoted_in(quote, found, pos):
                    skip("sentenza:quote_not_shown")
                    break
                pairs.append(PremisePair(
                    key=f"pf-{'s' if false else 'sv'}-{code}-{a.number}",
                    kind="premessa_sentenza" if false else "premessa_sentenza_vera",
                    code=code, number=a.number, question=q,
                    instruction=open_book_prompt(q, found),
                    output=_sentenza_answer(fmt, a.number, of, say(n, unit),
                                            say(wrong, unit) if false else None, pos, quote,
                                            ruling),
                    keywords=[_deadline_keyword(n, unit)]))

        taken = {a.number for a in pool}
        last = max(int(re.match(r"\d+", a.number).group()) for a in pool)
        plain = [a for a in live if a.number.isdigit() and f"{a.number}-bis" not in taken]
        made: set[str] = set()
        for _ in range(per_code):
            if plain and rng.random() < bis_share:
                fake = f"{rng.choice(plain).number}-bis"
            else:
                fake = str(last + rng.randint(30, 900))
            if fake in made or (code, fake) in exclude:
                skip("inesistente:repeat")
                continue
            made.add(fake)
            q = rng.choice(INESISTENTE).format(n=fake, of=of)
            # The same test as the note the exam gives (missing_article_note),
            # without giving the note: is the article named in the collection?
            if not index.missing_article_note(q):
                skip("inesistente:exists")
                continue
            found = index.search(q, k)
            pairs.append(PremisePair(
                key=f"pf-i-{code}-{fake}", kind="premessa_inesistente", code=code,
                number=fake, question=q, instruction=open_book_prompt(q, found),
                output=_inesistente_answer(fmt, fake, of)))

        if code == "codice_penale":
            names = [offence_name(a.text) for a in live]
            unique = {n for n in names if n and names.count(n) == 1}

            def named(a) -> str | None:
                name = offence_name(a.text)
                return name if name in unique and rng.random() < named_share else None

            punished = [(a, p) for a in live
                        if defines_offence(a.text) and (p := penalty_of(a.text))]
            for a, (pena, lo, lo_unit, hi, unit, sentence, comma) in rng.sample(
                    punished, min(per_code, len(punished))):
                pool_hi = [m for m in USUAL.get(unit, []) if m != hi
                           and (lo_unit != unit or m > lo)]
                if not pool_hi:
                    skip("pena:no_wrong_maximum")
                    continue
                wrong = rng.choice(pool_hi)
                name = named(a)
                template = rng.choice(PENA_NAMED_Q if name else PENA_Q)
                values = {"n": a.number, "of": of, "comma": comma, "pena": pena,
                          "lo": say(lo, lo_unit), "name": name,
                          "base": ", nell'ipotesi del primo comma," if comma else ""}
                quote = quote_span(sentence, anchor=str(hi))
                states = (f"l'art. {a.number} {of}{comma} prevede la {pena} da "
                          f"{say(lo, lo_unit)} a {say(hi, unit)}")
                for false in [True] + ([False] if rng.random() < true_share else []):
                    q = template.format(hi=say(wrong if false else hi, unit), **values)
                    found, pos = _with_article(index, q, code, a.number, k, rng)
                    if pos is None or not _quoted_in(quote, found, pos):
                        skip("pena:quote_not_shown")
                        break
                    pairs.append(PremisePair(
                        key=f"pf-{'p' if false else 'pv'}-{code}-{a.number}",
                        kind="premessa_pena" if false else "premessa_vera", code=code,
                        number=a.number, question=q, instruction=open_book_prompt(q, found),
                        output=_steps(fmt, false, f"[{pos}]", f"art. {a.number} {of}", states,
                                      f"fino a {say(wrong, unit)}", quote,
                                      f"la pena prevista è la {pena} da {say(lo, lo_unit)} a "
                                      f"{say(hi, unit)}."),
                        keywords=[_deadline_keyword(hi, unit)]))

            # The offence by its name, the article by a number that does not exist.
            for a, (pena, lo, lo_unit, hi, unit, sentence, comma) in rng.sample(
                    punished, min(per_code, len(punished))):
                name = offence_name(a.text)
                fake = unused_number(a.number, taken) if name in unique else None
                if not fake or (code, fake) in exclude:
                    skip("numero:no_name_or_number")
                    continue
                q = rng.choice(NUMERO_Q).format(fake=fake, of=of, name=name)
                if not index.missing_article_note(q):
                    skip("numero:exists")
                    continue
                found, pos = _with_article(index, q, code, a.number, k, rng)
                quote = quote_span(sentence, anchor=str(hi))
                if pos is None or not _quoted_in(quote, found, pos):
                    skip("numero:quote_not_shown")
                    continue
                where = f"{comma.strip(' ,')} " if comma else ""
                penalty = f"{where}la {pena} da {say(lo, lo_unit)} a {say(hi, unit)}"
                pairs.append(PremisePair(
                    key=f"pf-n-{code}-{fake}", kind="premessa_numero", code=code,
                    number=a.number, question=q, instruction=open_book_prompt(q, found),
                    output=_numero_answer(fmt, fake, a.number, of, name, pos, quote, penalty),
                    keywords=[a.number, _deadline_keyword(hi, unit)]))

            stated = [(a, p) for a in live
                      if defines_offence(a.text) and (p := procedibility_of(a.text))]
            for a, (how, sentence) in rng.sample(stated, min(per_code, len(stated))):
                said, false_qs, true_qs = PROC[how]
                quote = quote_span(sentence, anchor="querela" if how == "querela" else "ufficio")
                other = "d'ufficio" if how == "querela" else "a querela della persona offesa"
                states = f"l'art. {a.number} {of} stabilisce che si procede {said}"
                name = named(a)
                reato = (f"il reato di {name}" if name
                         else f"il reato previsto dall'art. {a.number} {of}")
                for false in [True] + ([False] if rng.random() < true_share else []):
                    q = rng.choice(false_qs if false else true_qs).format(
                        reato=reato, Reato=reato[0].upper() + reato[1:])
                    found, pos = _with_article(index, q, code, a.number, k, rng)
                    if pos is None or not _quoted_in(quote, found, pos):
                        skip("procedibilita:quote_not_shown")
                        break
                    pairs.append(PremisePair(
                        key=f"pf-{'q' if false else 'qv'}-{code}-{a.number}",
                        kind="premessa_procedibilita" if false else "premessa_vera",
                        code=code, number=a.number, question=q,
                        instruction=open_book_prompt(q, found),
                        output=_steps(fmt, false, f"[{pos}]", f"art. {a.number} {of}", states,
                                      other, quote,
                                      f"il reato è procedibile {said}."),
                        keywords=[how]))

        repealed = [a for a in pool if _repealed_sentence(a)]
        for a in rng.sample(repealed, min(per_code, len(repealed))):
            line = _repealed_sentence(a)
            q = rng.choice(ABROGATO).format(n=a.number, of=of)
            found, pos = _with_article(index, q, code, a.number, k, rng)
            quote = quote_span(line, anchor="brogat")
            if pos is None or not _quoted_in(quote, found, pos):
                skip("abrogato:quote_not_shown")
                continue
            pairs.append(PremisePair(
                key=f"pf-a-{code}-{a.number}", kind="premessa_abrogato", code=code,
                number=a.number, question=q, instruction=open_book_prompt(q, found),
                output=_abrogato_answer(fmt, a.number, of, pos, quote)))
    return pairs
