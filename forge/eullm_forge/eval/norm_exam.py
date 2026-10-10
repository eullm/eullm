"""An exam built from the text of the law, which nobody working on the models reads.

The ten seed questions found real faults and became the development set: every
fix since has been checked on them, so their score no longer says how a model
does on questions it was not tuned against. A held-out set written by a lawyer
is expensive; one written by ChatGPT has the same failure the models have —
it misremembers article numbers and post-reform deadlines — and a wrong
reference rewards the model that is wrong the same way.

This builds the questions from the legislation files themselves, so every
reference IS the text of an article and cannot be misremembered:

* ``contenuto``   — "Che cosa prevede l'art. N del <codice>?"; reference: the
  article.
* ``termine``     — only articles stating exactly one deadline ("entro
  sessanta giorni"), so the question has one right answer; reference: the
  sentence that states it; keyword: the deadline, in digits or words.
* ``termine_argomento`` — the same deadline asked by the article's heading
  instead of its number, which is how people ask and what retrieval by words
  has to handle.
* ``inesistente`` — an article number past the end of the code. The right
  answer is that it does not exist; a model that describes it is inventing.

The articles are drawn at random with a seed the builder does not print, and
the script that runs it reports counts only, so the exam can live on the
cluster without anyone improving the models against it by reading it.
Nothing here imports torch.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

from .dataset import EvalItem
from .metrics import ANY_DEADLINE, DEADLINE_UNITS, NUMBER_WORDS, normalize_text, number_of
from .retrieval import _HEADER, _article_key

# How each code is named in a question, as "of" and "in" — worded so
# `named_code` recognises it.
CODE_LABELS: dict[str, tuple[str, str]] = {
    "codice_civile": ("del codice civile", "nel codice civile"),
    "codice_penale": ("del codice penale", "nel codice penale"),
    "codice_procedura_civile": ("del codice di procedura civile",
                                "nel codice di procedura civile"),
    "codice_procedura_penale": ("del codice di procedura penale",
                                "nel codice di procedura penale"),
    "codice_consumo": ("del codice del consumo", "nel codice del consumo"),
    "costituzione": ("della Costituzione", "nella Costituzione"),
    "codice_processo_amministrativo": ("del codice del processo amministrativo",
                                       "nel codice del processo amministrativo"),
    "legge_procedimento_amministrativo": ("della legge n. 241/1990",
                                          "nella legge n. 241/1990"),
    "ricorsi_amministrativi": ("del d.P.R. n. 1199/1971", "nel d.P.R. n. 1199/1971"),
}
AMMINISTRATIVO = {"codice_processo_amministrativo", "legge_procedimento_amministrativo",
                  "ricorsi_amministrativi"}

# The numbers and units of a deadline are metrics' (NUMBER_WORDS,
# DEADLINE_UNITS, number_of): one copy, read the same way by the exam, the
# keyword coverage, the GRPO reward and the paired score.
_WORD_FOR = {v: k for k, v in NUMBER_WORDS.items() if k not in ("un", "una")}
# The singular of each plural, from the same table, so a one-unit deadline can
# be named the way it is written rather than the way it is counted.
_SINGULAR = {plural: singular for singular, plural in DEADLINE_UNITS.items()
             if singular != plural}
_DEADLINE = re.compile(
    r"\b(?:entro|nel termine(?: perentorio| di decadenza)? di|non oltre|decorsi|"
    r"nei|trascorsi)\s+(?:il termine (?:perentorio |di decadenza )?di\s+)?"
    r"(\d+|[a-z]+)\s+(giorni|giorno|mesi|mese|anni|anno|ore)\b", re.IGNORECASE)


def _is_rubrica(text: str) -> bool:
    """A rubrica is a short title with a capital: "Termine di prova". Text
    between double parentheses is also how Normattiva marks words an
    amendment inserted, and "in ogni caso" once became the topic of a
    question as if it were a heading."""
    return len(text) < 120 and text[:1].isupper()


@dataclass
class Article:
    """One article, reassembled from however the file chunked it."""

    code: str
    number: str
    text: str

    @property
    def heading(self) -> str:
        """The rubrica, when the text carries one right after the header."""
        lines = [ln.strip() for ln in self.text.splitlines() if ln.strip()]
        for ln in lines[:3]:
            for m in (re.fullmatch(r"\(+\s*(.+?)\s*\)+\.?", ln),
                      re.search(r"\(\(\s*(.+?)\s*\)\)", ln)):
                if m and _is_rubrica(m.group(1)):
                    return m.group(1)
        return ""


# Normattiva appends its amendment notes to the article they amend: a run of
# footnote markers ("(3a) (15a) ((289a))"), a line of dashes, then
# "AGGIORNAMENTO (3a) La L. 3 aprile 1957, n. 235 ha disposto ...". The notes
# are not the article. On the development set of 2026-09-28 a deadline in a
# note ("la presente modifica si applica ... entro centoventi giorni") became
# the answer key to "Quale termine prevede l'art. 289 del codice penale?".
_NOTES = re.compile(r"\s*(?:-{5,}\s*AGGIORNAMENTO\b|Note all'\s*art\.).*", re.DOTALL)
_MARKERS = re.compile(r"(?:\s*\(\(?\d+[a-z]?\)\)?)+\s*$")
# The same markers inside the text, where the note refers to a single comma:
# "... argomenti di prova. (171) ((173)) Se rifiuta il terzo ...".
_INLINE_MARKERS = re.compile(r"(?:\s+\(\(?\d+[a-z]?\)\)?)+(?=\s)")


def strip_notes(text: str) -> str:
    """The article without Normattiva's notes (the "AGGIORNAMENTO" blocks and
    the "Note all'art." of the consolidated texts) and the footnote markers
    that point at them."""
    text = _MARKERS.sub("", _NOTES.sub("", text))
    return _INLINE_MARKERS.sub("", text).strip()


# A line of a table of contents: a header and a title, nothing else. Real
# articles have a body; one shorter than this could never be asked about
# anyway (`_usable` wants 150 characters).
_INDEX_LINE_CHARS = 150


def _drop_index_lines(text: str, marks: list) -> tuple[str, list]:
    """The record without the lines of a table of contents, and its headers.

    The legislation files open the administrative codes with one: a heading
    reading ``INDICE GENERALE`` and a line per article, each a number and a
    title with no text of its own. Read at face value, every line leaves a stub
    under its number, and the real article then looks like a second, ambiguous
    occurrence of a number already there -- so it is dropped, and the code
    contributes nothing. The index is recognised by its shape, not by its
    heading, because it is longer than one chunk and only the first one begins
    with ``INDICE``: in a record with several headers, a stretch from one
    header to the next that is shorter than `_INDEX_LINE_CHARS` is blanked out
    (same length, so every position stays where it was) and the headers are
    read again. A record with a single header is left alone, so a real
    article whose rubrica is "Indice delle materie" is still an article, and
    the last two lines of an index, alone in a chunk, are still caught.
    """
    if len(marks) < 2:
        return text, marks
    bounds = [m.start() for m in marks] + [len(text)]
    out = text
    for a, b in zip(bounds, bounds[1:]):
        if len(" ".join(text[a:b].split())) < _INDEX_LINE_CHARS:
            out = out[:a] + " " * (b - a) + out[b:]
    if out == text:
        return text, marks
    return out, list(_HEADER.finditer(out))


def articles_from_records(records: list[dict]) -> dict[tuple[str, str], Article]:
    """Split legislation records into whole articles, keyed by (code, number).

    A record may hold one article, the continuation of the previous one, or
    several (the c.p.a. chunks). Text before the first header of a record
    continues the article before it. A number that turns up twice in one
    code, not as a continuation — the allegati of the c.p.a. restart their
    numbering — is ambiguous and dropped: a question about it has no single
    right answer.
    """
    parts: dict[tuple[str, str], list[str]] = defaultdict(list)
    ambiguous: set[tuple[str, str]] = set()
    last: dict[str, str] = {}
    for r in records:
        code, text = r.get("code") or "", r.get("text", "")
        marks = list(_HEADER.finditer(text))
        # A table of contents is not the articles (see `_drop_index_lines`).
        text, marks = _drop_index_lines(text, marks)
        lead = text[: marks[0].start()] if marks else text
        if lead.strip() and code in last and r.get("chunk_index", 0):
            parts[(code, last[code])].append(lead)
        for i, m in enumerate(marks):
            key = (code, _article_key(m.group(1), m.group(2) or ""))
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            if parts.get(key) and last.get(code) != key[1]:
                ambiguous.add(key)
            parts[key].append(text[m.start():end])
            last[code] = key[1]
    return {k: Article(k[0], k[1], strip_notes("\n".join(v)))
            for k, v in parts.items() if k not in ambiguous}


def _deadlines(text: str) -> set[tuple[int, str]]:
    found = set()
    for num, unit in _DEADLINE.findall(_unmarked(text)):
        n = number_of(num)
        if n:
            found.add((n, DEADLINE_UNITS[unit.lower()]))
    return found


# Any "<number> <unit>", whatever precedes it (metrics.ANY_DEADLINE).
# `_DEADLINE` only knows the
# phrasings that introduce the deadline a question is built on ("entro",
# "non oltre", "decorsi"...), so an article that also says "un termine non
# inferiore a venti giorni" (art. 554-ter c.p.p.), "non oltre i dieci giorni"
# (art. 2861 c.c.) or "trascorsi ((cinque)) anni" (art. 58 c.c., with
# Normattiva's amendment marks) looked like it had one deadline when it has
# two. "Quale termine prevede l'art. N?" then has two right answers and the
# exam accepted one: the review of 2026-10-03 found a model marked wrong for
# the other, true, deadline three times in eighteen.


def _unmarked(text: str) -> str:
    """The text without Normattiva's (( )) amendment marks, which split a
    deadline from its number ("trascorsi ((cinque)) anni")."""
    return text.replace("((", " ").replace("))", " ")


def all_deadlines(text: str) -> set[tuple[int, str]]:
    """Every distinct "<number> <unit>" an article states, however phrased."""
    found = set()
    for num, unit in ANY_DEADLINE.findall(_unmarked(text)):
        n = number_of(num)
        if n:
            found.add((n, DEADLINE_UNITS[unit.lower()]))
    return found


def _deadline_keyword(n: int, unit: str) -> str:
    """The keyword a deadline answer is scored on, in digits or in words.

    ``unit`` has already been pluralised, so a one-unit deadline needs the
    singular put back: coverage is a normalised substring test, and neither
    "un anno" nor "1 anno" contains "1 anni". That made every single-unit
    deadline item unwinnable -- the correct answer scored zero, and a model
    that invented a different deadline scored the same.
    """
    if n == 1:
        singular = _SINGULAR.get(unit, unit)
        # "un'ora" normalises to "un ora", so the article form covers it.
        return f"1 {singular}|un {singular}"
    alts = [f"{n} {unit}"]
    if n in _WORD_FOR:
        alts.append(f"{_WORD_FOR[n]} {unit}")
    return "|".join(alts)


def _sentences(flat: str) -> list[tuple[int, str]]:
    """The flattened text's sentences, each with where it starts."""
    out, offset = [], 0
    for sentence in re.split(r"(?<=[.;])\s+", flat):
        out.append((offset, sentence))
        offset += len(sentence) + 1  # the single space the split consumed
    return out


def _sentence_with(text: str, n: int, unit: str) -> str:
    """The sentence of the article that states the deadline.

    Located by the deadline pattern that found the deadline, not by the bare
    number: an article can carry that number earlier for another reason — a
    rate ("il 6 per cento"), a cross-reference ("l'articolo 120") — and that
    sentence then becomes the answer key while the rubric still asks for the
    deadline, so the grader is handed a reference that states none and marks
    the correct answer wrong.
    """
    flat = " ".join(text.split())
    spans = _sentences(flat)
    for m in _DEADLINE.finditer(flat):
        if number_of(m.group(1)) == n and DEADLINE_UNITS.get(m.group(2).lower()) == unit:
            for offset, sentence in spans:
                if offset <= m.start() < offset + len(sentence):
                    return sentence.strip()
    # Nothing in the article states it that way after all: keep the old
    # behaviour of quoting the sentence that merely mentions the number,
    # rather than nothing.
    alts = [str(n)] + ([_WORD_FOR[n]] if n in _WORD_FOR else [])
    pat = re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.IGNORECASE)
    for _, sentence in spans:
        if pat.search(sentence):
            return sentence.strip()
    return flat[:400]


# The grader reads the reference to decide what is right, and treats what
# the reference does not say as invented. Cut at 800 characters, it failed
# answers that quoted the article correctly past the cut (development set,
# 2026-09-28: art. 11 and 6 of L. 241/1990, art. 452-ter c.p.). So the
# reference is the whole article, and articles too long to hand the grader
# whole are not asked about.
MAX_REFERENCE_CHARS = 6000


def _flat(text: str) -> str:
    return " ".join(text.split())


# What the grader is told, per kind. Version 2 (2026-10-03). The first
# contenuto rubric said "sbagliato se attribuisce all'articolo contenuti che
# il testo non contiene", and the judge applied it to true context: the
# amendment history of art. 56 Cost., the abolition of the death penalty after
# art. 286 c.p. Checked blind on 40 answers, six of its seven harsh verdicts
# fell on such long answers of the untuned models, so every comparison between
# a verbose base model and a terse tuned one leaned towards the tuned one.
# What is wrong is unchanged: contradicting the text, describing another
# article, or inventing rules and presenting them as the article's.
CONTENT_RUBRIC = (
    "Il riferimento è il testo integrale dell'articolo. Corretto se ne riporta il "
    "contenuto essenziale senza contraddirlo. Informazioni in più che non contraddicono "
    "il testo (storia e modifiche della norma, contesto, esempi, rinvii) non sono errori. "
    "Parziale se manca una parte essenziale dell'articolo o se riporta in modo inesatto "
    "un suo punto. Sbagliato se descrive un altro articolo, contraddice il testo, o "
    "presenta come contenuto dell'articolo regole che il testo non contiene.")


def deadline_rubric(n: int, unit: str) -> str:
    """The rubric of a deadline question."""
    return (f"Corretto se indica il termine di {n} {unit}. Altri dettagli presenti nel "
            "testo dell'articolo, e informazioni in più che non lo contraddicono, non sono "
            "errori. Sbagliato se indica un termine diverso, nessun termine, o nega che "
            "l'articolo preveda il termine.")


_RUBRIC_DEADLINE = re.compile(r"termine di (\d+) (\w+)")


def rubric_v2(item_id: str, rubric: str, reference: str = "") -> str:
    """The version-2 rubric for an item graded under the first one.

    So answers already given are graded again without asking the models
    again (judge_answers.py --rubric v2). A deadline item whose article turns
    out to state more than one deadline (see `all_deadlines`) also accepts
    any of them: the question did not say which.
    """
    kind = item_id.split("-")[1] if item_id.startswith("norm-") else ""
    if kind == "contenuto":
        return CONTENT_RUBRIC
    if kind in ("termine", "termine_argomento"):
        m = _RUBRIC_DEADLINE.search(rubric)
        if not m:
            return rubric
        out = deadline_rubric(int(m.group(1)), m.group(2))
        text = reference.split("Testo integrale dell'articolo:", 1)[-1]
        others = sorted(all_deadlines(text) - {(int(m.group(1)), m.group(2))})
        if others:
            alts = ", ".join(f"{k} {u}" for k, u in others)
            out += (f" L'articolo prevede anche altri termini ({alts}) e la domanda non dice "
                    "quale: è corretto anche indicare correttamente uno di questi.")
        return out
    return rubric


def _usable(a: Article) -> bool:
    head = normalize_text(a.text[:300])
    return (150 <= len(a.text) and len(_flat(a.text)) <= MAX_REFERENCE_CHARS
            and "abrogat" not in head)


def build_exam(records: list[dict], per_code: int = 10, seed: int | None = None,
               codes: set[str] | None = None,
               exclude: set[tuple[str, str]] | None = None) -> list[EvalItem]:
    """Draw the exam: for each code, up to ``per_code`` items of each kind.

    ``seed`` None means a random one, which is the point: the builder must
    not be able to reproduce the draw from anything it can see.

    ``exclude`` holds (code, article) pairs the models were trained on — the
    open-book stage-3 pairs, see `trained_articles` — including the made-up
    numbers of the absent-article pairs. None of them is drawn: an exam that
    asks what training answered measures recall of the training set.
    """
    exclude = exclude or set()
    rng = random.Random(seed if seed is not None else random.SystemRandom().random())
    arts = articles_from_records(records)
    by_code: dict[str, list[Article]] = defaultdict(list)
    for a in arts.values():
        if (a.code in CODE_LABELS and (codes is None or a.code in codes) and _usable(a)
                and (a.code, a.number) not in exclude):
            by_code[a.code].append(a)

    items: list[EvalItem] = []
    for code in sorted(by_code):
        pool = by_code[code]
        of, in_ = CODE_LABELS[code]
        vertical = "amministrativo" if code in AMMINISTRATIVO else "civile_penale"

        def item(kind, a_num, question, reference, keywords, rubric, _code=code,
                 _vertical=vertical):
            return EvalItem(
                id=f"norm-{kind}-{_code}-{a_num}", domain="legal", lang="it",
                category=_code, question=question, reference=reference, rubric=rubric,
                keywords=keywords,
                metadata={"tipo": kind, "code": _code, "articolo": a_num,
                          "vertical": _vertical, "fonte": "testo di legge"})

        for a in rng.sample(pool, min(per_code, len(pool))):
            items.append(item(
                "contenuto", a.number, f"Che cosa prevede l'art. {a.number} {of}?",
                _flat(a.text), [], CONTENT_RUBRIC))

        # One deadline by the statute's phrasing AND no other number of days,
        # months or years anywhere in the article (see `all_deadlines`).
        timed = [(a, next(iter(d))) for a in pool
                 if len(d := _deadlines(a.text)) == 1 and all_deadlines(a.text) == d]
        for a, (n, unit) in rng.sample(timed, min(per_code, len(timed))):
            kw = [_deadline_keyword(n, unit)]
            # The sentence that states the deadline first, so the key is
            # unmistakable; then the article, so that what else a right
            # answer says can be checked instead of counted as invented.
            ref = (f"{_sentence_with(a.text, n, unit)}\n\n"
                   f"Testo integrale dell'articolo: {_flat(a.text)}")
            rub = deadline_rubric(n, unit)
            items.append(item("termine", a.number,
                              f"Quale termine prevede l'art. {a.number} {of}?",
                              ref, kw, rub))
            if a.heading:
                items.append(item("termine_argomento", a.number,
                                  f"{in_[0].upper()}{in_[1:]}, in materia di "
                                  f"«{a.heading}», qual è il termine previsto?",
                                  ref, kw, rub))

        last = max(int(re.match(r"\d+", a.number).group()) for a in pool)
        drawn: set[int] = set()
        for _ in range(max(1, per_code // 5)):
            fake = last + rng.randint(50, 900)
            # Redrawn against the trained numbers AND against this draw:
            # a repeated randint makes two items with the same id, and the
            # setdefault below silently discards the second, so the exam
            # comes back one item short.
            while (code, str(fake)) in exclude or fake in drawn:
                fake = last + rng.randint(50, 900)
            drawn.add(fake)
            it = item(
                "inesistente", str(fake), f"Che cosa prevede l'art. {fake} {of}?",
                f"Non esiste l'art. {fake} {of}.",
                # Only ways of saying the article is not there. "non contiene"
                # and "non prevede un" are satisfied by an answer that invents
                # the article's content and then hedges, which the rubric calls
                # wrong.
                ["non esiste|inesistente|non è previsto"],
                "Corretto solo se dice che l'articolo non esiste; sbagliato se ne "
                "descrive un contenuto.")
            # What the draw assumed, kept in the metadata so the assumption is
            # auditable: `last` is the end of the corpus this exam was built
            # from, which is the end of the code only if the corpus is whole.
            # The reference does not state it, because the builder cannot know
            # it -- a corpus that stops at art. 120 once produced "l'art. 969
            # non esiste: la numerazione arriva all'art. 120", and art. 969
            # exists.
            it.metadata["last_article"] = last
            items.append(it)
    unique: dict[str, EvalItem] = {}
    for it in items:
        unique.setdefault(it.id, it)
    return list(unique.values())


def trained_articles(pairs) -> set[tuple[str, str]]:
    """(code, article) of every open-book training pair, read from its key.

    Grounded pairs are keyed ``ob-g-<code>-<article>``, absent-article pairs
    ``ob-m-<code>-<number>-<i>`` (see `openbook_gen`); codes are written with
    underscores, so the first hyphen after the prefix ends the code. The RAFT
    pairs of make_raft_absent.py append ``-absent`` to a grounded key. GRPO
    prompts (make_grpo_prompts.py) have no key but say ``code`` and
    ``articolo`` outright. Other pairs carry no article and are skipped.

    Every file a model was trained on goes through here before a development
    set is drawn, the GRPO prompts included: an article a model was rewarded
    on is as much training as one it was shown an answer for.
    """
    out = set()
    for p in pairs:
        key = str(p.get("key", ""))
        if not key and p.get("code") and p.get("articolo"):
            out.add((str(p["code"]), str(p["articolo"])))
            continue
        key = re.sub(r"-absent$", "", key)
        if key.startswith("ob-g-"):
            code, _, number = key[5:].partition("-")
            number = re.sub(r"-v\d+$", "", number)     # a second question, same article
        elif key.startswith("ob-m-"):
            code, _, rest = key[5:].partition("-")
            number = rest.split("-")[0]
        else:
            continue
        if code and number:
            out.add((code, number))
    return out


def retrieval_hits(items: list[EvalItem], index, k: int = 3) -> dict[str, dict[str, float]]:
    """How often the retrieval puts the item's own article in the top 1 and top k.

    Measured on the drawn exam rather than on the ten development questions,
    so an improvement to retrieval has to hold on articles nobody picked.
    ``inesistente`` items have no article to find and are skipped.

    An item with no ``tipo`` is tallied under ``?`` rather than under
    ``None``. check_retrieval.py takes eval files as arguments without
    checking where they came from, and the project's own seed set carries no
    ``tipo`` in any of its items -- which made it die sorting ``None`` against
    a string, before printing a single row.
    """
    from .retrieval import record_articles

    # The index resolves a continuation chunk to its article (the text of a
    # long article is mostly in those); the chunk alone names none, and
    # reading it alone counted a right hit as a miss.
    articles = getattr(index, "articles_of", record_articles)
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for it in items:
        code, art = it.metadata.get("code"), it.metadata.get("articolo")
        kind = it.metadata.get("tipo") or "?"
        if kind == "inesistente":
            continue
        found = index.search(it.question, k)
        ok = [r.get("code") == code and art in articles(r) for r in found]
        t = tally[kind]
        t[0] += 1
        t[1] += bool(ok[:1] and ok[0])
        t[2] += any(ok)
    return {kind: {"n": n, "top1": a / n, f"top{k}": b / n}
            for kind, (n, a, b) in sorted(tally.items())}
