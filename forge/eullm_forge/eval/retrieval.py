"""Find the text of the norm a legal question is about, so the model can read it.

legal-it-4b v0.2, asked closed-book, put the TAR appeal deadline at 30 days
(it is 60) and the ricorso straordinario at 30 (it is 120). Qwen's own 4B
instruct model said article 2043 of the civil code does not exist. A model
of this size does not reliably remember what a given article says, and in
law the number is the answer. The remedy everyone uses is to hand the model
the text instead of hoping it remembers it; this is the retrieval half.

Two ways in, tried in order:

* **By article.** A question that names an article ("art. 2043 c.c.",
  "l'articolo 27 della Costituzione") gets that article, looked up by code
  and number in the records `prepare_legislation.py` writes. Asking by
  number is how lawyers ask, and ranking cannot beat an exact lookup.
* **By words.** Everything else goes to BM25 over the same records. Plain,
  dependency-free, and good enough to answer the question this exists for:
  does reading the norm fix the answers?

Records are the ``legislazione_*.chunks.jsonl`` lines: ``text``, ``code``
(e.g. ``codice_civile``), ``article_num``. Nothing here imports torch.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import normalize_text

# Words that say nothing about which norm is meant. Kept short on purpose:
# BM25's idf already discounts what is common, this only drops the noise.
_STOP = set("""
il lo la i gli le un uno una di a da in con su per tra fra e o ma se che chi
cui non piu del dello della dei degli delle al allo alla ai agli alle dal
dallo dalla dai dagli dalle nel nello nella nei negli nelle sul sullo sulla
sui sugli sulle come quale quali quando dove cosa sono essere ha hanno deve
devono puo possono viene questo questa questi queste quello quella
art artt articolo articoli comma
""".split())

# How a question names a code, normalised, mapped to the ``code`` field.
# Longest names first, so "codice di procedura civile" is not read as
# "codice civile".
CODE_NAMES: list[tuple[str, str]] = [
    ("codice del processo amministrativo", "codice_processo_amministrativo"),
    ("processo amministrativo", "codice_processo_amministrativo"),
    ("c p a", "codice_processo_amministrativo"),
    ("104 2010", "codice_processo_amministrativo"),
    ("241 1990", "legge_procedimento_amministrativo"),
    ("legge sul procedimento amministrativo", "legge_procedimento_amministrativo"),
    ("1199 1971", "ricorsi_amministrativi"),
    ("codice di procedura civile", "codice_procedura_civile"),
    ("codice di procedura penale", "codice_procedura_penale"),
    ("procedura civile", "codice_procedura_civile"),
    ("procedura penale", "codice_procedura_penale"),
    ("codice del consumo", "codice_consumo"),
    ("codice civile", "codice_civile"),
    ("codice penale", "codice_penale"),
    ("costituzione", "costituzione"),
    ("c p c", "codice_procedura_civile"),
    ("c p p", "codice_procedura_penale"),
    ("c c", "codice_civile"),
    ("c p", "codice_penale"),
    ("gdpr", "gdpr"),
    ("2016 679", "gdpr"),
    ("ai act", "ai_act"),
    ("2024 1689", "ai_act"),
]

_ARTICLE = re.compile(r"\bart(?:icolo|icoli|t)?\s+(\d+)(?:\s*(bis|ter|quater|quinquies))?\b")


def tokens(text: str) -> list[str]:
    """Normalised words worth matching on."""
    return [w for w in normalize_text(text).split()
            if w not in _STOP and (len(w) > 2 or w.isdigit())]


def named_code(text: str) -> str | None:
    """The code a question names, if it names one."""
    norm = f" {normalize_text(text)} "
    for name, code in CODE_NAMES:
        if f" {name} " in norm:
            return code
    return None


_SUFFIXES = "bis|ter|quater|quinquies|sexies|septies|octies|novies|decies"
# An article's own header, at the start of a line: "Art. 2043.", "Art. Art. 1."
# (the XML parser writes the number with its own "Art." prefix), "Art. 29 -".
# Anchored to a line start so "ai sensi dell'art. 2043" in the body of another
# article is not read as that article.
_HEADER = re.compile(
    rf"(?im)^[ \t]*(?:art\.[ \t]*)?art(?:icolo)?\.?[ \t]*(\d+)(?:[ \t-]*({_SUFFIXES}))?\b")
_NUMBER = re.compile(rf"(?i)(\d+)(?:[\s-]*({_SUFFIXES}))?")


def _article_key(num: str, suffix: str = "") -> str:
    return f"{int(num)}-{suffix.lower()}" if suffix else str(int(num))


def record_articles(record: dict) -> list[str]:
    """The articles a record belongs to, read from what the files really hold.

    The legislation files are not uniform, and the first version of this
    module assumed they were: ``article_num`` is "" for the codes parsed out
    of the Normattiva ZIP, "Art. 1." for the laws parsed from single XML, and
    a chunk of the codice del processo amministrativo can hold several
    articles (or the whole index). So the field is used when it holds a
    number, and the headers in the text otherwise.
    """
    m = _NUMBER.search(str(record.get("article_num") or ""))
    if m:
        return [_article_key(m.group(1), m.group(2) or "")]
    return [_article_key(n, sfx) for n, sfx in _HEADER.findall(record.get("text", ""))]


_PAREN_TITLE = re.compile(r"\(+\s*([^()]+?)\s*\)+\.?")


def record_heading(record: dict) -> str:
    """The rubrica of the article a record starts, or "".

    The three ways the files write it, right after the header:
    "Art. 2043. \\n (Risarcimento per fatto illecito).", "Art. Art. 2.
    ((Conclusione del procedimento))" on the header's own line, and the c.p.a.
    style "Art. 29 \\n Azione di annullamento \\n 1. L'azione ...". A title
    starts with a capital: words between double parentheses are also how
    Normattiva marks text an amendment inserted.
    """
    text = record.get("text", "")
    m = _HEADER.search(text)
    if not m:
        return ""
    rest = text[m.end():m.end() + 300].lstrip(" .\t")
    lines = [ln.strip() for ln in rest.splitlines() if ln.strip()]
    if not lines:
        return ""
    first = lines[0]
    title = _PAREN_TITLE.match(first)
    if title:
        cand = title.group(1).strip()
    elif not first.startswith("(") and not first[:1].isdigit() and not first.endswith(
            (".", ";", ":", ",")) and len(first) < 100:
        cand = first                      # the c.p.a. style: a bare title line
    else:
        return ""
    return cand if cand[:1].isupper() and len(cand) < 120 else ""


def named_articles(text: str) -> list[str]:
    """Article numbers a question names, as the records spell them."""
    return [_article_key(num, suffix) for num, suffix in _ARTICLE.findall(normalize_text(text))]


@dataclass
class NormIndex:
    """Article lookup plus BM25 over legislation records."""

    records: list[dict]
    k1: float = 1.5
    b: float = 0.75
    # How many times the words of an article's rubrica count, on top of their
    # place in the text. People ask by topic in the rubrica's words ("il
    # contenuto della citazione"), and in plain BM25 a long article's body
    # outweighs its own title. The usual fix is a field weight. 0 — off — is
    # the default until a setting is chosen on the development sets
    # (scripts/check_retrieval.py): changing it changes what every model is
    # shown, so scores from before and after are not comparable.
    heading_boost: int = 0
    _docs: list[Counter] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _avg: float = 0.0

    def __post_init__(self) -> None:
        # Which article each record is, once. A chunk without a header of its
        # own continues the previous record of the same code: articles are
        # written in order, a long one as consecutive chunks.
        self._arts: list[list[str]] = []
        self._headings: list[str] = []
        last: dict[str, str] = {}
        last_heading: dict[str, str] = {}
        for r in self.records:
            arts = record_articles(r)
            code = r.get("code") or ""
            heading = record_heading(r)
            if arts:
                last[code] = arts[-1]
                last_heading[code] = heading
            elif r.get("chunk_index", 0) and code in last:
                arts = [last[code]]
                heading = last_heading.get(code, "")
            self._arts.append(arts)
            self._headings.append(heading)
        # By identity, the way search() tells its own hits apart, so a caller
        # holding a record can ask which articles it belongs to.
        self._arts_of = {id(r): arts for r, arts in zip(self.records, self._arts)}
        self._docs = []
        for r, heading in zip(self.records, self._headings):
            doc = Counter(tokens(r.get("text", "")))
            for w in tokens(heading) * self.heading_boost:
                doc[w] += 1
            self._docs.append(doc)
        for d in self._docs:
            self._df.update(d.keys())
        self._avg = sum(sum(d.values()) for d in self._docs) / max(1, len(self._docs))

    @classmethod
    def from_files(cls, paths: list[str | Path]) -> NormIndex:
        """Load every record of the given JSONL files."""
        records = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                records.extend(json.loads(line) for line in f if line.strip())
        if not records:
            raise ValueError(f"no legislation records in {[str(p) for p in paths]}")
        return cls(records)

    def by_article(self, question: str) -> list[dict]:
        """Records of the article(s) the question names in the code it names."""
        code = named_code(question)
        nums = named_articles(question)
        if not code or not nums:
            return []
        hits = [(len(arts), i) for i, (r, arts) in enumerate(zip(self.records, self._arts))
                if r.get("code") == code and set(arts) & set(nums)]
        # A chunk that is the article itself before one that merely lists it
        # among many (an index); within an article, its chunks in order.
        hits.sort(key=lambda h: (h[0] > 3, h[1]))
        return [self.records[i] for _, i in hits]

    def bm25(self, question: str, k: int, code: str | None = None) -> list[dict]:
        """The k records BM25 ranks highest for the question, within ``code``
        when the question names one."""
        q = tokens(question)
        n = len(self._docs)
        scores = []
        for i, d in enumerate(self._docs):
            if code and self.records[i].get("code") != code:
                continue
            length = sum(d.values())
            s = 0.0
            for w in q:
                tf = d.get(w, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self._df[w] + 0.5) / (self._df[w] + 0.5))
                s += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * length / (self._avg or 1)))
            if s > 0:
                scores.append((s, i))
        # A tie keeps index order, the rule rrf() applies to a fused ranking
        # and its test states. reverse=True on (score, index) tuples reversed
        # both keys, so equal scores came back last-record-first.
        scores.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self.records[i] for _, i in scores[:k]]

    def missing_article_note(self, question: str) -> str:
        """What to tell the model when the question names an article the
        collection does not hold, or "" when it names none or holds it.

        Without this, retrieval quietly fills the prompt with other articles
        of the same code, and a model asked about a nonexistent article
        describes one of those as if it were the one asked: on the held-out
        exam of 2026-09-27 every legal-it model did so on 18 questions out
        of 18. Saying so in the prompt gives the right answer somewhere to
        come from.
        """
        code = named_code(question)
        nums = named_articles(question)
        if not code or not nums or self.by_article(question):
            return ""
        what = ", ".join(f"art. {n}" for n in nums)
        return (f"Nota: nella raccolta normativa non è presente {what} "
                f"({code.replace('_', ' ')}).")

    def articles_of(self, record: dict) -> list[str]:
        """The articles a record belongs to, continuations resolved.

        ``record_articles`` reads what the chunk itself says, so a chunk that
        continues an article -- ``article_num`` is "" for the codes parsed out
        of the Normattiva ZIP, and only the first chunk carries a header --
        reports no article at all. This is the answer the index works out in
        ``_arts``, which any caller asking "is this chunk the article I am
        removing?" has to use: the text of a long article is mostly in its
        continuation chunks, so those are what BM25 hands back.
        """
        return self._arts_of.get(id(record), record_articles(record))

    def search(self, question: str, k: int = 3) -> list[dict]:
        """Named article first, then BM25 to fill up to k, without repeats."""
        found = self.by_article(question)[:k]
        seen = {id(r) for r in found}
        for r in self.bm25(question, k, code=named_code(question)):
            if len(found) >= k:
                break
            if id(r) not in seen:
                found.append(r)
                seen.add(id(r))
        return found


def label(record: dict) -> str:
    """How a record is cited in the prompt: code and article."""
    code = (record.get("code") or "").replace("_", " ")
    arts = record_articles(record)
    if len(arts) == 1:
        return f"{code}, art. {arts[0]}"
    if arts:
        return f"{code}, artt. {arts[0]}-{arts[-1]}"
    return code


def open_book_prompt(question: str, records: list[dict], max_chars: int = 3000,
                     note: str = "") -> str:
    """The question with the retrieved texts in front of it.

    Worded like the context tasks of stage 3 — a text, then what to do with
    it — so the model meets the format it was trained on.

    ``max_chars`` matches the ``--max-chars`` the corpus is chunked with
    (prepare_legislation.py), so a retrieved article is shown whole. A text
    that does not fit is marked as cut: a block that simply stops mid-word
    reads as the end of the article, and the answer gets graded on it.
    """
    if not records and not note:
        return question
    blocks = []
    for i, r in enumerate(records, 1):
        text = r.get("text", "")
        body = text[:max_chars].rstrip()
        if len(text) > max_chars:
            body += " […]"
        blocks.append(f"[{i}] {label(r)}\n{body}")
    if not blocks:
        blocks = ["(nessun testo pertinente trovato)"]
    head = (note + "\n\n") if note else ""
    return (head + "Testi normativi di riferimento:\n\n" + "\n\n".join(blocks)
            + "\n\nRispondi alla domanda basandoti sui testi sopra, se sono "
              "pertinenti.\n\nDomanda: " + question)
