"""Is model A better than model B on the same questions, or is it noise?

Every candidate so far has been compared by its total on the 207-question
exam, and the totals sit between 160 and 170. A total throws away what the
exam knows: which questions each model got right. Two models that both score
169 can disagree on thirty questions, or on two; only the second pair is the
same model. The paired test asks the question that decides: of the questions
where exactly one of the two is right, how many go each way? With b of them
for A and c for B, under "no difference" each goes either way with
probability 1/2, so the two-sided exact binomial (McNemar) p-value is the
chance of a split at least as uneven as b:c.

On 207 questions with the 10-20% disagreement our models show, a real
difference needs to be some 11-16 questions to show up; on 900 it is
roughly half as many in proportion. That is why development decisions are
taken on the large development set, and the held-out exam is kept for the
end.

A second thing a total hides is how long the answers are. An LLM judge can
favour long answers or short ones (Dubois et al. 2024; Soumik 2026), and our
SFT models answer in 440 characters where their base answers in 1,160. So
every comparison also reports, among the questions where the two disagree
and the answers differ in length, how often the answer judged right is the
longer one: far from one half, the judge may be grading length as much as
law. A tie in length is no evidence either way, so it is left out.

Nothing here prints a question: ids name the article asked.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("termine_argomento", "termine", "contenuto", "inesistente")
_KIND = re.compile(r"^norm-(" + "|".join(KINDS) + r")-")

# What a person writes in the review sheet, read as the judge's labels.
HUMAN_LABELS = {"corretto": "correct", "corretta": "correct", "giusto": "correct",
                "parziale": "partial", "sbagliato": "wrong", "sbagliata": "wrong",
                "errato": "wrong", "errata": "wrong",
                "correct": "correct", "partial": "partial", "wrong": "wrong"}


def kind_of(item_id: str) -> str:
    """The question kind an exam id names (norm-<kind>-<code>-<article>)."""
    m = _KIND.match(item_id)
    return m.group(1) if m else "?"


_RUBRIC_DEADLINE = re.compile(r"termine di (\d+) (\w+)")


def verifiable(row: dict) -> float | None:
    """1.0 / 0.0 by a check that needs no judge, or None for a kind that does.

    A deadline question has a number for an answer, and an absent article
    one sentence: both are checked the way the GRPO reward checks them
    (`eullm_forge.rl.rewards`), so this score cannot share the judge's
    habits. On forty answers graded blind by a second model the judge agreed
    on 68-85%, whichever judge and rubric; a difference that holds here as
    well as under the judge does not depend on it.

    A deadline answer is right when it names the deadline the question was
    built on -- or, for an item drawn before the builder learnt to skip them,
    another deadline its article states (the version-2 rubric's rule) -- and
    not more than `MAX_DEADLINES` of them, nor a refusal.
    """
    from ..rl.rewards import MAX_DEADLINES, abstains, mentioned_deadlines, refuses
    from .metrics import DEADLINE_UNITS
    from .norm_exam import all_deadlines

    kind = kind_of(str(row.get("id", "")))
    answer = row.get("answer") or ""
    if kind == "inesistente":
        return 1.0 if abstains(answer) and not mentioned_deadlines(answer) else 0.0
    if kind in ("termine", "termine_argomento"):
        m = _RUBRIC_DEADLINE.search(row.get("rubric") or "")
        if not m:
            return None
        wanted = (int(m.group(1)), DEADLINE_UNITS.get(m.group(2).lower(), m.group(2)))
        article = (row.get("reference") or "").split("Testo integrale dell'articolo:", 1)[-1]
        named = mentioned_deadlines(answer)
        if refuses(answer) or len(named) > MAX_DEADLINES:
            return 0.0
        return 1.0 if named & ({wanted} | all_deadlines(article)) else 0.0
    return None


@dataclass
class Graded:
    """One model's graded answers, by item id."""

    label: str
    grades: dict[str, str] = field(default_factory=dict)
    lengths: dict[str, int] = field(default_factory=dict)
    #: judge-free scores of the items that have one (see `verifiable`)
    verif: dict[str, float] = field(default_factory=dict)

    def verif_counts(self) -> tuple[int, int]:
        return int(sum(self.verif.values())), len(self.verif)

    def right(self, item_id: str, lenient: bool = False) -> bool:
        g = self.grades.get(item_id)
        return g == "correct" or (lenient and g == "partial")

    def counts(self) -> Counter:
        return Counter(self.grades.values())

    def mean_length(self) -> float:
        return sum(self.lengths.values()) / len(self.lengths) if self.lengths else 0.0


def label_of(path: Path) -> str:
    """answers-v0.3-open.graded.jsonl -> v0.3-open"""
    name = path.name
    for suffix in (".graded.jsonl", ".jsonl"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name[len("answers-"):] if name.startswith("answers-") else name


def load_graded(path: Path, exclude_rulings: frozenset[str] = frozenset()) -> Graded:
    """One model's grades; items about a ruling in ``exclude_rulings`` are left out.

    The case-law exam's items carry the ``ruling`` they were written from.
    Leaving some out compares the models on the rest only: on 2026-10-07 the
    OPD prompts of the 4B run were found to hold passages of 158 of the 1,300
    development rulings, and a gain that holds without their questions is
    not owed to having seen them.
    """
    out = Graded(label_of(path))
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if exclude_rulings and r.get("ruling") in exclude_rulings:
                continue
            out.grades[r["id"]] = r.get("grade", "unparsed")
            out.lengths[r["id"]] = len(r.get("answer") or "")
            v = verifiable(r)
            if v is not None:
                out.verif[r["id"]] = v
    return out


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b discordant pairs one way, c the other."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@dataclass
class Comparison:
    """Model ``a`` against ``base`` on the items both answered."""

    a: str
    base: str
    n: int
    a_only: int          # a right, base not
    base_only: int       # base right, a not
    p: float
    a_only_lenient: int
    base_only_lenient: int
    p_lenient: float
    by_kind: dict[str, tuple[int, int]]
    # among discordant items whose answers differ in length, share where the
    # right answer is longer (a tie is in neither count)
    longer_wins: float | None
    verif_a_only: int = 0       # judge-free: a right, base not
    verif_base_only: int = 0
    verif_n: int = 0

    @property
    def p_verif(self) -> float:
        return mcnemar_p(self.verif_a_only, self.verif_base_only)

    @property
    def diff(self) -> int:
        return self.a_only - self.base_only


def compare(a: Graded, base: Graded) -> Comparison:
    ids = sorted(set(a.grades) & set(base.grades))
    ids = [i for i in ids if "unparsed" not in (a.grades[i], base.grades[i])]

    def split(lenient: bool) -> tuple[int, int]:
        x = sum(a.right(i, lenient) and not base.right(i, lenient) for i in ids)
        y = sum(base.right(i, lenient) and not a.right(i, lenient) for i in ids)
        return x, y

    b, c = split(False)
    bl, cl = split(True)
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    longer, discordant = 0, 0
    for i in ids:
        ar, br = a.right(i), base.right(i)
        if ar == br:
            continue
        by_kind[kind_of(i)][0 if ar else 1] += 1
        la, lb = a.lengths.get(i, 0), base.lengths.get(i, 0)
        if la != lb:
            discordant += 1
            longer += (la > lb) == ar
    shared = sorted(set(a.verif) & set(base.verif))
    va = sum(a.verif[i] > base.verif[i] for i in shared)
    vb = sum(base.verif[i] > a.verif[i] for i in shared)
    return Comparison(a.label, base.label, len(ids), b, c, mcnemar_p(b, c), bl, cl,
                      mcnemar_p(bl, cl), {k: (v[0], v[1]) for k, v in sorted(by_kind.items())},
                      longer / discordant if discordant else None, va, vb, len(shared))


def human_agreement(rows: list[dict], models: dict[str, Graded]) -> dict:
    """How the judge's grades compare with a person's on the same answers.

    ``rows`` are the review sheet's lines (export_grade_review.py): ``chiave``
    is "<label>|<item id>", ``giudizio`` what the person wrote. Rows left
    blank or naming a model not given are not counted.
    """
    pairs = []
    for r in rows:
        human = HUMAN_LABELS.get(str(r.get("giudizio", "")).strip().lower())
        label, _, item = str(r.get("chiave", "")).partition("|")
        judge = models[label].grades.get(item) if label in models else None
        if human and judge and judge != "unparsed":
            pairs.append((judge, human))
    confusion = Counter(pairs)
    n = len(pairs)
    same = sum(v for (j, h), v in confusion.items() if j == h)
    binary = sum(v for (j, h), v in confusion.items() if (j == "correct") == (h == "correct"))
    return {
        "n": n,
        "same_label": same / n if n else None,
        "same_right_or_not": binary / n if n else None,
        # The two errors that move a comparison: the judge passing what a
        # person fails, and failing what a person passes.
        "judge_too_kind": confusion[("correct", "wrong")] + confusion[("correct", "partial")],
        "judge_too_harsh": confusion[("wrong", "correct")] + confusion[("partial", "correct")],
        "confusion": {f"{j}->{h}": v for (j, h), v in sorted(confusion.items())},
    }
