"""Whether an answer cites the article its question is about.

On the practical exam about half of every model's wrong answers cite an
article outside the question's (the neighbouring number, or the same number
of another code among the retrieved texts), against 13-22% of its right ones
(citation_check.py, 27-10-2026). The judge grades the answer as a whole;
this reward pays, apart, for citing the question's own article, read with
its code the way citation_check.py reads it. It is weighted below the
judge's grade (grpo_train.py --citation-weight): citing the article is
necessary, not sufficient, and an answer that cites it without applying it
still loses the judge's point.
"""

from __future__ import annotations

from ..eval.abstain import _CITE, _MORE, _key
from ..eval.retrieval import named_code
from .rewards import JUDGED_TYPES, _text


def citations(text: str, default_code: str | None) -> set[tuple[str | None, str]]:
    """(code, article) of every article the text cites.

    The code is the one named right after the article ("art. 1120 del codice
    civile", "art. 32 c.p.a."), else the last one named, else ``default_code``.
    """
    out, code = set(), default_code
    for m in _CITE.finditer(text or ""):
        code = named_code(text[m.end():m.end() + 60]) or code
        out.add((code, _key(m.group(1), m.group(2))))
        out.update((code, _key(n, s)) for n, s in _MORE.findall(m.group(3)))
    return out


def cites(answer: str, code: str, article: str) -> bool:
    """Whether the answer cites ``article`` of ``code``."""
    return (code, str(article)) in citations(answer, code)


def citation_reward(completions, tipo, code, articolo, **_) -> list[float | None]:
    """TRL reward function: 1 when a judged answer cites its question's article,
    0 when it does not; None for the other kinds and for rows naming none."""
    return [(1.0 if cites(_text(c), cd, str(a)) else 0.0)
            if t in JUDGED_TYPES and cd and str(a or "") else None
            for c, t, cd, a in zip(completions, tipo, code, articolo)]
