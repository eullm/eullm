"""Metrics for the eval harness.

Two families:

- **QA metrics** (pure Python, no heavy deps): normalized exact-match and
  keyword coverage. Coarse but deterministic and cheap.
- **Perplexity** on held-out text: lazily imports ``torch``/``transformers``.
  A pure helper, :func:`perplexity_from_nll`, is testable without a model.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .dataset import EvalItem

_WS = re.compile(r"\s+")

# Deadlines, the one place they are read: "<number> <unit>", the number in
# digits or in words, the unit to its plural. The exam builds its questions and
# keywords with them (norm_exam), keyword_coverage and the GRPO reward
# (rl/rewards.py) ask whether an answer names a deadline, as a number, and the
# paired score does too (paired.py). Here because every one of them can import
# this module, and norm_exam imports it already: kept in norm_exam, the tables
# could not be imported here without a cycle, and they were copied instead.

#: Each unit a deadline is counted in, singular or plural, to its plural.
DEADLINE_UNITS = {"giorni": "giorni", "giorno": "giorni", "mesi": "mesi", "mese": "mesi",
                  "anni": "anni", "anno": "anni", "ore": "ore", "ora": "ore"}
#: The number words the exam can produce. Digits need no table.
NUMBER_WORDS = {"un": 1, "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5,
                "sei": 6, "sette": 7, "otto": 8, "nove": 9, "dieci": 10, "undici": 11,
                "dodici": 12, "quindici": 15, "venti": 20, "ventiquattro": 24, "trenta": 30,
                "quaranta": 40, "quarantacinque": 45, "cinquanta": 50, "sessanta": 60,
                "settanta": 70, "novanta": 90, "centoventi": 120, "centocinquanta": 150,
                "centottanta": 180, "trecentosessantacinque": 365}
#: Any "<number> <unit>", whatever precedes it, in an article or an answer as
#: written. Without the singular "ora": what an article states and an answer
#: names are counted the way they always were, so exams built and answers
#: scored before stay comparable.
ANY_DEADLINE = re.compile(r"\b(\d+|[a-zà-ù]+)\s+(giorni|giorno|mesi|mese|anni|anno|ore)\b",
                          re.IGNORECASE)
# The same with "ora", for a keyword and the normalised answer it is matched
# against: a one-hour deadline's keyword is "1 ora|un ora", and "un'ora"
# normalises to "un ora".
_KEYWORD_DEADLINE = re.compile(
    r"\b(\d+|[a-zà-ù]+)\s+(giorni|giorno|mesi|mese|anni|anno|ore|ora)\b", re.IGNORECASE)


def number_of(token: str) -> int | None:
    """The number a deadline token carries, in digits or in words."""
    return int(token) if token.isdigit() else NUMBER_WORDS.get(token.lower())


def _keyword_deadline(keyword: str) -> tuple[int, str] | None:
    """The deadline a keyword from the exam stands for, or None.

    The keyword is built by `norm_exam._deadline_keyword` from the article's
    own number and unit ("20 giorni|venti giorni"), so it is read back with
    the same pattern used on an answer. None means the keyword is not a
    deadline in this shape, and the caller falls back to the substring test
    rather than assuming a number.
    """
    for num, unit in _KEYWORD_DEADLINE.findall(keyword):
        n = number_of(num)
        if n:
            return n, DEADLINE_UNITS[unit.lower()]
    return None


def _names_deadline(text: str, deadline: tuple[int, str]) -> bool:
    """Whether `text` names exactly that deadline, as a number."""
    wanted_n, wanted_unit = deadline
    for num, unit in _KEYWORD_DEADLINE.findall(text):
        n = number_of(num)
        if n and n == wanted_n and DEADLINE_UNITS[unit.lower()] == wanted_unit:
            return True
    return False


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = without_accents.lower()
    depunct = _PUNCT.sub(" ", lowered)
    return _WS.sub(" ", depunct).strip()


def exact_match(prediction: str, reference: str) -> bool:
    """Normalized exact-match between a prediction and a reference answer."""
    if not reference:
        return False
    return normalize_text(prediction) == normalize_text(reference)


def keyword_coverage(prediction: str, keywords: list[str]) -> float:
    """Fraction of required keyword groups present in the prediction (0..1).

    Returns ``1.0`` when there are no required keywords (nothing to miss).
    Matching is done on normalized text so accents/case/punctuation are ignored.

    A group may offer alternatives separated by ``|``: ``"120|centoventi"`` is
    one requirement, satisfied by either spelling. Without it the only way to
    say "either" is to list both, which makes a correct answer score half for
    using the other spelling — and an unwinnable item for whoever curated it.
    """
    if not keywords:
        return 1.0
    norm_pred = normalize_text(prediction)
    # A keyword that is a deadline is matched as a number, not as a
    # substring: "20 giorni" is inside "120 giorni" and "sessanta giorni"
    # inside "centosessanta giorni", so a deadline six times too long used to
    # satisfy the exam's own requirement. The GRPO reward compares the number
    # for exactly this reason (rl/rewards.py); the headline number here and
    # the reward were disagreeing about the same answer.
    groups = [[normalize_text(alt) for alt in kw.split("|") if alt.strip()] for kw in keywords]
    hits = 0
    for alts in groups:
        deadlines = [_keyword_deadline(alt) for alt in alts]
        if alts and all(d is not None for d in deadlines):
            # The deadline has to be named in the answer, as that number.
            hits += any(_names_deadline(norm_pred, d) for d in deadlines if d is not None)
        else:
            hits += any(alt in norm_pred for alt in alts)
    return hits / len(groups)


@dataclass
class QAResult:
    """Per-item QA score."""

    id: str
    exact: bool
    keyword_coverage: float


def score_item(prediction: str, item: "EvalItem") -> QAResult:
    """Score a single prediction against an eval item."""
    return QAResult(
        id=item.id,
        exact=exact_match(prediction, item.reference),
        keyword_coverage=keyword_coverage(prediction, item.keywords),
    )


def aggregate(results: list[QAResult]) -> dict:
    """Aggregate per-item QA results into summary statistics."""
    n = len(results)
    if n == 0:
        return {"n": 0, "exact_match": float("nan"), "keyword_coverage": float("nan")}
    return {
        "n": n,
        "exact_match": sum(1 for r in results if r.exact) / n,
        "keyword_coverage": sum(r.keyword_coverage for r in results) / n,
    }


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    """Perplexity from a summed negative log-likelihood and a token count.

    ``perplexity = exp(mean NLL)``. Returns ``nan`` for an empty count.
    """
    if total_tokens <= 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


def perplexity(
    texts: list[str],
    *,
    model_name: str | None = None,
    model=None,
    tokenizer=None,
    max_length: int = 2048,
    device: str | None = None,
) -> float:
    """Token-level perplexity of ``model`` over ``texts`` (held-out corpus).

    Lazily imports ``torch``/``transformers``. Either pass a loaded
    ``model``+``tokenizer`` or a ``model_name`` to load from the Hub.
    """
    try:  # lazy heavy import
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "perplexity() needs torch + transformers; install the ML extras."
        ) from exc

    if model is None or tokenizer is None:
        if not model_name:
            raise ValueError("Provide either (model, tokenizer) or model_name.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        if input_ids.size(1) < 2:
            continue
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        # HF returns mean NLL over (n_tokens - 1) shifted tokens
        n = input_ids.size(1) - 1
        total_nll += float(out.loss) * n
        total_tokens += n
    return perplexity_from_nll(total_nll, total_tokens)
