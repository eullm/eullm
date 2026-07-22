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
    """Fraction of required keywords present in the prediction (0..1).

    Returns ``1.0`` when there are no required keywords (nothing to miss).
    Matching is done on normalized text so accents/case/punctuation are ignored.
    """
    if not keywords:
        return 1.0
    norm_pred = normalize_text(prediction)
    hits = sum(1 for kw in keywords if normalize_text(kw) in norm_pred)
    return hits / len(keywords)


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
