"""Evaluation harness — rank one model, or two models blind (A/B).

Model access is injected as ``generate_fn: (prompt) -> text`` so the harness is
decoupled from any specific runtime (EULLM Engine, HF, an OpenAI-compatible
endpoint). Nothing here imports ``torch``.
"""

from __future__ import annotations

from typing import Callable

from .dataset import EvalItem
from .judge import ABOutcome, Judge, blind_pairwise
from .metrics import QAResult, aggregate, score_item

GenerateFn = Callable[[str], str]


def collect_answers(items: list[EvalItem], generate_fn: GenerateFn) -> dict[str, str]:
    """Run ``generate_fn`` over every item's question, keyed by item id."""
    return {item.id: generate_fn(item.question) for item in items}


def evaluate_qa(items: list[EvalItem], answers: dict[str, str]) -> dict:
    """Score answers against items with the QA metrics.

    Items with no answer are scored on the empty string (a miss), so the
    denominator is always the full item count. Returns a dict with per-item
    results and an aggregate summary.
    """
    results: list[QAResult] = [score_item(answers.get(it.id, ""), it) for it in items]
    return {
        "summary": aggregate(results),
        "per_item": [
            {"id": r.id, "exact": r.exact, "keyword_coverage": r.keyword_coverage}
            for r in results
        ],
    }


def compare_models(
    items: list[EvalItem],
    answers_a: dict[str, str],
    answers_b: dict[str, str],
    judge: Judge,
    *,
    seed: int = 0,
) -> ABOutcome:
    """Blind A/B comparison of two models' answers using ``judge``."""
    return blind_pairwise(items, answers_a, answers_b, judge, seed=seed)
