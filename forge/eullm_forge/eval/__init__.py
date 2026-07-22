"""EULLM Forge — evaluation harness (F0).

A reproducible, blind way to rank verticalized models. Parametric by
(domain, language) so the same harness serves ``legal-it``, ``medical-de``
and ``finance-fr``.

Design notes
------------
- **Decoupled from any runtime.** Model access is injected as a
  ``generate_fn: (prompt) -> text`` and the judge as an interface, so nothing
  here imports ``torch``/``transformers`` at module load. Heavy/optional deps
  are imported lazily inside the functions that need them.
- **No vendor lock-in.** The LLM-as-judge and the answer generator are plain
  callables — wire them to the EULLM Engine, any OpenAI-compatible endpoint,
  or a local model.

Typical use::

    from eullm_forge.eval import load_seed, evaluate_qa, build_report

    items = load_seed()
    answers = {it.id: my_generate(it.question) for it in items}
    report = build_report(items, answers)
"""

from __future__ import annotations

from .dataset import EvalItem, filter_items, load_eval_set, load_seed, save_eval_set
from .harness import collect_answers, compare_models, evaluate_qa
from .judge import ABOutcome, Judge, Judgement, LLMJudge, MockJudge, blind_pairwise
from .metrics import (
    aggregate,
    exact_match,
    keyword_coverage,
    normalize_text,
    perplexity_from_nll,
    score_item,
)
from .report import build_report, spotcheck_markdown, to_markdown

__all__ = [
    "ABOutcome",
    "EvalItem",
    "Judge",
    "Judgement",
    "LLMJudge",
    "MockJudge",
    "aggregate",
    "blind_pairwise",
    "build_report",
    "collect_answers",
    "compare_models",
    "evaluate_qa",
    "exact_match",
    "filter_items",
    "keyword_coverage",
    "load_eval_set",
    "load_seed",
    "normalize_text",
    "perplexity_from_nll",
    "save_eval_set",
    "score_item",
    "spotcheck_markdown",
    "to_markdown",
]
