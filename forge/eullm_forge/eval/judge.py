"""LLM-as-judge with blind, position-bias-controlled A/B pairing.

The judge is an *interface* (no vendor lock-in): supply any callable that maps
a prompt to text — the EULLM Engine, an OpenAI-compatible endpoint, or a local
model. :class:`MockJudge` is deterministic and dependency-free, for tests and
dry runs.

Blinding: for each item the two answers are shown to the judge in a **random
order** (seeded), and the verdict is mapped back to the true A/B identities, so
the aggregate win-rate is not contaminated by the judge's position bias.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .metrics import keyword_coverage, normalize_text


@dataclass
class Judgement:
    """The judge's verdict for one pairwise comparison."""

    winner: str  # "A" | "B" | "tie"
    rationale: str = ""


class Judge(Protocol):
    """Anything that can compare two answers to a question."""

    def compare(
        self, question: str, answer_a: str, answer_b: str, rubric: str = ""
    ) -> Judgement: ...


class MockJudge:
    """Deterministic judge for tests/dry-runs.

    Prefers the answer whose keyword overlap with the rubric is higher; falls
    back to answer length; ties when equal. No network, no model.
    """

    def compare(
        self, question: str, answer_a: str, answer_b: str, rubric: str = ""
    ) -> Judgement:
        keys = [w for w in normalize_text(rubric).split() if len(w) > 3]
        score_a = keyword_coverage(answer_a, keys) if keys else 0.0
        score_b = keyword_coverage(answer_b, keys) if keys else 0.0
        if score_a == score_b:
            score_a, score_b = len(answer_a), len(answer_b)
        if score_a > score_b:
            return Judgement("A", "mock: higher rubric/length score")
        if score_b > score_a:
            return Judgement("B", "mock: higher rubric/length score")
        return Judgement("tie", "mock: equal")


_JUDGE_TEMPLATE = """You are an impartial expert evaluator.
Compare two answers to the same question and decide which is better.
Judge only on correctness, completeness and relevance — ignore length and style.

Question:
{question}

{rubric_block}Answer 1:
{first}

Answer 2:
{second}

Respond with exactly one line: "Verdict: 1", "Verdict: 2", or "Verdict: tie".
Then, on a new line, a one-sentence justification.
"""

_VERDICT_RE = re.compile(r"verdict\s*[:=]?\s*(1|2|tie|a|b)", re.IGNORECASE)


class LLMJudge:
    """LLM-as-judge backed by an injected ``chat_fn: (prompt) -> text``.

    ``chat_fn`` is deliberately generic (the interface is the seam, not a
    vendor): wire it to the EULLM Engine or any OpenAI-compatible endpoint.
    """

    def __init__(self, chat_fn: Callable[[str], str], name: str = "llm-judge") -> None:
        self.chat_fn = chat_fn
        self.name = name

    def compare(
        self, question: str, answer_a: str, answer_b: str, rubric: str = ""
    ) -> Judgement:
        rubric_block = f"Scoring rubric:\n{rubric}\n\n" if rubric else ""
        prompt = _JUDGE_TEMPLATE.format(
            question=question, rubric_block=rubric_block, first=answer_a, second=answer_b
        )
        raw = self.chat_fn(prompt)
        match = _VERDICT_RE.search(raw or "")
        if not match:
            return Judgement("tie", f"unparseable verdict: {raw!r}")
        token = match.group(1).lower()
        winner = {"1": "A", "a": "A", "2": "B", "b": "B", "tie": "tie"}[token]
        return Judgement(winner, (raw or "").strip())


@dataclass
class ABOutcome:
    """Aggregated result of a blind A/B comparison."""

    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0

    @property
    def n(self) -> int:
        return self.a_wins + self.b_wins + self.ties

    @property
    def win_rate_a(self) -> float:
        decided = self.a_wins + self.b_wins
        return self.a_wins / decided if decided else float("nan")

    def to_dict(self) -> dict:
        return {
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "ties": self.ties,
            "n": self.n,
            "win_rate_a": self.win_rate_a,
        }


def blind_pairwise(
    items,
    answers_a: dict[str, str],
    answers_b: dict[str, str],
    judge: Judge,
    *,
    seed: int = 0,
) -> ABOutcome:
    """Blind pairwise A/B over the items answered by both models.

    For each item the two answers are presented to the judge in a random
    (seeded) order; the verdict is then de-blinded back to A/B. Items missing
    from either answer set are skipped.
    """
    rng = random.Random(seed)
    outcome = ABOutcome()
    for item in items:
        if item.id not in answers_a or item.id not in answers_b:
            continue
        ans_a = answers_a[item.id]
        ans_b = answers_b[item.id]
        swap = rng.random() < 0.5
        first, second = (ans_b, ans_a) if swap else (ans_a, ans_b)
        verdict = judge.compare(item.question, first, second, item.rubric)
        winner = verdict.winner
        if winner == "tie":
            outcome.ties += 1
            continue
        # de-blind: "A" means "first shown"; map back through the swap
        first_is_a = not swap
        a_won = (winner == "A") == first_is_a
        if a_won:
            outcome.a_wins += 1
        else:
            outcome.b_wins += 1
    return outcome
