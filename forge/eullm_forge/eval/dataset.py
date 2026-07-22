"""Evaluation dataset — held-out items used to rank verticalized models.

Items are stored as JSONL (one JSON object per line). Each item is parametric
by ``domain`` and ``lang`` so a single harness serves every vertical. The seed
set for the Italian legal domain lives at
``data/legal_it_heldout.seed.jsonl`` and is meant to be expanded and validated
by a domain expert (F0-A in the Forge R&D roadmap).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SEED_PATH = Path(__file__).parent / "data" / "legal_it_heldout.seed.jsonl"


@dataclass
class EvalItem:
    """A single held-out evaluation item.

    Attributes:
        id: Stable unique identifier.
        domain: Vertical domain (e.g. ``"legal"``, ``"medical"``, ``"finance"``).
        lang: ISO language code (e.g. ``"it"``, ``"de"``, ``"fr"``).
        question: The prompt shown to the model.
        reference: A reference answer (optional; used by exact-match / judge).
        rubric: Scoring guidance for the LLM-as-judge (optional).
        keywords: Terms that a correct answer should contain (keyword coverage).
        category: Sub-domain tag (e.g. ``"civile"``, ``"gdpr"``).
        metadata: Free-form extra fields (source note, article, ...).
    """

    id: str
    domain: str
    lang: str
    question: str
    reference: str = ""
    rubric: str = ""
    keywords: list[str] = field(default_factory=list)
    category: str = ""
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> EvalItem:
        """Build an item from a plain dict, ignoring unknown keys."""
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_dict(self) -> dict:
        return asdict(self)


def load_eval_set(path: str | Path) -> list[EvalItem]:
    """Load a JSONL eval set. Blank lines and ``#`` comment lines are skipped."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")
    items: list[EvalItem] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not data.get("id"):
                raise ValueError(f"{path}:{lineno}: item is missing 'id'")
            try:
                item = EvalItem.from_dict(data)
            except TypeError as exc:  # missing required field (domain/lang/question)
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if item.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id '{item.id}'")
            seen.add(item.id)
            items.append(item)
    return items


def save_eval_set(items: list[EvalItem], path: str | Path) -> Path:
    """Write items to JSONL, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return path


def filter_items(
    items: list[EvalItem],
    *,
    domain: str | None = None,
    lang: str | None = None,
    category: str | None = None,
) -> list[EvalItem]:
    """Filter items by domain / language / category (case-insensitive)."""

    def keep(it: EvalItem) -> bool:
        if domain and it.domain.lower() != domain.lower():
            return False
        if lang and it.lang.lower() != lang.lower():
            return False
        if category and it.category.lower() != category.lower():
            return False
        return True

    return [it for it in items if keep(it)]


def load_seed() -> list[EvalItem]:
    """Load the bundled Italian-legal seed held-out set."""
    return load_eval_set(SEED_PATH)
