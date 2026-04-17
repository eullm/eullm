"""EULLM Forge — dataset preparation modules for domain corpora."""

from .italgiure import (
    ItalgiureQuery,
    fetch_italgiure,
    load_italgiure_jsonl,
)
from .legal_it import prepare_legal_it

__all__ = [
    "ItalgiureQuery",
    "fetch_italgiure",
    "load_italgiure_jsonl",
    "prepare_legal_it",
]
