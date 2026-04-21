"""EULLM Forge — dataset preparation modules for domain corpora."""

from .anonymize import (
    AnonymiserConfig,
    RedactionStats,
    anonymize_record,
    anonymize_text,
    load_spacy_ner,
)
from .italgiure import (
    ItalgiureQuery,
    fetch_italgiure,
    load_italgiure_jsonl,
)
from .legal_it import prepare_legal_it

__all__ = [
    "AnonymiserConfig",
    "ItalgiureQuery",
    "RedactionStats",
    "anonymize_record",
    "anonymize_text",
    "fetch_italgiure",
    "load_italgiure_jsonl",
    "load_spacy_ner",
    "prepare_legal_it",
]
