"""EULLM Forge — the teacher's top-K distributions, on disk.

The v1.1 distillation pipeline (ADR-001) lets one teacher pass serve any
number of student runs. This package is the record it leaves behind, and the
checks that make it trustworthy.
"""

from .fingerprint import fingerprint_tokenizer, fingerprint_vocab
from .format import (
    DEFAULT_TEMPERATURES,
    CachedSequence,
    CacheProvenance,
    LogitCacheReader,
    LogitCacheWriter,
    check_normalisers,
    from_bf16_bits,
    to_bf16_bits,
    validate_sequence,
)

__all__ = [
    "DEFAULT_TEMPERATURES",
    "CacheProvenance",
    "CachedSequence",
    "LogitCacheReader",
    "LogitCacheWriter",
    "check_normalisers",
    "fingerprint_tokenizer",
    "fingerprint_vocab",
    "from_bf16_bits",
    "to_bf16_bits",
    "validate_sequence",
]
