"""Persistent legal memory grafted onto trained models (`docs/research/legal-memory-plan.md`)."""

from .ngram import (
    AttachedMemory,
    NGramMemory,
    NGramMemoryConfig,
    attach,
    decoder_layers,
    memory_config_for,
)

__all__ = ["AttachedMemory", "NGramMemory", "NGramMemoryConfig", "attach", "decoder_layers",
           "memory_config_for"]
