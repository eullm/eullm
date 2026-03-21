"""Verticalizzazione profiles for European domain/language combinations.

Each profile is a YAML file that defines the full compression pipeline
for a specific domain and language. Profiles include:

- legal_it: Italian legal domain (civil code, GDPR, Cassazione)
- medical_de: German medical domain (clinical guidelines, medical docs)
- finance_fr: French finance domain (AMF regulations, BCE directives)
"""

from __future__ import annotations

from pathlib import Path

PROFILES_DIR = Path(__file__).parent


def list_profiles() -> list[str]:
    """List available profile names."""
    return [p.stem for p in sorted(PROFILES_DIR.glob("*.yaml"))]
