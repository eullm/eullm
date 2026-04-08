"""German medical corpus preparation for EULLM Forge.

Planned sources:
  - DIMDI / BfArM  : German medical coding and guidelines (ICD-10-GM)
  - G-BA           : Gemeinsamer Bundesausschuss guidelines
  - AWMF           : Clinical guidelines (Leitlinien)
  - EUR-Lex DE     : EU medical device and pharma regulations

TODO: implement when building eullm/medical-de-7b.
"""

from __future__ import annotations

from pathlib import Path


def prepare_medical_de(
    output_dir: str | Path,
    **kwargs,
) -> Path:
    """Prepare German medical corpus.

    Not yet implemented. See forge/eullm_forge/datasets/legal_it.py for
    the reference implementation to follow.
    """
    raise NotImplementedError(
        "German medical corpus preparation is not yet implemented. "
        "Planned sources: DIMDI/BfArM ICD-10-GM, G-BA guidelines, AWMF Leitlinien."
    )
