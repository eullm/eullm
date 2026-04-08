"""French finance corpus preparation for EULLM Forge.

Planned sources:
  - Légifrance      : French financial legislation (Code monétaire et financier)
  - AMF             : Autorité des marchés financiers regulations
  - BCE / ECB       : European Central Bank publications in French
  - EUR-Lex FR      : EU financial services regulations

TODO: implement when building eullm/finance-fr-7b.
"""

from __future__ import annotations

from pathlib import Path


def prepare_finance_fr(
    output_dir: str | Path,
    **kwargs,
) -> Path:
    """Prepare French finance corpus.

    Not yet implemented. See forge/eullm_forge/datasets/legal_it.py for
    the reference implementation to follow.
    """
    raise NotImplementedError(
        "French finance corpus preparation is not yet implemented. "
        "Planned sources: Légifrance Code monétaire, AMF regulations, BCE publications."
    )
