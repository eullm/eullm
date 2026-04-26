#!/usr/bin/env python3
"""Fetch Cassazione sentences from italgiure.giustizia.it.

Streams one JSONL slice per ``(kind, year)`` under the output directory.
Resumable via a ``_progress.json`` checkpoint, so it is safe to interrupt
and re-run, or to launch alongside an in-progress ``anonymize_italgiure``
run on the same directory (the anon script enumerates its inputs at
startup and ignores new arrivals during execution).

Usage:
    # Default coverage (civil + 2021..2026):
    python forge/scripts/fetch_italgiure.py ~/italgiure_corpus

    # Just the missing years on top of an existing corpus:
    python forge/scripts/fetch_italgiure.py ~/italgiure_corpus \\
        --years 2024 2025 2026

    # Both civil and criminal, single section:
    python forge/scripts/fetch_italgiure.py ~/italgiure_corpus \\
        --kinds snciv snpen --sezione 5

    # Disable TLS verification (only if your CA bundle is broken):
    python forge/scripts/fetch_italgiure.py ~/italgiure_corpus --no-verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.italgiure import (  # noqa: E402
    DEFAULT_KINDS,
    fetch_italgiure,
)


def _parse_years(items: list[str]) -> list[int]:
    """Accept '2021 2022 2023' or '2021-2024' or a mix."""
    years: list[int] = []
    for item in items:
        if "-" in item:
            lo, hi = item.split("-", 1)
            years.extend(range(int(lo), int(hi) + 1))
        else:
            years.append(int(item))
    return sorted(set(years))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where italgiure_<kind>_<year>.jsonl files are written.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2021-2026"],
        help="Years to fetch. Either single ints or ranges (e.g. 2024-2026). "
        "Default: 2021-2026.",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=list(DEFAULT_KINDS),
        help="Italgiure 'kinds' to fetch (snciv = civile, snpen = penale). "
        f"Default: {' '.join(DEFAULT_KINDS)}.",
    )
    parser.add_argument(
        "--sezione",
        type=int,
        default=None,
        help="Restrict to a single section number (1..7, 0=UNITE, 9=LAVORO).",
    )
    parser.add_argument(
        "--max-docs-per-query",
        type=int,
        default=None,
        help="Safety cap per (kind, year) slice — useful for tests.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.5,
        help="Seconds to wait between successive Solr calls (default: 1.5).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable TLS verification (insecure, only as last resort).",
    )
    args = parser.parse_args(argv)

    years = _parse_years(args.years)
    print(
        f"Fetching kinds={args.kinds} years={years} "
        f"sezione={args.sezione} into {args.output_dir}",
        file=sys.stderr,
    )
    fetch_italgiure(
        output_dir=args.output_dir,
        years=years,
        kinds=args.kinds,
        sezione=args.sezione,
        max_docs_per_query=args.max_docs_per_query,
        rate_limit_sec=args.rate_limit,
        verify=not args.no_verify,
    )
    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
