#!/usr/bin/env python3
"""Prepare Italian legislation (Normattiva codici) for training.

Reads the AKN OpenData ZIP from dati.normattiva.it, extracts articles
from the requested codes (Costituzione, Codice Civile, Codice Penale,
CPC, CPP, Codice del Consumo), runs them through the same chunker used
for italgiure, and writes the result next to the italgiure files so the
final ``format_pretraining`` step picks them up automatically.

No anonymisation needed: codici are public reference texts with no PII.
No dedup either: each article is unique (and tiny relative to italgiure
volume — a few thousand chunks vs ~1.1M).

Usage:
    # Process the bundled codes into the corpus directory:
    python forge/scripts/prepare_legislation.py \\
        ~/Scaricati/Codici_AKN_VIGENTE_2026-04-08.zip \\
        --output ~/italgiure_corpus

    # Only specific codes:
    python forge/scripts/prepare_legislation.py \\
        ~/Scaricati/Codici_AKN_VIGENTE_2026-04-08.zip \\
        --sources codice_civile codice_penale \\
        --output ~/italgiure_corpus

Output naming: ``legislazione_<source_id>.chunks.jsonl``
(e.g. ``legislazione_codice_civile.chunks.jsonl``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.chunk import ChunkConfig, chunk_text  # noqa: E402
from eullm_forge.datasets.legal_it import (  # noqa: E402
    NORMATTIVA_LAWS,
    parse_normattiva_opendata_zip,
)


def _slug_from_article(source_id: str, article_num: str | None) -> str:
    """Build a stable, human-readable source_id per article.

    Example: ('codice_civile', '2086') -> 'codice_civile/art_2086'.
    """
    if article_num:
        return f"{source_id}/art_{article_num}"
    return source_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zip_path",
        type=Path,
        help="Local AKN ZIP from dati.normattiva.it (Codici_AKN_VIGENTE_*.zip)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where legislazione_<id>.chunks.jsonl files are written.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Subset of code IDs to extract (default: all known). Choices: "
        f"{', '.join(law.id for law in NORMATTIVA_LAWS)}.",
    )
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats without writing legislazione_*.chunks.jsonl",
    )
    args = parser.parse_args(argv)

    if not args.zip_path.is_file():
        parser.error(f"{args.zip_path} not found")
    args.output.mkdir(parents=True, exist_ok=True)

    # Validate --sources against the catalogue.
    catalogue = {law.id: law for law in NORMATTIVA_LAWS}
    if args.sources:
        unknown = [s for s in args.sources if s not in catalogue]
        if unknown:
            parser.error(
                f"Unknown source(s): {', '.join(unknown)}. "
                f"Choose from: {', '.join(catalogue)}"
            )
        wanted_ids = list(args.sources)
    else:
        wanted_ids = list(catalogue)

    print(
        f"Reading {args.zip_path.name} "
        f"({args.zip_path.stat().st_size / 1024 / 1024:.1f} MB)...",
        file=sys.stderr,
    )
    zip_bytes = args.zip_path.read_bytes()

    print(
        f"Parsing AKN XML for {len(wanted_ids)} source(s): "
        f"{', '.join(wanted_ids)}",
        file=sys.stderr,
    )
    parsed = parse_normattiva_opendata_zip(zip_bytes, wanted_ids)

    chunk_cfg = ChunkConfig(
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_chars=args.min_chars,
    )

    total_articles = 0
    total_chunks = 0
    t0 = time.time()
    for source_id in wanted_ids:
        articles = parsed.get(source_id, [])
        if not articles:
            print(f"[skip] {source_id}: no articles in ZIP", file=sys.stderr)
            continue

        dst = args.output / f"legislazione_{source_id}.chunks.jsonl"
        n_chunks = 0
        out_f = None if args.dry_run else dst.open("w", encoding="utf-8")
        try:
            for art in articles:
                text = art.get("text") or ""
                if not text.strip():
                    continue
                article_num = art.get("article_num")
                source_record_id = _slug_from_article(source_id, article_num)
                chunks = chunk_text(
                    text,
                    max_chars=chunk_cfg.max_chars,
                    overlap=chunk_cfg.overlap,
                    min_chars=chunk_cfg.min_chars,
                    boundary_lookback=chunk_cfg.boundary_lookback,
                )
                total = len(chunks)
                for i, ch in enumerate(chunks):
                    record = {
                        "text": ch,
                        "source_id": source_record_id,
                        "kind": "legge",
                        "code": source_id,
                        "article_num": article_num,
                        "article_title": art.get("article_title"),
                        "chunk_index": i,
                        "chunk_total": total,
                    }
                    if out_f is not None:
                        out_f.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )
                    n_chunks += 1
        finally:
            if out_f is not None:
                out_f.close()

        total_articles += len(articles)
        total_chunks += n_chunks
        print(
            f"  {source_id}: {len(articles):,} articles → {n_chunks:,} "
            f"chunks → {dst.name}",
            file=sys.stderr,
        )

    dt = time.time() - t0
    print()
    print("=" * 72)
    print(f"Total: {total_articles:,} articles → {total_chunks:,} chunks "
          f"in {dt:.1f}s")
    print(f"Output dir: {args.output}")
    print("=" * 72)
    if args.dry_run:
        print("(dry-run — no .chunks.jsonl files written)")
    else:
        print(
            "Next: re-run format_pretraining.py on the corpus dir to mix "
            "legislation into the train/val split."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
