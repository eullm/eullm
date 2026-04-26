#!/usr/bin/env python3
"""Prepare Italian legislation (Normattiva codici + standalone laws) for training.

Two input modes:

    *.zip   AKN OpenData bundle from dati.normattiva.it (Codici_AKN_VIGENTE_*.zip).
            Multiple codes parsed in one pass, source_id taken from FRBRthis.

    *.xml   Single AKN XML file (e.g. Costituzione, downloaded one-shot from
            normattiva.it). Requires --source-id to name the output file.

Articles are then run through the same chunker used for italgiure and
written next to the italgiure files so the final ``format_pretraining``
step picks them up automatically.

No anonymisation needed: codici and leggi are public reference texts
with no PII. No dedup either: each article is unique.

Usage — bulk ZIP:
    python forge/scripts/prepare_legislation.py \\
        ~/Scaricati/Codici_AKN_VIGENTE_2026-04-08.zip \\
        --output ~/italgiure_corpus

Usage — single XML (Costituzione):
    python forge/scripts/prepare_legislation.py \\
        ~/Scaricati/19471227_047U0001_VIGENZA_20231022.xml \\
        --source-id costituzione \\
        --output ~/italgiure_corpus

Output naming: ``legislazione_<source_id>.chunks.jsonl``
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
    _parse_akn_xml,
    parse_normattiva_opendata_zip,
)


def _slug_from_article(source_id: str, article_num: str | None) -> str:
    """Build a stable, human-readable source_id per article.

    Example: ('codice_civile', '2086') -> 'codice_civile/art_2086'.
    """
    if article_num:
        return f"{source_id}/art_{article_num}"
    return source_id


def _write_source(
    source_id: str,
    articles: list[dict],
    output_dir: Path,
    chunk_cfg: ChunkConfig,
    dry_run: bool,
) -> int:
    """Chunk and write articles for one source. Returns the chunk count."""
    if not articles:
        print(f"[skip] {source_id}: no articles", file=sys.stderr)
        return 0
    dst = output_dir / f"legislazione_{source_id}.chunks.jsonl"
    n_chunks = 0
    out_f = None if dry_run else dst.open("w", encoding="utf-8")
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
    print(
        f"  {source_id}: {len(articles):,} articles → {n_chunks:,} "
        f"chunks → {dst.name}",
        file=sys.stderr,
    )
    return n_chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="AKN ZIP from dati.normattiva.it OR a single AKN XML file.",
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
        help="ZIP mode only — subset of code IDs to extract (default: all known). "
        f"Choices: {', '.join(law.id for law in NORMATTIVA_LAWS)}.",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="XML mode — required, names the output file (e.g. 'costituzione').",
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

    if not args.input_path.is_file():
        parser.error(f"{args.input_path} not found")
    args.output.mkdir(parents=True, exist_ok=True)

    chunk_cfg = ChunkConfig(
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_chars=args.min_chars,
    )

    suffix = args.input_path.suffix.lower()
    total_articles = 0
    total_chunks = 0
    t0 = time.time()

    if suffix == ".zip":
        # --- ZIP mode: bulk parse via the OpenData parser -------------------
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
            f"Reading {args.input_path.name} "
            f"({args.input_path.stat().st_size / 1024 / 1024:.1f} MB)...",
            file=sys.stderr,
        )
        zip_bytes = args.input_path.read_bytes()
        print(
            f"Parsing AKN XML for {len(wanted_ids)} source(s): "
            f"{', '.join(wanted_ids)}",
            file=sys.stderr,
        )
        parsed = parse_normattiva_opendata_zip(zip_bytes, wanted_ids)
        for source_id in wanted_ids:
            articles = parsed.get(source_id, [])
            n = _write_source(source_id, articles, args.output, chunk_cfg, args.dry_run)
            total_articles += len(articles)
            total_chunks += n

    elif suffix == ".xml":
        # --- XML mode: single-file parse ------------------------------------
        if not args.source_id:
            parser.error(
                "--source-id is required when input is a single XML file "
                "(used to name the output legislazione_<source-id>.chunks.jsonl)."
            )
        if args.sources:
            print(
                "[warn] --sources is ignored in XML mode "
                "(use --source-id instead)",
                file=sys.stderr,
            )
        print(
            f"Reading {args.input_path.name} "
            f"({args.input_path.stat().st_size / 1024:.1f} KB)...",
            file=sys.stderr,
        )
        xml_text = args.input_path.read_text(encoding="utf-8")
        print(f"Parsing AKN XML as source '{args.source_id}'...", file=sys.stderr)
        articles = _parse_akn_xml(xml_text, args.source_id)
        if not articles:
            print(
                f"[error] No articles extracted from {args.input_path.name}. "
                f"Check that the XML is valid AKN with <article> or <doc> "
                f"elements.",
                file=sys.stderr,
            )
            return 1
        n = _write_source(
            args.source_id, articles, args.output, chunk_cfg, args.dry_run,
        )
        total_articles += len(articles)
        total_chunks += n

    else:
        parser.error(
            f"Unsupported input extension {suffix!r}. "
            "Use a .zip (bulk OpenData) or .xml (single law)."
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
