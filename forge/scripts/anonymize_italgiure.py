#!/usr/bin/env python3
"""Anonymise a downloaded italgiure corpus in place, JSONL → .anon.jsonl.

Streams each ``italgiure_*.jsonl`` under the input directory and writes a
parallel ``italgiure_*.anon.jsonl`` with personal data redacted (see
``eullm_forge.datasets.anonymize`` for the rules).

The process is resumable — a ``_anon_progress.json`` file tracks how many
records have been processed per slice, so re-runs pick up where they left
off. Output files are only truncated on the first write.

Usage:
    # Default run — regex layers + spaCy NER for person names. Needs the
    # Italian model (pip install 'eullm-forge[legal]', or
    # python -m spacy download it_core_news_lg); the script exits non-zero
    # rather than proceeding without it.
    python forge/scripts/anonymize_italgiure.py ~/italgiure_corpus

    # Dry run on a sample:
    python forge/scripts/anonymize_italgiure.py ~/italgiure_corpus \\
        --sample 20 --dry-run

    # Regex-only, no spaCy. Only fully-uppercase names get redacted —
    # Title-Case names in the body of the reasoning remain in the output.
    python forge/scripts/anonymize_italgiure.py ~/italgiure_corpus --no-ner
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.anonymize import (  # noqa: E402
    AnonymiserConfig,
    anonymize_record,
    load_spacy_ner,
)

PROGRESS_FILE = "_anon_progress.json"


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield lineno, json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] {path.name}:{lineno} malformed JSON: {exc}",
                    file=sys.stderr,
                )


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def anonymise_slice(
    src: Path,
    dst: Path,
    *,
    config: AnonymiserConfig,
    ner: Optional[Callable] = None,
    resume_from: int = 0,
    sample: int = 0,
    dry_run: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> tuple[int, Counter]:
    """Anonymise one JSONL slice.

    Args:
        src: Input italgiure_*.jsonl.
        dst: Output italgiure_*.anon.jsonl (ignored if dry_run).
        config: AnonymiserConfig.
        ner: Optional NER callable.
        resume_from: Skip the first N records (resumption).
        sample: If > 0, stop after this many records.
        dry_run: If True, don't write output — still computes stats.
        progress_cb: Optional callback(processed, total_in_file) for progress.

    Returns:
        ``(records_written, category_counts)``.
    """
    written = 0
    counts: Counter = Counter()

    mode = "w" if resume_from == 0 and not dry_run else "a"
    out_f = None if dry_run else dst.open(mode, encoding="utf-8")
    try:
        for idx, (_lineno, rec) in enumerate(_iter_jsonl(src)):
            if idx < resume_from:
                continue
            out_rec = anonymize_record(rec, config=config, ner=ner)
            audit = out_rec.get("metadata", {}).get("anonymization", {})
            for k, v in audit.items():
                counts[k] += v
            if not dry_run and out_f is not None:
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            written += 1
            if progress_cb and written % 500 == 0:
                progress_cb(resume_from + written, resume_from + written)
            if sample and written >= sample:
                break
    finally:
        if out_f is not None:
            out_f.close()
    return written, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path, help="Directory with italgiure_*.jsonl")
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help=(
            "Disable the spaCy NER layer. NOT RECOMMENDED: without it only "
            "fully-uppercase names are redacted, and Title-Case names in the "
            "body of the reasoning pass through in clear text."
        ),
    )
    parser.add_argument(
        "--ner-model",
        default="it_core_news_lg",
        help="spaCy model name (default: it_core_news_lg)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Process only the first N records of each slice (dry-run friendly)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats without writing output files",
    )
    parser.add_argument(
        "--no-allcaps",
        action="store_true",
        help="Disable the all-caps name heuristic",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoint and re-anonymise all slices from scratch",
    )
    args = parser.parse_args(argv)

    corpus = args.corpus_dir
    if not corpus.is_dir():
        parser.error(f"{corpus} is not a directory")

    sources = sorted(corpus.glob("italgiure_*.jsonl"))
    # Skip already-anonymised files (in case of *.anon.jsonl naming).
    sources = [p for p in sources if not p.name.endswith(".anon.jsonl")]
    if not sources:
        parser.error(f"No italgiure_*.jsonl files under {corpus}")

    use_ner = not args.no_ner
    config = AnonymiserConfig(
        use_ner=use_ner,
        redact_allcaps_names=not args.no_allcaps,
    )

    ner = None
    if use_ner:
        print(f"Loading spaCy NER model {args.ner_model}...", file=sys.stderr)
        try:
            ner = load_spacy_ner(args.ner_model)
        except RuntimeError as exc:
            # Hard failure, never a silent downgrade: continuing without NER
            # would emit a corpus the caller believes is redacted while
            # Title-Case names pass through untouched.
            print(f"ERROR: {exc}", file=sys.stderr)
            print(
                "Install it with: pip install 'eullm-forge[legal]'\n"
                "Or, if you accept that only fully-uppercase names will be "
                "redacted, re-run with --no-ner.",
                file=sys.stderr,
            )
            return 2
    else:
        print(
            "WARNING: --no-ner — only fully-uppercase names will be redacted. "
            "Title-Case names in the body of the reasoning will remain in the "
            "output. Do not treat this corpus as redacted for personal names.",
            file=sys.stderr,
        )

    progress_path = corpus / PROGRESS_FILE
    progress: dict[str, int] = {}
    if progress_path.exists() and not args.force and not args.sample:
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] progress file corrupt: {exc}", file=sys.stderr)

    total_counts: Counter = Counter()
    total_written = 0
    t0 = time.time()

    for src in sources:
        slug = src.stem.replace("italgiure_", "")
        dst = src.with_name(f"italgiure_{slug}.anon.jsonl")
        total_in_file = _count_lines(src)
        resume_from = progress.get(slug, 0) if not args.sample else 0

        if resume_from >= total_in_file and not args.sample:
            print(f"[skip] {src.name} already complete ({total_in_file} records)")
            continue

        print(
            f"[work] {src.name} → {dst.name} "
            f"(from {resume_from}/{total_in_file})",
            file=sys.stderr,
        )

        t_slice = time.time()
        written, counts = anonymise_slice(
            src,
            dst,
            config=config,
            ner=ner,
            resume_from=resume_from,
            sample=args.sample,
            dry_run=args.dry_run,
        )
        total_counts.update(counts)
        total_written += written

        if not args.dry_run and not args.sample:
            progress[slug] = resume_from + written
            progress_path.write_text(
                json.dumps(progress, indent=2), encoding="utf-8"
            )

        dt = time.time() - t_slice
        rate = written / dt if dt > 0 else 0.0
        print(
            f"  wrote {written:,} records in {dt:.1f}s ({rate:.1f} rec/s); "
            f"redactions in slice: {sum(counts.values()):,}",
            file=sys.stderr,
        )

    dt_all = time.time() - t0
    print()
    print("=" * 64)
    print(f"Total records processed: {total_written:,} in {dt_all:.1f}s")
    print(f"Total redactions:        {sum(total_counts.values()):,}")
    print("By category:")
    for k, v in sorted(total_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>20}: {v:>10,}")
    print("=" * 64)
    if args.dry_run:
        print("(dry-run — no output files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
