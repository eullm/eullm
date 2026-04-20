#!/usr/bin/env python3
"""Validate a downloaded italgiure SentenzeWeb corpus and print statistics.

Streams every ``italgiure_*.jsonl`` file under the given directory, checks
record shape, and prints aggregate statistics. Designed to be run against a
full corpus (hundreds of thousands of records) without loading it into
memory.

Usage:
    python forge/scripts/validate_italgiure_corpus.py ~/italgiure_corpus
    python forge/scripts/validate_italgiure_corpus.py ~/italgiure_corpus --sample 5
    python forge/scripts/validate_italgiure_corpus.py ~/italgiure_corpus --json stats.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

REQUIRED_TOP_FIELDS = (
    "text",
    "source",
    "sentence_id",
    "article_num",
    "article_title",
    "url",
    "metadata",
)
REQUIRED_META_FIELDS = (
    "ecli",
    "kind",
    "sezione",
    "anno",
    "tipoprov",
    "datdec",
    "datdep",
    "presidente",
    "relatore",
)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] {path.name}:{lineno} — malformed JSON: {exc}",
                    file=sys.stderr,
                )


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    k = max(0, min(len(values) - 1, int(round(pct / 100 * (len(values) - 1)))))
    return sorted(values)[k]


def _validate_record(rec: dict) -> list[str]:
    issues: list[str] = []
    for field in REQUIRED_TOP_FIELDS:
        if field not in rec:
            issues.append(f"missing top-level field: {field}")
    meta = rec.get("metadata") or {}
    for field in REQUIRED_META_FIELDS:
        if field not in meta:
            issues.append(f"missing metadata field: {field}")
    if not rec.get("text"):
        issues.append("empty text")
    if rec.get("source") != "italgiure":
        issues.append(f"unexpected source: {rec.get('source')!r}")
    return issues


def validate_corpus(
    corpus_dir: Path,
    *,
    sample_per_slice: int = 0,
) -> dict[str, Any]:
    files = sorted(corpus_dir.glob("italgiure_*.jsonl"))
    if not files:
        raise SystemExit(f"No italgiure_*.jsonl files found under {corpus_dir}")

    stats: dict[str, Any] = {
        "corpus_dir": str(corpus_dir),
        "files": [],
        "total_records": 0,
        "total_bytes": 0,
        "total_chars": 0,
        "by_kind": Counter(),
        "by_year": Counter(),
        "by_kind_year": defaultdict(int),
        "by_sezione": Counter(),
        "by_tipoprov": Counter(),
        "text_lengths": [],
        "issues": Counter(),
        "malformed_records": 0,
        "duplicate_sentence_ids": 0,
        "samples": [],
    }

    seen_ids: set[str] = set()

    for path in files:
        file_stats = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "records": 0,
            "kind": None,
            "year": None,
        }
        stats["total_bytes"] += file_stats["bytes"]

        slice_samples: list[dict] = []
        for rec in _iter_jsonl(path):
            issues = _validate_record(rec)
            if issues:
                stats["malformed_records"] += 1
                for issue in issues:
                    stats["issues"][issue] += 1
                continue

            meta = rec.get("metadata") or {}
            kind = meta.get("kind", "?")
            year = meta.get("anno", "?")
            sezione = meta.get("sezione", "?")
            tipoprov = meta.get("tipoprov", "?")
            text = rec.get("text", "")

            if file_stats["kind"] is None:
                file_stats["kind"] = kind
                file_stats["year"] = year

            stats["total_records"] += 1
            stats["total_chars"] += len(text)
            stats["by_kind"][kind] += 1
            stats["by_year"][str(year)] += 1
            stats["by_kind_year"][f"{kind}_{year}"] += 1
            stats["by_sezione"][f"{kind}/sez{sezione}"] += 1
            stats["by_tipoprov"][tipoprov] += 1
            stats["text_lengths"].append(len(text))
            file_stats["records"] += 1

            sid = rec.get("sentence_id")
            if sid:
                if sid in seen_ids:
                    stats["duplicate_sentence_ids"] += 1
                else:
                    seen_ids.add(sid)

            if sample_per_slice and len(slice_samples) < sample_per_slice:
                slice_samples.append(
                    {
                        "sentence_id": sid,
                        "article_title": rec.get("article_title"),
                        "text_preview": text[:200].replace("\n", " "),
                        "text_len": len(text),
                    }
                )

        stats["files"].append(file_stats)
        if slice_samples:
            stats["samples"].append({"file": path.name, "records": slice_samples})

    lengths = stats["text_lengths"]
    if lengths:
        stats["length_stats"] = {
            "min": min(lengths),
            "p10": _percentile(lengths, 10),
            "p50": _percentile(lengths, 50),
            "p90": _percentile(lengths, 90),
            "p99": _percentile(lengths, 99),
            "max": max(lengths),
            "mean": int(statistics.fmean(lengths)),
        }
    else:
        stats["length_stats"] = {}
    del stats["text_lengths"]

    stats["by_kind"] = dict(stats["by_kind"])
    stats["by_year"] = dict(sorted(stats["by_year"].items()))
    stats["by_kind_year"] = dict(sorted(stats["by_kind_year"].items()))
    stats["by_sezione"] = dict(sorted(stats["by_sezione"].items()))
    stats["by_tipoprov"] = dict(stats["by_tipoprov"])
    stats["issues"] = dict(stats["issues"])
    return stats


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def print_report(stats: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"Corpus: {stats['corpus_dir']}")
    print("=" * 72)
    print(f"Files: {len(stats['files'])}")
    print(f"Total records:    {stats['total_records']:>12,}")
    print(f"Total on disk:    {_fmt_bytes(stats['total_bytes']):>12}")
    print(f"Total text chars: {stats['total_chars']:>12,}")
    if stats["total_records"]:
        avg = stats["total_chars"] // stats["total_records"]
        print(f"Avg record chars: {avg:>12,}")
    print()

    print("By kind:")
    for kind, n in sorted(stats["by_kind"].items()):
        print(f"  {kind:>8}: {n:>10,}")
    print()

    print("By year:")
    for year, n in stats["by_year"].items():
        print(f"  {year:>8}: {n:>10,}")
    print()

    print("By (kind, year):")
    for slug, n in stats["by_kind_year"].items():
        print(f"  {slug:>16}: {n:>10,}")
    print()

    print("By sezione (top 15):")
    items = sorted(stats["by_sezione"].items(), key=lambda kv: -kv[1])[:15]
    for sez, n in items:
        print(f"  {sez:>20}: {n:>10,}")
    print()

    print("By tipoprov:")
    for tp, n in sorted(stats["by_tipoprov"].items(), key=lambda kv: -kv[1]):
        print(f"  {tp:>20}: {n:>10,}")
    print()

    ls = stats.get("length_stats") or {}
    if ls:
        print("Text length (chars):")
        for k in ("min", "p10", "p50", "mean", "p90", "p99", "max"):
            print(f"  {k:>6}: {ls[k]:>10,}")
        print()

    print(f"Malformed records:      {stats['malformed_records']:>10,}")
    print(f"Duplicate sentence_ids: {stats['duplicate_sentence_ids']:>10,}")
    if stats["issues"]:
        print("Issues:")
        for issue, n in sorted(stats["issues"].items(), key=lambda kv: -kv[1]):
            print(f"  [{n:>5}] {issue}")
    print()

    if stats.get("samples"):
        print("-" * 72)
        print("Samples (first N per slice):")
        print("-" * 72)
        for block in stats["samples"]:
            print(f"\n### {block['file']}")
            for rec in block["records"]:
                print(f"  - {rec['sentence_id']} ({rec['text_len']:,} chars)")
                print(f"    {rec['article_title']}")
                print(f"    {rec['text_preview']}...")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path, help="Directory with italgiure_*.jsonl")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Show N sample records per slice (default: 0)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write full stats as JSON to this path",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    stats = validate_corpus(args.corpus_dir, sample_per_slice=args.sample)
    print_report(stats)

    if args.json:
        args.json.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nFull stats written to {args.json}")

    if stats["malformed_records"] > 0 or stats["issues"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
