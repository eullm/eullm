#!/usr/bin/env python3
"""Sweep structured PII out of a corpus that has ALREADY been anonymised.

`anonymize_italgiure.py` runs on the raw italgiure slices, before chunking
and formatting. This script is the safety net for the other end of the
pipeline: it re-checks the *final* training files (``train.jsonl`` /
``val.jsonl``) for structured identifiers that survived — because the corpus
was produced by an older revision of ``eullm_forge.datasets.anonymize``,
because a slice was skipped, or because a chunk boundary split a record.

It is deliberately NOT a second anonymisation pass:

* The **name** layers (spaCy NER and the all-caps heuristic) are off and
  cannot be turned on here. Person tokens are numbered per document; a
  chunked corpus has no documents left, so a second pass would assign
  ``[PERSONA_1]`` independently inside every chunk and destroy the coherence
  the first pass created. Names are the first pass's job.
* Only the deterministic, context-free regex layers run, and they are
  idempotent: ``[CODICE_FISCALE]`` does not match ``RE_CF``, so running the
  sweep twice changes nothing the second time.

Default layers are the four that cannot produce a false positive on Italian
legal text: codice fiscale, partita IVA, IBAN, email. ``phone``, ``birth``
and ``address`` are opt-in via ``--layers`` — ``RE_PHONE`` in particular
matches bare digit runs and will eat article/protocol numbers if the corpus
never went through the first pass.

Usage:
    # Report only — writes nothing, exits 1 if anything was found.
    python forge/scripts/sweep_structured_pii.py "$EULLM_DATA_DIR"/*.jsonl

    # Show the offending strings (they are personal data — terminal only).
    python forge/scripts/sweep_structured_pii.py train.jsonl --show

    # Rewrite in place (atomic: temp file + rename, original kept as .bak).
    python forge/scripts/sweep_structured_pii.py train.jsonl val.jsonl --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.anonymize import (  # noqa: E402
    RE_ADDRESS,
    RE_BIRTH_CLAUSE,
    RE_CF,
    RE_CF_AZIENDA,
    RE_EMAIL,
    RE_IBAN,
    RE_PHONE,
    RE_PIVA,
    AnonymiserConfig,
    anonymize_text,
)

# Layer name → the AnonymiserConfig flag it drives. Kept explicit rather than
# derived from the dataclass so adding a name-based layer to the config never
# silently becomes selectable here.
LAYERS: dict[str, str] = {
    "cf": "redact_cf",
    "piva": "redact_piva",
    "iban": "redact_iban",
    "email": "redact_email",
    "phone": "redact_phone",
    "birth": "redact_birth",
    "address": "redact_address",
}

# Layer name → the patterns that layer can fire on. Used to pre-filter raw
# lines before paying for json.loads (see `_build_prefilter`). Every pattern
# `anonymize_text` applies for a layer MUST be listed, or the sweep would skip
# a line it should have redacted — that is the one way this optimisation can
# be wrong, so the mapping is spelled out rather than inferred.
LAYER_PATTERNS = {
    "cf": (RE_CF,),
    "piva": (RE_CF_AZIENDA, RE_PIVA),
    "iban": (RE_IBAN,),
    "email": (RE_EMAIL,),
    "phone": (RE_PHONE,),
    "birth": (RE_BIRTH_CLAUSE,),
    "address": (RE_ADDRESS,),
}

DEFAULT_LAYERS = ("cf", "piva", "iban", "email")

# How often the progress line is refreshed, in seconds.
PROGRESS_INTERVAL = 2.0


def build_config(layers: list[str]) -> AnonymiserConfig:
    """AnonymiserConfig with every layer off except the requested ones.

    ``use_ner`` and ``redact_allcaps_names`` are forced off — see the module
    docstring for why the name layers must not run on a chunked corpus.
    """
    kwargs = {flag: False for flag in LAYERS.values()}
    for layer in layers:
        kwargs[LAYERS[layer]] = True
    return AnonymiserConfig(
        use_ner=False,
        redact_allcaps_names=False,
        **kwargs,
    )


def build_prefilter(layers: list[str]) -> tuple:
    """Patterns that decide whether a raw JSONL line is worth parsing.

    This is NOT a speed optimisation — measured on a 187 MB corpus it is a
    wash, because the regex pass dominates and the prefilter runs the same
    patterns the redaction would. It is here so that ``--apply`` can copy
    untouched lines through byte for byte instead of round-tripping them
    through ``json.loads``/``json.dumps``, which would silently renormalise
    key order, spacing and escaping across the whole corpus. On a rewrite of
    the training data, "identical" is worth more than "equivalent".

    It is safe because every default pattern matches pure ASCII with no
    character JSON has to escape: a codice fiscale, IBAN, email or P.IVA
    cannot span a ``\\n`` or a ``\\"``, so JSON encoding can neither hide a
    match nor split one. Matching against the whole line (metadata fields
    included) can only over-select, which costs a wasted parse and never a
    missed redaction.
    """
    patterns: list = []
    for layer in layers:
        patterns.extend(LAYER_PATTERNS[layer])
    return tuple(patterns)


def sweep_file(
    path: Path,
    *,
    config: AnonymiserConfig,
    prefilter: tuple,
    field: str,
    apply: bool,
    show: bool,
    progress: bool = False,
) -> tuple[Counter, int]:
    """Sweep one JSONL file.

    Returns ``(category_counts, records_changed)``. With ``apply`` the file is
    rewritten atomically: output goes to ``<path>.tmp``, the original is moved
    to ``<path>.bak``, then the temp file takes its place. A crash mid-run
    therefore never leaves a truncated corpus behind.

    Lines the ``prefilter`` clears are never parsed, and in ``apply`` mode are
    copied through byte for byte — so a clean corpus comes out of ``--apply``
    identical to what went in, not merely equivalent after a JSON round-trip.
    """
    counts: Counter = Counter()
    changed = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    out_f = tmp.open("w", encoding="utf-8") if apply else None

    total_bytes = path.stat().st_size
    seen_bytes = 0
    lineno = 0
    t0 = time.monotonic()
    next_tick = t0 + PROGRESS_INTERVAL

    try:
        with path.open(encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                seen_bytes += len(raw.encode("utf-8"))
                if progress and time.monotonic() >= next_tick:
                    _tick(path, seen_bytes, total_bytes, lineno, sum(counts.values()), t0)
                    next_tick = time.monotonic() + PROGRESS_INTERVAL

                # Fast path: nothing any enabled layer could match, so the
                # record cannot change. Skip the JSON round-trip entirely.
                if not any(p.search(raw) for p in prefilter):
                    if out_f is not None:
                        out_f.write(raw)
                    continue

                stripped = raw.strip()
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    print(
                        f"[WARN] {path.name}:{lineno} malformed JSON: {exc}",
                        file=sys.stderr,
                    )
                    if out_f is not None:
                        out_f.write(raw)
                    continue

                text = rec.get(field)
                if not isinstance(text, str):
                    if out_f is not None:
                        out_f.write(raw)
                    continue

                new_text, stats = anonymize_text(text, config=config)
                hits = stats.to_dict()
                if not sum(hits.values()):
                    # Prefilter matched something outside `field` (a URL in
                    # metadata, say). Nothing changed, so pass the line through.
                    if out_f is not None:
                        out_f.write(raw)
                    continue

                changed += 1
                for k, v in hits.items():
                    if v:
                        counts[k] += v
                if show:
                    _report_hits(path, lineno, text, new_text)
                rec[field] = new_text
                if out_f is not None:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        if out_f is not None:
            out_f.close()
            tmp.unlink(missing_ok=True)
        raise

    if out_f is not None:
        out_f.close()
        os.replace(path, path.with_suffix(path.suffix + ".bak"))
        os.replace(tmp, path)

    if progress:
        _tick(path, seen_bytes, total_bytes, lineno, sum(counts.values()), t0, final=True)

    return counts, changed


def _tick(
    path: Path,
    seen: int,
    total: int,
    lineno: int,
    hits: int,
    t0: float,
    *,
    final: bool = False,
) -> None:
    """Write a one-line progress update to stderr, overwritten in place."""
    dt = max(time.monotonic() - t0, 1e-6)
    mb = seen / 1e6
    pct = (seen / total * 100) if total else 100.0
    eta = ""
    if not final and seen and total > seen:
        eta = f" eta {(total - seen) / (seen / dt) / 60:.1f}m"
    line = (
        f"  {path.name}: {pct:5.1f}%  {lineno:,} rec  "
        f"{mb / dt:.0f} MB/s  {hits} hit(s){eta}"
    )
    end = "\n" if final else ""
    print(f"\r{line:<78}{end}", end=end or "", file=sys.stderr, flush=True)


def _report_hits(path: Path, lineno: int, before: str, after: str) -> None:
    """Print the redacted spans with a little surrounding context.

    Diffing token-by-token would be overkill: we only need to show a human
    what disappeared, so we walk the two strings in parallel and print the
    slice of ``before`` that the substitution replaced.
    """
    i = j = 0
    while i < len(before) and j < len(after):
        if before[i] == after[j]:
            i += 1
            j += 1
            continue
        # A placeholder starts here in `after`; find where it ends.
        end_j = after.find("]", j)
        if end_j == -1:
            break
        placeholder = after[j:end_j + 1]
        # Re-sync: find the first following run that matches again.
        tail = after[end_j + 1:end_j + 21]
        end_i = before.find(tail, i) if tail else len(before)
        if end_i == -1:
            end_i = len(before)
        original = before[i:end_i]
        ctx = before[max(0, i - 40):i].replace("\n", " ")
        print(f"  {path.name}:{lineno}  …{ctx}»{original}« → {placeholder}")
        i = end_i
        j = end_j + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="JSONL files to sweep")
    parser.add_argument(
        "--field",
        default="text",
        help="Record field holding the training text (default: text)",
    )
    parser.add_argument(
        "--layers",
        default=",".join(DEFAULT_LAYERS),
        help=(
            "Comma-separated regex layers to run. Available: "
            + ", ".join(LAYERS)
            + f" (default: {','.join(DEFAULT_LAYERS)}). phone/birth/address are "
            "opt-in: they can match legitimate legal text on a corpus that "
            "never went through anonymize_italgiure.py."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the files. Without it the script only reports.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Print each redacted span with context. The output IS personal "
            "data — keep it on the terminal, do not redirect it into the repo."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-file progress line (it goes to stderr).",
    )
    args = parser.parse_args(argv)

    layers = [layer.strip() for layer in args.layers.split(",") if layer.strip()]
    unknown = [layer for layer in layers if layer not in LAYERS]
    if unknown:
        parser.error(f"unknown layer(s): {', '.join(unknown)}")
    if not layers:
        parser.error("--layers selected nothing to run")

    missing = [p for p in args.files if not p.is_file()]
    if missing:
        parser.error(f"not a file: {', '.join(str(p) for p in missing)}")

    config = build_config(layers)
    prefilter = build_prefilter(layers)
    print(f"Layers: {', '.join(layers)}", file=sys.stderr)
    print(
        "Mode:   " + ("APPLY (files rewritten, .bak kept)" if args.apply else "report only"),
        file=sys.stderr,
    )

    grand: Counter = Counter()
    for path in args.files:
        counts, changed = sweep_file(
            path,
            config=config,
            prefilter=prefilter,
            field=args.field,
            apply=args.apply,
            show=args.show,
            progress=not args.quiet and sys.stderr.isatty(),
        )
        grand.update(counts)
        total = sum(counts.values())
        detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "clean"
        print(f"{path.name}: {total} hit(s) in {changed} record(s) — {detail}")

    total = sum(grand.values())
    print("=" * 64)
    if not total:
        print("No structured PII found.")
        return 0
    for k, v in sorted(grand.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>20}: {v:>8,}")
    print("=" * 64)
    if args.apply:
        print("Corpus rewritten. Re-run without --apply to confirm it is clean.")
        return 0
    print("Re-run with --apply to redact them.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
