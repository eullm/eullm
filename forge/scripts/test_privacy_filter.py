#!/usr/bin/env python3
"""spaCy vs OpenAI Privacy Filter — with Viterbi operating-point sweep.

Italian F1 on PII-Masking-300k is 0.921 (near English 0.934), so Privacy
Filter is multilingual by design. Legal text is however out-of-distribution
(model card § 7.4.2 — SPY dataset goes 0.545 → 0.962 F1 with 10% finetuning).

This script sweeps Viterbi transition biases (the six documented in the
model card, no training required) to see how much of the Italian-legal gap
we can close by calibration alone before reaching for fine-tuning.

Usage
-----
    # Default sweep across four preset profiles (default / recall+ / recall++ / precision+):
    python forge/scripts/test_privacy_filter.py \\
        ~/italgiure_corpus/italgiure_snciv_2023.jsonl --n 3 --seed 42

    # Single profile with the usual per-record verbose output:
    python forge/scripts/test_privacy_filter.py \\
        ~/italgiure_corpus/italgiure_snciv_2023.jsonl --n 3 --seed 42 \\
        --profile recall++ --verbose

Requirements:
    pip install git+https://github.com/openai/privacy-filter.git
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.anonymize import load_spacy_ner  # noqa: E402

PF_MODEL = "openai/privacy-filter"

BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)

# Preset operating points. All biases default to 0.0 (the shipped calibration).
# Positive background_stay discourages span entry; negative encourages it.
# Positive background_to_start encourages span entry. Positive
# inside_to_continue keeps spans alive longer.
PROFILES: dict[str, dict[str, float]] = {
    "default": {k: 0.0 for k in BIAS_KEYS},
    # Moderate recall boost: easier to enter and stay in a span.
    "recall+": {
        "transition_bias_background_stay": -1.0,
        "transition_bias_background_to_start": +1.0,
        "transition_bias_inside_to_continue": +1.0,
        "transition_bias_inside_to_end": -0.5,
        "transition_bias_end_to_background": 0.0,
        "transition_bias_end_to_start": 0.0,
    },
    # Aggressive recall boost: be very reluctant to stay in background.
    "recall++": {
        "transition_bias_background_stay": -2.5,
        "transition_bias_background_to_start": +2.5,
        "transition_bias_inside_to_continue": +2.0,
        "transition_bias_inside_to_end": -1.5,
        "transition_bias_end_to_background": -0.5,
        "transition_bias_end_to_start": +0.5,
    },
    # Precision boost: make entering a span expensive (fewer FP, more FN).
    "precision+": {
        "transition_bias_background_stay": +1.0,
        "transition_bias_background_to_start": -1.0,
        "transition_bias_inside_to_continue": -0.5,
        "transition_bias_inside_to_end": +0.5,
        "transition_bias_end_to_background": 0.0,
        "transition_bias_end_to_start": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_lines(path: Path, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    picks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for i, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if len(picks) < n:
                picks.append(rec)
            else:
                j = rng.randint(0, i)
                if j < n:
                    picks[j] = rec
    return picks


# ---------------------------------------------------------------------------
# Privacy Filter backend
# ---------------------------------------------------------------------------


def _load_privacy_filter():
    from opf._api import OPF  # type: ignore

    return OPF(output_text_only=False)


def _write_calibration(biases: dict[str, float]) -> Path:
    """Write a viterbi_calibration.json with the given biases and return path."""
    path = Path(tempfile.mkstemp(prefix="opf_calib_", suffix=".json")[1])
    payload = {"operating_points": {"default": {"biases": dict(biases)}}}
    path.write_text(json.dumps(payload, indent=2))
    return path


def _run_pf(
    redactor, text: str, *, calibration_path: Path | None
) -> list[tuple[str, str, int, int, float]]:
    """Run Privacy Filter with optional calibration override."""
    from opf._api import DecodeOptions  # type: ignore

    kwargs = {}
    if calibration_path is not None:
        kwargs["decode"] = DecodeOptions(
            viterbi_calibration_path=str(calibration_path),
        )
    result = redactor.redact(text, **kwargs)
    return [
        (str(sp.label), str(sp.text), int(sp.start), int(sp.end), 1.0)
        for sp in result.detected_spans
    ]


# ---------------------------------------------------------------------------
# Diff + stats
# ---------------------------------------------------------------------------


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


@dataclass
class ProfileStats:
    name: str
    both: int = 0
    only_spacy: int = 0
    only_pf: int = 0
    pf_total: int = 0
    by_label: dict[str, int] = None  # type: ignore[assignment]
    examples_only_spacy: list[str] = None  # type: ignore[assignment]
    examples_only_pf: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.by_label is None:
            self.by_label = {}
        if self.examples_only_spacy is None:
            self.examples_only_spacy = []
        if self.examples_only_pf is None:
            self.examples_only_pf = []


def _diff_person_spans(
    spacy_spans: list[tuple[str, int, int]],
    pf_spans: list[tuple[str, str, int, int, float]],
) -> tuple[list, list, list]:
    pf_persons = [x for x in pf_spans if x[0] == "private_person"]
    only_spacy: list[tuple[str, int, int]] = []
    both: list[tuple] = []
    matched_pf: set[int] = set()
    for sp in spacy_spans:
        sp_range = (sp[1], sp[2])
        match_idx = None
        for i, pf in enumerate(pf_persons):
            if i in matched_pf:
                continue
            if _overlap(sp_range, (pf[2], pf[3])):
                match_idx = i
                break
        if match_idx is None:
            only_spacy.append(sp)
        else:
            matched_pf.add(match_idx)
            both.append((sp, pf_persons[match_idx]))
    only_pf = [p for i, p in enumerate(pf_persons) if i not in matched_pf]
    return both, only_spacy, only_pf


# ---------------------------------------------------------------------------
# Verbose per-record output (only when --verbose + single profile)
# ---------------------------------------------------------------------------


def _fmt_span(text: str, start: int, end: int, *, context: int = 30) -> str:
    left = max(0, start - context)
    right = min(len(text), end + context)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    snippet = text[left:start] + "«" + text[start:end] + "»" + text[end:right]
    return f"{prefix}{snippet.replace(chr(10), ' ')}{suffix}"


def _print_record_verbose(
    idx: int,
    total: int,
    rec: dict,
    spacy_spans,
    pf_spans,
) -> None:
    text = rec.get("text", "")
    both, only_spacy, only_pf = _diff_person_spans(spacy_spans, pf_spans)
    print("=" * 100)
    print(f"[{idx}/{total}] {rec.get('sentence_id')} "
          f"({rec.get('article_title', '')})")
    print(f"    length: {len(text):,} chars | "
          f"spaCy PER: {len(spacy_spans)} | "
          f"PF total: {len(pf_spans)} "
          f"(person={sum(1 for x in pf_spans if x[0] == 'private_person')})")
    by_label: dict[str, int] = {}
    for (lbl, *_rest) in pf_spans:
        by_label[lbl] = by_label.get(lbl, 0) + 1
    if by_label:
        print("    PF labels: " + ", ".join(
            f"{k}={v}" for k, v in sorted(by_label.items(), key=lambda x: -x[1])
        ))
    print("-" * 100)
    print(f"  ## Both ({len(both)})")
    for sp, pf in both[:15]:
        print(f"    spaCy «{sp[0]}»  |  PF «{pf[1]}»")
    if len(both) > 15:
        print(f"    ... +{len(both) - 15} more")
    print(f"  ## Only spaCy ({len(only_spacy)})")
    for sp in only_spacy[:15]:
        print(f"    «{sp[0]}»  ← {_fmt_span(text, sp[1], sp[2])}")
    print(f"  ## Only PF ({len(only_pf)})")
    for pf in only_pf[:15]:
        print(f"    «{pf[1]}»  ← {_fmt_span(text, pf[2], pf[3])}")
    extra = [x for x in pf_spans if x[0] != "private_person"]
    if extra:
        print(f"  ## PF non-person ({len(extra)})")
        per_label: dict[str, int] = {}
        for (lbl, txt, s, e, _sc) in extra:
            per_label[lbl] = per_label.get(lbl, 0) + 1
            if per_label[lbl] <= 3:
                print(f"    [{lbl}] «{txt}»  ← {_fmt_span(text, s, e)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacy-model", default="it_core_news_lg")
    parser.add_argument(
        "--profile",
        default="all",
        choices=["all"] + list(PROFILES.keys()),
        help="Which Viterbi profile to run. 'all' sweeps every preset; a "
        "specific name (e.g. recall++) runs just that one.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-record diff (only meaningful with a single profile).",
    )
    args = parser.parse_args(argv)

    if not args.jsonl.is_file():
        parser.error(f"{args.jsonl} not found")

    print(f"Loading spaCy NER {args.spacy_model}...", file=sys.stderr)
    try:
        spacy_ner = load_spacy_ner(args.spacy_model)
    except RuntimeError as exc:
        print(f"ERROR spaCy: {exc}", file=sys.stderr)
        return 2

    print(f"Loading Privacy Filter {PF_MODEL}...", file=sys.stderr)
    try:
        pf = _load_privacy_filter()
    except ImportError as exc:
        print(
            "ERROR: missing dependency. Install with:\n"
            "    pip install git+https://github.com/openai/privacy-filter.git\n"
            f"({exc})",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR loading Privacy Filter: {exc}", file=sys.stderr)
        return 2

    records = _sample_lines(args.jsonl, args.n, args.seed)
    if not records:
        print(f"No records in {args.jsonl}", file=sys.stderr)
        return 1

    # Pre-compute spaCy spans once — they don't depend on PF profile.
    print("Running spaCy on sampled records...", file=sys.stderr)
    spacy_cache: list[list[tuple[str, int, int]]] = []
    for rec in records:
        spacy_cache.append(spacy_ner(rec.get("text", "")))
    spacy_total = sum(len(s) for s in spacy_cache)
    print(f"spaCy found {spacy_total} PER spans total.", file=sys.stderr)

    profiles_to_run = list(PROFILES.keys()) if args.profile == "all" else [args.profile]
    all_stats: list[ProfileStats] = []

    for prof_name in profiles_to_run:
        biases = PROFILES[prof_name]
        calib_path = _write_calibration(biases)
        print(f"\n>>> Profile '{prof_name}' "
              f"(calibration: {calib_path})", file=sys.stderr)
        stats = ProfileStats(name=prof_name)

        for i, rec in enumerate(records, start=1):
            text = rec.get("text", "")
            pf_spans = _run_pf(pf, text, calibration_path=calib_path)
            stats.pf_total += len(pf_spans)
            for (lbl, *_r) in pf_spans:
                stats.by_label[lbl] = stats.by_label.get(lbl, 0) + 1
            both, only_sp, only_pf_ = _diff_person_spans(
                spacy_cache[i - 1], pf_spans
            )
            stats.both += len(both)
            stats.only_spacy += len(only_sp)
            stats.only_pf += len(only_pf_)
            # Keep a handful of examples for the summary
            for sp in only_sp:
                if len(stats.examples_only_spacy) < 8:
                    stats.examples_only_spacy.append(sp[0])
            for pf_span in only_pf_:
                if len(stats.examples_only_pf) < 8:
                    stats.examples_only_pf.append(pf_span[1])

            if args.verbose and args.profile != "all":
                _print_record_verbose(
                    i, len(records), rec, spacy_cache[i - 1], pf_spans
                )

        all_stats.append(stats)

    # --- Summary table ---
    print()
    print("=" * 100)
    print("SWEEP SUMMARY  (person spans aggregated across all records; "
          f"spaCy total = {spacy_total})")
    print("=" * 100)
    hdr = f"{'profile':<12} {'both':>5} {'only spaCy':>11} {'only PF':>9} "
    hdr += f"{'PF total':>9}   PF labels"
    print(hdr)
    print("-" * 100)
    for s in all_stats:
        labels = ", ".join(
            f"{k}={v}" for k, v in sorted(s.by_label.items(), key=lambda x: -x[1])
        )
        print(f"{s.name:<12} {s.both:>5} {s.only_spacy:>11} {s.only_pf:>9} "
              f"{s.pf_total:>9}   {labels}")
    print()
    for s in all_stats:
        print(f"[{s.name}] Only spaCy examples: "
              + ", ".join(f"«{x}»" for x in s.examples_only_spacy[:6]))
        print(f"[{s.name}] Only PF examples:    "
              + ", ".join(f"«{x}»" for x in s.examples_only_pf[:6]))
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
