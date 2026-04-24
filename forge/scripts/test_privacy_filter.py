#!/usr/bin/env python3
"""Side-by-side comparison: spaCy it_core_news_lg vs OpenAI Privacy Filter.

Runs both NER backends on the SAME sampled italgiure records and prints
the entities each one finds, plus a diff. Purely exploratory — does NOT
touch anonymize.py and does NOT write anywhere.

Goal: decide whether to swap the spaCy NER layer in anonymize.py for
openai/privacy-filter (HF), which should give us:
    * higher F1 (96% on PII-Masking-300k vs ~80% typical spaCy)
    * fewer FP on Italian legal jargon (acronyms, institutions)
    * native detection of address / email / phone / date / url / account

Usage
-----
    # Compare on 3 records with the same seed as the last peek:
    python forge/scripts/test_privacy_filter.py \\
        ~/italgiure_corpus/italgiure_snciv_2023.jsonl --n 3 --seed 42

    # Use CUDA (default is auto; set explicitly if needed):
    python forge/scripts/test_privacy_filter.py \\
        ~/italgiure_corpus/italgiure_snciv_2023.jsonl --n 3 --device cuda

Requirements (install in the conda env, no pyproject change yet):
    pip install transformers torch
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.anonymize import load_spacy_ner  # noqa: E402

PF_MODEL = "openai/privacy-filter"


# ---------------------------------------------------------------------------
# Sampling (same logic as peek_anonymization.py for reproducibility)
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


def _load_privacy_filter(device: str):
    from transformers import pipeline

    kwargs: dict = {"task": "token-classification", "model": PF_MODEL,
                    "aggregation_strategy": "simple"}
    if device == "cpu":
        kwargs["device"] = -1
    elif device == "cuda":
        kwargs["device"] = 0
    return pipeline(**kwargs)


def _run_pf(pipe, text: str) -> list[tuple[str, str, int, int, float]]:
    """Run Privacy Filter and return (label, span_text, start, end, score)."""
    out = pipe(text)
    results: list[tuple[str, str, int, int, float]] = []
    for item in out:
        label = item.get("entity_group") or item.get("entity", "")
        word = item.get("word", "")
        start = int(item.get("start", 0))
        end = int(item.get("end", 0))
        score = float(item.get("score", 0.0))
        results.append((label, word, start, end, score))
    return results


# ---------------------------------------------------------------------------
# Diff: which spans overlap between the two extractors?
# ---------------------------------------------------------------------------


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _diff_person_spans(
    spacy_spans: list[tuple[str, int, int]],
    pf_spans: list[tuple[str, str, int, int, float]],
) -> tuple[list, list, list]:
    """Split spans into (both, only_spacy, only_pf) based on character overlap.

    Only PF ``private_person`` spans are considered (so the comparison is
    apples-to-apples with spaCy PER).
    """
    pf_persons = [
        (lbl, txt, s, e, sc) for (lbl, txt, s, e, sc) in pf_spans
        if lbl == "private_person"
    ]
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
# Pretty printing
# ---------------------------------------------------------------------------


def _fmt_span(text: str, start: int, end: int, *, context: int = 30) -> str:
    left = max(0, start - context)
    right = min(len(text), end + context)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    snippet = text[left:start] + "«" + text[start:end] + "»" + text[end:right]
    snippet = snippet.replace("\n", " ")
    return f"{prefix}{snippet}{suffix}"


def _print_record(
    idx: int,
    total: int,
    rec: dict,
    spacy_spans: list[tuple[str, int, int]],
    pf_spans: list[tuple[str, str, int, int, float]],
) -> None:
    text = rec.get("text", "")
    print("=" * 100)
    print(f"[{idx}/{total}] {rec.get('sentence_id')} "
          f"({rec.get('article_title', '')})")
    print(f"    length: {len(text):,} chars")
    print(f"    spaCy PER spans: {len(spacy_spans)}")
    print(f"    PF spans total:  {len(pf_spans)} "
          f"(person={sum(1 for x in pf_spans if x[0] == 'private_person')})")
    print("-" * 100)

    # Per-label breakdown of PF
    by_label: dict[str, int] = {}
    for (lbl, *_rest) in pf_spans:
        by_label[lbl] = by_label.get(lbl, 0) + 1
    if by_label:
        print("  PF labels: " + ", ".join(
            f"{k}={v}" for k, v in sorted(by_label.items(), key=lambda x: -x[1])
        ))

    both, only_spacy, only_pf = _diff_person_spans(spacy_spans, pf_spans)

    print()
    print(f"  ## Both ({len(both)}) — agreement on persons")
    for sp, pf in both[:20]:
        print(f"    spaCy «{sp[0]}»  |  PF «{pf[1]}» (score={pf[4]:.2f})")
    if len(both) > 20:
        print(f"    ... +{len(both) - 20} more")

    print()
    print(f"  ## Only spaCy ({len(only_spacy)}) — likely FP of spaCy "
          f"(or FN of PF)")
    for sp in only_spacy[:20]:
        print(f"    «{sp[0]}»  ← {_fmt_span(text, sp[1], sp[2])}")
    if len(only_spacy) > 20:
        print(f"    ... +{len(only_spacy) - 20} more")

    print()
    print(f"  ## Only PF ({len(only_pf)}) — extra catches by PF "
          f"(or FP of PF)")
    for pf in only_pf[:20]:
        print(f"    «{pf[1]}» (score={pf[4]:.2f})  ← "
              f"{_fmt_span(text, pf[2], pf[3])}")
    if len(only_pf) > 20:
        print(f"    ... +{len(only_pf) - 20} more")

    # Extra PF labels that are NOT person — interesting for potential
    # replacement of regex layer. Only show a few per class.
    extra = [x for x in pf_spans if x[0] != "private_person"]
    if extra:
        print()
        print(f"  ## PF non-person entities ({len(extra)}) — audit vs "
              f"regex layer")
        per_label_shown: dict[str, int] = {}
        for (lbl, txt, s, e, sc) in extra:
            per_label_shown[lbl] = per_label_shown.get(lbl, 0) + 1
            if per_label_shown[lbl] > 5:
                continue
            print(f"    [{lbl}] «{txt}» (score={sc:.2f})  ← "
                  f"{_fmt_span(text, s, e)}")
        for lbl, n in per_label_shown.items():
            if n > 5:
                print(f"    [{lbl}] ... +{n - 5} more")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Path to a raw italgiure_*.jsonl")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacy-model", default="it_core_news_lg")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Where to run Privacy Filter (default: auto — uses cuda if "
        "available, else cpu)",
    )
    args = parser.parse_args(argv)

    if not args.jsonl.is_file():
        parser.error(f"{args.jsonl} not found")

    # Resolve device
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # Load spaCy
    print(f"Loading spaCy NER {args.spacy_model}...", file=sys.stderr)
    try:
        spacy_ner = load_spacy_ner(args.spacy_model)
    except RuntimeError as exc:
        print(f"ERROR spaCy: {exc}", file=sys.stderr)
        return 2

    # Load Privacy Filter
    print(f"Loading Privacy Filter {PF_MODEL} on {device}...", file=sys.stderr)
    try:
        pf = _load_privacy_filter(device)
    except ImportError as exc:
        print(
            "ERROR: missing dependency. Install with:\n"
            "    pip install transformers torch\n"
            f"({exc})",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001 — want the raw error surfaced
        print(f"ERROR loading Privacy Filter: {exc}", file=sys.stderr)
        return 2

    records = _sample_lines(args.jsonl, args.n, args.seed)
    if not records:
        print(f"No records in {args.jsonl}", file=sys.stderr)
        return 1

    # Per-record comparison + global tallies
    total_both = total_only_spacy = total_only_pf = 0
    for i, rec in enumerate(records, start=1):
        text = rec.get("text", "")
        spacy_spans = spacy_ner(text)
        pf_spans = _run_pf(pf, text)
        _print_record(i, len(records), rec, spacy_spans, pf_spans)
        both, only_spacy, only_pf = _diff_person_spans(spacy_spans, pf_spans)
        total_both += len(both)
        total_only_spacy += len(only_spacy)
        total_only_pf += len(only_pf)

    print("=" * 100)
    print("SUMMARY (person spans across all records)")
    print(f"    Agreement (both):   {total_both}")
    print(f"    Only spaCy:         {total_only_spacy}  "
          f"← suspected spaCy FP if low score / junk text")
    print(f"    Only Privacy Filter: {total_only_pf}  "
          f"← potential new catches (or PF FP)")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
