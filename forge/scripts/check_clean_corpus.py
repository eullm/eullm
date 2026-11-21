#!/usr/bin/env python3
"""Leonardo's cleaned corpus (eullm_runs/clean) checked against JUPITER's evaluation sets.

    python forge/scripts/check_clean_corpus.py --clean $D/leonardo-clean \\
        --sets norm-exam=$D/eval/norm-exam.jsonl trap-test=$D/eval/trap-test.jsonl ... \\
        --out $D/leonardo-clean-ok

Leonardo cleaned its corpus against its own evaluation sets, each named in
the folder's manifest.json with its sha256. A set of ours whose sha256 is
among them is covered by that cleaning. Any other (the trap test and its
rewrites, the LexIA traps, a set in another version) is not, and every row
on one of its articles or made-up numbers is dropped here, whatever its
form: the strict rule, since we cannot tell which of them are held out.
Also checked: each file's sha256 against the manifest, so a copy cut short
is caught. Writes the kept rows under --out, one file per input; prints
counts only, never a text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.law_corpus import excluded  # noqa: E402

_SHA = re.compile(r"\b[0-9a-f]{64}\b")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_hashes(manifests: list[Path]) -> set[str]:
    """Every sha256 a manifest names, sets and files alike, wherever it is."""
    return {m for p in manifests for m in _SHA.findall(p.read_text(encoding="utf-8"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", type=Path, required=True, help="the copy of eullm_runs/clean")
    ap.add_argument("--sets", nargs="+", required=True, help="name=path of our evaluation sets")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    manifests = sorted(args.clean.rglob("manifest.json"))
    if not manifests:
        raise SystemExit(f"[clean] no manifest.json under {args.clean}")
    known = manifest_hashes(manifests)
    uncovered = []
    for spec in args.sets:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not p.is_file():
            print(f"[clean] {name}: missing ({p}), left out of the check")
            continue
        covered = sha256(p) in known
        state = "covered by Leonardo" if covered else "NOT covered: filtered here"
        print(f"[clean] {name}: {state}")
        if not covered:
            uncovered.append(p)
    fakes, asked = excluded(uncovered)
    drop = asked | fakes
    print(f"[clean] {len(drop):,} articles and numbers of the uncovered sets left out")

    bad = 0
    for f in sorted(args.clean.rglob("*.jsonl")):
        digest = sha256(f)
        if digest not in known:
            print(f"[!!] {f.relative_to(args.clean)}: sha256 not in the manifest")
            bad += 1
            continue
        out = args.out / f.relative_to(args.clean)
        out.parent.mkdir(parents=True, exist_ok=True)
        kept, why = 0, Counter()
        with f.open(encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (str(r.get("code")), str(r.get("articolo", "")).lower())
                if key in drop:
                    why[r.get("forma")] += 1
                    continue
                dst.write(line if line.endswith("\n") else line + "\n")
                kept += 1
        print(f"[clean] {f.relative_to(args.clean)}: {kept:,} kept, {sum(why.values()):,} dropped"
              + (f" ({', '.join(f'{k} {v}' for k, v in sorted(why.items()))})" if why else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
