#!/usr/bin/env python3
"""Training rows with nothing of the evaluation sets in them, and a manifest of counts.

    python forge/scripts/clean_corpus.py --out-dir $WORK/eullm_runs/clean/gemma-ok \\
        --rows $WORK/eullm_runs/memory-gemma-ok/rewrites-*.jsonl \\
        --heldout $WORK/eval/norm-exam.jsonl $WORK/eval/practical-heldout.jsonl \\
        --dev $WORK/eval/closed-book.jsonl $WORK/eval/practical-dev.jsonl

The knowledge corpora were written from every article, the evaluation sets'
included: only the closed-book suite's articles were kept out, and only of
the questions style. On 2026-11-20 the Gemma rows held 141 cases on the
articles of practical-dev and 65 on those of practical-heldout. The rule
here, for every row (written by a teacher or by the program):

* an article a held-out set asks about: no row of any form;
* an article a development set asks about (the closed-book suite too): no
  row of the forms that read like an exam question (``--dev-forms``: questions,
  cases, explanations); its text, paraphrases, elements and structure stay,
  since learning the statute is what the corpus is for and what the
  closed-book suite measures;
* an article number a set makes up (``esiste_falsa``): no row at all.

Each row file is written again under ``--out-dir`` with its own name, and
``manifest.json`` beside it says what went in and out: rows by file, form
and teacher, dropped by reason, the sets used with their article counts and
sha256, and each output's sha256. Counts only, never a text: the manifest
can travel with the files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.law_corpus import excluded  # noqa: E402

DEV_FORMS = "domande,caso,spiegazione"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def articles(paths: list[Path]) -> tuple[set, set]:
    """(articles asked about, made-up numbers asked about) of the sets."""
    fakes, asked = excluded(paths) if paths else (set(), set())
    return asked, fakes


def why_dropped(row: dict, heldout: set, dev: set, fakes: set, dev_forms: set) -> str | None:
    key = (str(row.get("code")), str(row.get("articolo")).lower())
    if key in fakes:
        return "made-up number of a set"
    if key in heldout:
        return "held-out article"
    if key in dev and row.get("forma") in dev_forms:
        return "development article, question-like form"
    return None


def clean(rows: list[Path], out_dir: Path, heldout_sets: list[Path], dev_sets: list[Path],
          dev_forms: set[str]) -> dict:
    h_asked, h_fakes = articles(heldout_sets)
    d_asked, d_fakes = articles(dev_sets)
    fakes = h_fakes | d_fakes
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for src in rows:
        if src.name.endswith(".rejects.jsonl"):
            continue
        dst = out_dir / src.name
        if dst.resolve() == src.resolve():
            raise SystemExit(f"{dst} is the input: give another --out-dir")
        kept: Counter = Counter()
        dropped: Counter = Counter()
        with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                reason = why_dropped(row, h_asked, d_asked, fakes, dev_forms)
                if reason:
                    dropped[reason] += 1
                    continue
                out.write(line if line.endswith("\n") else line + "\n")
                kept[(row.get("teacher") or row.get("source") or "?",
                      row.get("forma") or row.get("tipo") or "?")] += 1
        by = defaultdict(dict)
        for (teacher, forma), n in sorted(kept.items()):
            by[teacher][forma] = n
        files[dst.name] = {"source": str(src), "rows": sum(kept.values()),
                           "by_teacher_and_form": by, "dropped": dict(dropped),
                           "sha256": sha256(dst)}
    sets = {"held_out": {str(p): {"sha256": sha256(p)} for p in heldout_sets},
            "development": {str(p): {"sha256": sha256(p)} for p in dev_sets}}
    for kind, paths in (("held_out", heldout_sets), ("development", dev_sets)):
        for p in paths:
            asked, made_up = articles([p])
            sets[kind][str(p)].update(articles=len(asked), made_up_numbers=len(made_up))
    manifest = {"rule": {"held_out": "no row on the article, any form",
                         "development": f"no row of {sorted(dev_forms)} on the article",
                         "made_up_numbers": "no row on a number a set makes up"},
                "sets": sets, "files": files,
                "rows": sum(f["rows"] for f in files.values()),
                "dropped": dict(sum((Counter(f["dropped"]) for f in files.values()), Counter()))}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False)
                                           + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--heldout", nargs="*", type=Path, default=[],
                    help="held-out sets: no row of any form on their articles")
    ap.add_argument("--dev", nargs="*", type=Path, default=[],
                    help="development sets and the closed-book suite: no question-like rows "
                         "on their articles")
    ap.add_argument("--dev-forms", default=DEV_FORMS)
    a = ap.parse_args(argv)
    missing = [p for p in a.heldout + a.dev + a.rows if not p.is_file()]
    if missing:
        print(f"[clean] missing: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1
    m = clean(a.rows, a.out_dir, a.heldout, a.dev, set(a.dev_forms.split(",")))
    for name, f in m["files"].items():
        print(f"[clean] {name}: {f['rows']:,} kept, dropped {f['dropped'] or 'none'}")
    print(f"[clean] {m['rows']:,} rows in {a.out_dir}; dropped {m['dropped']}; "
          f"manifest {a.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
