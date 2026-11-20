#!/usr/bin/env python3
"""Is an exam result owed to training rows on the exam's own articles?

    python forge/scripts/split_by_article.py $WORK/eval/practical-dev-graded \\
        --items $WORK/eval/practical-dev.jsonl --baseline r2l-steps-8b-quote-q4-hyb \\
        --corpus $WORK/eullm_runs/memory-gemma-ok/rewrites-*.jsonl

The knowledge corpora rewrite every article, those of the exams included
(only the closed-book suite's are kept out, and only of the questions
style): on 2026-11-20 the Gemma corpus held 141 cases on the articles of
practical-dev and 65 on those of practical-heldout. An exam's questions are
other cases on the same articles, so a model trained on those rows may
score higher on them than on articles it never practised on.

The exam's questions are split in two by their article: A, an article the
corpus has a row of ``--forms`` on (cases and explanations by default), and
B, one it has none on. Per model, its grades in each group; against the
baseline, what it wins and loses in each (paired, exact McNemar). A gain in A
and none in B is the overlap's; the same gain in both is the model's. B can
be small when the corpus covers nearly every article: its size is printed,
and with a handful of questions it says little either way.

Counts only, never a question: safe on held-out exams.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval.paired import Graded, compare, load_graded  # noqa: E402


def _rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def article_of_items(items: Path) -> dict[str, tuple[str, str]]:
    """Exam item id -> (code, article), from the item or its metadata."""
    out = {}
    for r in _rows(items):
        meta = {**r, **(r.get("metadata") or {})}
        if r.get("id") is not None and meta.get("code") and meta.get("articolo"):
            out[str(r["id"])] = (str(meta["code"]), str(meta["articolo"]).lower())
    return out


def corpus_articles(files: list[Path], forms: set[str]) -> set[tuple[str, str]]:
    return {(str(r.get("code")), str(r.get("articolo")).lower())
            for p in files if not p.name.endswith(".rejects.jsonl")
            for r in _rows(p) if r.get("forma") in forms}


def subset(g: Graded, ids: set[str]) -> Graded:
    return Graded(g.label, {i: v for i, v in g.grades.items() if i in ids},
                  {i: v for i, v in g.lengths.items() if i in ids},
                  {i: v for i, v in g.verif.items() if i in ids})


def report(models: dict[str, Graded], articles: dict[str, tuple[str, str]],
           covered: set[tuple[str, str]], baseline: str | None) -> list[str]:
    a = {i for i, art in articles.items() if art in covered}
    b = set(articles) - a
    graded_ids = set().union(*[set(g.grades) for g in models.values()])
    lines = [f"A, article with corpus rows: {len(a)} questions; B, without: {len(b)}; "
             f"no article named: {len(graded_ids - set(articles))}"]
    lines.append(f"{'model':<30} {'A right/n':>12} {'B right/n':>12}"
                 + ("   vs baseline: A won:lost (p)   B won:lost (p)" if baseline else ""))
    for label in sorted(models):
        g = models[label]
        row = []
        for ids in (a, b):
            s = subset(g, ids)
            c = s.counts()
            row.append(f"{c['correct']}/{len(s.grades)}")
        line = f"{label:<30} {row[0]:>12} {row[1]:>12}"
        if baseline and label != baseline:
            parts = []
            for ids in (a, b):
                cmp = compare(subset(g, ids), subset(models[baseline], ids))
                parts.append(f"{cmp.a_only}:{cmp.base_only} (p={cmp.p:.2f})")
            line += f"   {parts[0]:>18}   {parts[1]:>18}"
        lines.append(line)
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graded", nargs="+", type=Path,
                    help="a directory of *.graded.jsonl, or the files themselves")
    ap.add_argument("--items", type=Path, required=True, help="the exam's questions")
    ap.add_argument("--corpus", nargs="+", type=Path, required=True,
                    help="the training rows whose articles count as seen")
    ap.add_argument("--forms", default="caso,spiegazione",
                    help="comma-separated forme that count (default caso,spiegazione)")
    ap.add_argument("--baseline", help="label to compare every model against")
    ap.add_argument("--models", nargs="*", default=[], help="only these labels (and the baseline)")
    a = ap.parse_args(argv)

    files = []
    for p in a.graded:
        files.extend(sorted(p.glob("*.graded.jsonl")) if p.is_dir() else [p])
    models = {g.label: g for g in map(load_graded, files)}
    if a.models:
        keep = set(a.models) | ({a.baseline} if a.baseline else set())
        models = {k: v for k, v in models.items() if k in keep}
    if a.baseline and a.baseline not in models:
        print(f"[split] no graded answers for {a.baseline!r}; have: {', '.join(sorted(models))}",
              file=sys.stderr)
        return 1
    covered = corpus_articles(a.corpus, set(a.forms.split(",")))
    articles = article_of_items(a.items)
    print(f"[split] corpus: {len(covered):,} articles with {a.forms}; exam: "
          f"{len(articles):,} questions naming an article")
    print("\n".join(report(models, articles, covered, a.baseline)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
