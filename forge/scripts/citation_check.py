#!/usr/bin/env python3
"""Which articles the answers cite, against the articles each question is about.

    python forge/scripts/citation_check.py $P/practical-dev.jsonl \\
        $RUNS/teacher-eval/answers-practical-dev-*.jsonl \\
        --graded-dir $RUNS/teacher-eval/graded-v3

On 27-10-2026 three of the five practical questions g9r-r2 lost to
dqp-d-q9b cited the wrong article: the neighbouring number, or the same
number of another code among the retrieved texts. Whether GRPO taught that
or the start model did it as often is a count, not a judge's opinion: each
article an answer cites ("art. 1120 del codice civile", "artt. 1176 e
1375", "art. 32 c.p.a.") is read with its code (named right after it, or
the code last named, or the question's own), and set against the
question's articles (``metadata.gruppo`` of a practical item, or its
``code``/``articolo``).

Per answers file:

* ``cites``: answers citing any article;
* ``on``: answers citing at least one of the question's articles;
* ``off``: answers citing an article outside them, split into ``near``
  (same code, a number within 3 of one of them), ``code`` (another code)
  and the rest;
* ``only-off``: answers that cite, but none of the question's articles;
* with ``--graded-dir``, the share of ``off`` answers among those the judge
  graded correct, partial and wrong.

Counts only, never a question or an answer: the sets may be private.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval import load_eval_set  # noqa: E402
from eullm_forge.rl.citation import citations  # noqa: E402

NEAR = 3


def question_articles(md: dict) -> set[tuple[str, str]]:
    out = set()
    for ref in md.get("gruppo") or []:
        c, _, a = str(ref).partition(":")
        if c and a:
            out.add((c, a))
    if md.get("code") and md.get("articolo"):
        out.add((md["code"], str(md["articolo"])))
    return out


def _number(article: str) -> int | None:
    head = article.split("-", 1)[0]
    return int(head) if head.isdigit() else None


def kind(cite: tuple[str | None, str], group: set[tuple[str, str]]) -> str:
    """"on", "near", "code" or "far" for one citation."""
    if cite in group:
        return "on"
    codes = {c for c, _ in group}
    if cite[0] not in codes:
        return "code"
    n = _number(cite[1])
    if n is not None and any(c == cite[0] and (m := _number(a)) is not None and abs(m - n) <= NEAR
                             for c, a in group):
        return "near"
    return "far"


def label_of(path: Path) -> str:
    name = path.name
    return name[len("answers-"):-len(".jsonl")] if name.startswith("answers-") else path.stem


def check(path: Path, items: dict, graded: dict[str, str]) -> Counter:
    n = Counter()
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        md = items.get(row.get("id"))
        if md is None:
            continue
        group = question_articles(md)
        if not group:
            continue
        n["n"] += 1
        cites = citations(row.get("answer", ""), md.get("code") or next(iter(group))[0])
        kinds = {kind(c, group) for c in cites}
        grade = graded.get(row["id"])
        if grade:
            n[f"n-{grade}"] += 1
        if not cites:
            continue
        n["cites"] += 1
        n["on"] += "on" in kinds
        off = bool(kinds - {"on"})
        n["off"] += off
        n["only-off"] += "on" not in kinds
        for k in ("near", "code", "far"):
            n[k] += k in kinds
        if grade and off:
            n[f"off-{grade}"] += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", type=Path, help="the question set the answers are to")
    ap.add_argument("answers", nargs="+", type=Path, help="answers-*.jsonl of legal_eval.py")
    ap.add_argument("--graded-dir", type=Path,
                    help="where the judge's <answers>.graded.jsonl are (judge_answers.py)")
    args = ap.parse_args(argv)
    items = {it.id: it.metadata for it in load_eval_set(args.items)}
    rows = []
    for path in args.answers:
        if path.name.endswith(".graded.jsonl") or not path.is_file():
            continue
        graded = {}
        g = (args.graded_dir or path.parent) / path.with_suffix(".graded.jsonl").name
        if g.is_file():
            for line in g.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    graded[r.get("id")] = r.get("grade")
        n = check(path, items, graded)
        if n["n"]:
            rows.append((label_of(path), n))
    if not rows:
        print("[citations] no answers to these questions", file=sys.stderr)
        return 1
    print(f"{'model':<34} {'n':>4} {'cites':>6} {'on':>5} {'off':>5} {'near':>5} "
          f"{'code':>5} {'far':>5} {'only-off':>8}   off among correct/partial/wrong")
    for label, n in sorted(rows):
        among = " / ".join(f"{n[f'off-{g}']}/{n[f'n-{g}']}"
                           for g in ("correct", "partial", "wrong"))
        print(f"{label:<34} {n['n']:>4} {n['cites']:>6} {n['on']:>5} {n['off']:>5} "
              f"{n['near']:>5} {n['code']:>5} {n['far']:>5} {n['only-off']:>8}   {among}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
