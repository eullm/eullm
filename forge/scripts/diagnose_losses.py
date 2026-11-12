#!/usr/bin/env python3
"""Why a model loses to another: its losses tagged by kind of error, counted.

    python forge/scripts/diagnose_losses.py \\
        $WORK/eval/devbig-graded-v3/answers-r2l-replay-g2-q4-hyb.graded.jsonl \\
        --baseline $WORK/eval/devbig-graded-v3/answers-r2l-steps-8b-quote-q4-hyb.graded.jsonl \\
        [--dump losses.txt]

A paired comparison says how many questions a model loses; this says what
the losses are, so that the next run treats a cause and not a total
(2026-11-12: a month of variants read on totals alone). For every question
the baseline got right and the model did not ("losses", and with --wins the
other way round), the model's answer is tagged by program:

* ``vuota``: no answer at all;
* ``si astiene``: it says the texts do not answer, the baseline did not;
* ``articolo non dato``: it cites an article neither the question nor the
  reference names (one from memory, or from another code);
* ``nessun termine`` / ``altro termine dell'articolo`` / ``termine non
  nell'articolo``: on deadline questions, what the deadline it gives is;
* ``piu' corta``: under 60% of the baseline's length (an answer that stops
  early: one deadline of several);
* ``altro``: none of these, to be read.

An answer may carry several tags. Counts only, by tag and by question kind,
so it runs on held-out answers too; ``--dump`` writes the questions with
both answers and the judge's reasons for reading, and refuses a held-out
exam's answers (a path under ``norm-exam-answers``), as compare_graded.py
does.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval.abstain import abstained, cited_articles  # noqa: E402
from eullm_forge.eval.norm_exam import all_deadlines  # noqa: E402
from eullm_forge.eval.paired import _RUBRIC_DEADLINE, kind_of  # noqa: E402

HELD_OUT = "norm-exam-answers"


def read(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f))}


def right(row: dict) -> bool:
    return (row.get("grade") or "").lower() in ("correct", "corretto", "corretta")


def tags(row: dict, base: dict) -> list[str]:
    """The program's reading of why ``row`` is wrong where ``base`` is right."""
    from eullm_forge.eval.metrics import DEADLINE_UNITS
    from eullm_forge.rl.rewards import mentioned_deadlines

    answer, banswer = row.get("answer") or "", base.get("answer") or ""
    if not answer.strip():
        return ["vuota"]
    out = []
    if abstained(answer) and not abstained(banswer):
        out.append("si astiene")
    reference = row.get("reference") or ""
    given = cited_articles(row.get("question") or "") | cited_articles(reference)
    if cited_articles(answer) - given:
        out.append("articolo non dato")
    if kind_of(str(row.get("id", ""))) in ("termine", "termine_argomento"):
        named = mentioned_deadlines(answer)
        article = reference.split("Testo integrale dell'articolo:", 1)[-1]
        stated = all_deadlines(article)
        m = _RUBRIC_DEADLINE.search(row.get("rubric") or "")
        wanted = ({(int(m.group(1)), DEADLINE_UNITS.get(m.group(2).lower(), m.group(2)))}
                  if m else set())
        if not named:
            out.append("nessun termine")
        elif named - stated - wanted:
            out.append("termine non nell'articolo")
        elif wanted and not named & wanted:
            out.append("altro termine dell'articolo")
    if banswer and len(answer) < 0.6 * len(banswer):
        out.append("piu' corta")
    return out or ["altro"]


def diagnose(model: dict[str, dict], base: dict[str, dict], wins: bool = False):
    """(ids, tag counts, tag counts by kind) of the losses (or the wins)."""
    ids = [i for i in sorted(model) if i in base
           and (right(model[i]) and not right(base[i]) if wins
                else right(base[i]) and not right(model[i]))]
    total, by_kind = Counter(), defaultdict(Counter)
    for i in ids:
        a, b = (base[i], model[i]) if wins else (model[i], base[i])
        for t in tags(a, b):
            total[t] += 1
            by_kind[kind_of(i)][t] += 1
    return ids, total, by_kind


def summary(rows: dict[str, dict]) -> str:
    n = len(rows) or 1
    answers = [r.get("answer") or "" for r in rows.values()]
    chars = sum(map(len, answers)) / n
    return (f"{sum(right(r) for r in rows.values())} right of {len(rows)}, "
            f"mean {chars:.0f} chars, {sum(map(abstained, answers))} abstaining, "
            f"{sum(not a.strip() for a in answers)} empty")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", type=Path, help="graded answers of the model (judge_answers.py)")
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--wins", action="store_true", help="tag the model's wins instead")
    ap.add_argument("--dump", type=Path, help="write the questions and both answers here")
    a = ap.parse_args(argv)
    if a.dump and (HELD_OUT in str(a.model) or HELD_OUT in str(a.baseline)):
        ap.error("--dump prints items: not on the held-out exam")

    model, base = read(a.model), read(a.baseline)
    print(f"model    {summary(model)}")
    print(f"baseline {summary(base)}")
    ids, total, by_kind = diagnose(model, base, a.wins)
    what = "wins (the baseline's answer tagged)" if a.wins else "losses (the model's answer tagged)"
    print(f"\n{len(ids)} {what}")
    for t, n in total.most_common():
        print(f"  {t:<30} {n}")
    for k in sorted(by_kind):
        print(f"  [{k}] " + ", ".join(f"{t} {n}" for t, n in by_kind[k].most_common()))
    if a.dump:
        sh = lambda s, n: textwrap.shorten(str(s or ""), n, placeholder=" [...]")  # noqa: E731
        with a.dump.open("w", encoding="utf-8") as f:
            for i in ids:
                m, b = model[i], base[i]
                f.write(f"== {i}  {tags(*((b, m) if a.wins else (m, b)))}\n"
                        f"D: {sh(m.get('question'), 400)}\n"
                        f"  MODELLO [{m.get('grade')}]: {sh(m.get('answer'), 900)}\n"
                        f"    giudice: {sh(m.get('why'), 300)}\n"
                        f"  BASE [{b.get('grade')}]: {sh(b.get('answer'), 900)}\n"
                        f"    giudice: {sh(b.get('why'), 300)}\n\n")
        print(f"\nwritten {a.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
