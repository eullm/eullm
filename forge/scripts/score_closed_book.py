#!/usr/bin/env python3
"""Score answers to the closed-book suite by program, by kind.

    python forge/scripts/score_closed_book.py $WORK/eval/closed-book.jsonl \\
        $WORK/eval/closed-book-answers/answers-*.jsonl

One row per model: the share right of every kind, the total, and the wrong
answers given with no doubt expressed ("sure-wrong"), which count against a
model more than an honest "I do not know"; and how often it says a REAL
article does not exist ("denies-real"): on 2026-10-27 r2q denied 91% of the
made-up articles and recognised none of the real ones, a reflex the made-up
column alone reads as knowledge. Counts only, never a question.

``--norms`` leaves out the penalty, procedibility and repeal items whose
article no longer gives that answer: the suite was drawn by the functions
that, until 2026-11-15, called an article with one comma repealed "not in
force" (art. 600 c.p.), took an aggravated range for the base penalty (art.
507 c.p.) and a general mention for the article's procedibility (art. 158
c.p.). Those items asked models to repeat our mistakes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.eval.closed_book import KINDS, score  # noqa: E402
from eullm_forge.rl.rewards import denies_existence  # noqa: E402

SHORT = {"esiste_vera": "esist", "esiste_falsa": "falso", "numero_da_rubrica": "n<-rub",
         "rubrica_da_numero": "rub<-n", "termine": "termi", "pena": "pena",
         "procedibilita": "proc", "abrogato": "abrog", "sentenza_falsa": "sent?"}


def label_of(path: Path) -> str:
    name = path.name.removesuffix(".jsonl")
    return name.removeprefix("answers-")


# An answer that points to texts it was not given: what a model trained to
# answer from the retrieved statutes says when there are none ("il testo non
# è tra quelli forniti", "non è presente nella raccolta"). The closed book
# scores it wrong, rightly, but it is not the same mistake as a wrong article:
# the continued pretraining with open-book replay (2026-11-10) scored 0% on
# terms, which only this count can tell from not knowing them.
NO_TEXT = re.compile(
    r"(?i)\b(?:test[io]|document[io]|fonti|raccolta|estratt[io]|passagg[io]|brani)\b[^.\n]{0,40}?"
    r"\b(?:fornit|disponibil|a (?:mia )?disposizione|riportat|indicat|allegat|present|citat)"
    r"|\bnon (?:è|e'|sono|risulta(?:no)?) (?:present|riportat|contenut|inclus)"
    r"|\bnon (?:ho|dispongo|mi è stato fornito)[^.\n]{0,30}?\b(?:test|document|fonti)")


def points_to_texts(answer: str) -> bool:
    """Whether the answer appeals to statute texts, given none."""
    return bool(NO_TEXT.search(answer or ""))


def stale_items(items: dict[str, dict], norms: list[Path]) -> set[str]:
    """Ids of penalty, procedibility and repeal items the current functions no
    longer draw from their article, or draw with another answer."""
    from eullm_forge.datasets.premise_gen import (
        _repealed_sentence,
        defines_offence,
        penalty_of,
        procedibility_of,
    )
    from eullm_forge.eval.norm_exam import articles_from_records

    records = []
    for p in norms:
        with p.open(encoding="utf-8") as f:
            records += [json.loads(x) for x in f if x.strip()]
    arts = articles_from_records(records)
    out = set()
    for i, it in items.items():
        m = it["metadata"]
        kind = m["tipo"]
        if kind not in ("pena", "procedibilita", "abrogato"):
            continue
        a = arts.get((m["code"], m["articolo"]))
        if a is None:
            out.add(i)
        elif kind == "abrogato":
            if not _repealed_sentence(a):
                out.add(i)
        elif not defines_offence(a.text):
            out.add(i)
        elif kind == "pena":
            p = penalty_of(a.text)
            if not p or list(m.get("high", [])) != [p[3], p[4]]:
                out.add(i)
        else:
            r = procedibility_of(a.text)
            if not r or it.get("keywords") != [r[0]]:
                out.add(i)
    return out


def summarize(items: dict[str, dict], answers: dict[str, str]) -> dict:
    by = defaultdict(Counter)
    texts = Counter()
    sure_wrong = denied = real = 0
    for i, ans in answers.items():
        if i not in items:
            continue
        kind = items[i]["metadata"]["tipo"]
        grade, confident = score(items[i], ans)
        by[kind][grade] += 1
        texts[kind] += points_to_texts(ans)
        sure_wrong += grade == "wrong" and confident
        if kind in ("esiste_vera", "rubrica_da_numero"):
            real += 1
            denied += denies_existence(ans)
    total = sum(sum(c.values()) for c in by.values())
    right = sum(c["right"] for c in by.values())
    return {"by": by, "n": total, "right": right, "sure_wrong": sure_wrong,
            "denies_real": denied / real if real else 0.0, "texts": texts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", type=Path)
    ap.add_argument("answers", nargs="+", type=Path)
    ap.add_argument("--texts", action="store_true",
                    help="instead of the grades, the share of answers by kind that point to "
                         "statute texts the question did not give (counts only)")
    ap.add_argument("--norms", nargs="+", type=Path,
                    help="leave out the items the current functions no longer draw")
    args = ap.parse_args(argv)

    with args.suite.open(encoding="utf-8") as f:
        items = {d["id"]: d for d in map(json.loads, filter(str.strip, f))}
    if args.norms:
        stale = stale_items(items, args.norms)
        left = Counter(items[i]["metadata"]["tipo"] for i in stale)
        print(f"left out: {len(stale)} items no longer drawn ("
              + ", ".join(f"{k} {n}" for k, n in sorted(left.items())) + ")")
        items = {i: it for i, it in items.items() if i not in stale}
    kinds = [k for k in KINDS if any(it["metadata"]["tipo"] == k for it in items.values())]
    if args.texts:
        print("the kind columns: answers that point to statute texts not given, by kind")
    print(f"{'model':<30} {'right':>11} {'sure-wrong':>10} {'denies-real':>11}  "
          + " ".join(f"{SHORT[k]:>6}" for k in kinds))
    for path in args.answers:
        with path.open(encoding="utf-8") as f:
            answers = {r["id"]: r.get("answer", "") for r in map(json.loads, filter(str.strip, f))}
        s = summarize(items, answers)
        if not s["n"]:
            print(f"{label_of(path):<30} no answers to this suite")
            continue
        cells = []
        for k in kinds:
            c = s["by"].get(k, Counter())
            n = sum(c.values())
            num = s["texts"][k] if args.texts else c["right"]
            cells.append(f"{100 * num / n:5.0f}%" if n else f"{'-':>6}")
        print(f"{label_of(path):<30} {s['right']:>4}/{s['n']:<4} "
              f"{100 * s['right'] / s['n']:3.0f}% {s['sure_wrong']:>9} "
              f"{100 * s['denies_real']:10.0f}%  " + " ".join(cells))
    # Articles that state their procedibility are mostly "a querela" (the
    # ex officio rule is the unwritten default), so "querela" every time scores
    # this much without knowing anything.
    proc = [it for it in items.values() if it["metadata"]["tipo"] == "procedibilita"]
    if proc:
        q = sum(it["keywords"] == ["querela"] for it in proc) / len(proc)
        print(f"procedibility: {100 * q:.0f}% of the items want \"querela\", what answering "
              "it every time scores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
