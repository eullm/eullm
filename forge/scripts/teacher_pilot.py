#!/usr/bin/env python3
"""Which teacher writes cases and explanations that the checkers pass? A pilot.

    # 1. the pilot's rewrites, one file a teacher and style, beside Qwen3-30B's
    #    on the same articles (the teacher of the memory corpus)
    python forge/scripts/teacher_pilot.py collect --pilot-dir $WORK/eullm_runs/teacher-pilot \\
        --pool $WORK/eullm_runs/memory-v2/rewrites*.jsonl
    # 2. verify_rewrites.py prepare --all over <pilot-dir>/all/*.jsonl, then one
    #    check a checker (sbatch_verify_rewrites.slurm, VR_ITEMS=<pilot-dir>/items.jsonl)
    # 3. the rates
    python forge/scripts/teacher_pilot.py report --items <pilot-dir>/items.jsonl \\
        <pilot-dir>/verdicts-*.jsonl

The memory corpus's teacher, Qwen3-30B-A3B, wrote cases that both checkers
flag 42% of the time and explanations 77% (2026-11-17), and of eight cases
read five were wrong in law. The pilot asks the same articles of bigger
teachers (law_rewrites.py --limit, so every teacher gets the first articles
of the same list), and each teacher is judged only by the checkers that are
not itself: a model passes its own mistakes. Qwen3-30B's rows on the same
articles are judged by the same checkers as each teacher, so every
comparison is on one pair of judges. Counts only, never a text.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# a teacher's directory name in the pilot -> its label as a checker
SELF = {"gemma": "gemma4-31b", "gptoss": "gpt-oss-120b", "flashnext": "flashnext",
        "qwen38": "qwen38-27b"}     # Qwen3.8-27B: a teacher only, judged by all three
BASE = "qwen30b"


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def collect(pilot_dir: Path, pool: list[Path]) -> dict[str, int]:
    """<pilot-dir>/all/<teacher>-<style>.jsonl for every pilot file, and
    qwen30b-<style>.jsonl with the pool's first row for each (code, article,
    style) any pilot teacher wrote. Returns rows written by file name."""
    out = pilot_dir / "all"
    out.mkdir(parents=True, exist_ok=True)
    keys: set[tuple[str, str, str]] = set()
    written: dict[str, int] = {}
    for teacher in sorted(SELF):
        for src in sorted((pilot_dir / teacher).glob("rewrites-*.jsonl")):
            if src.name.endswith(".rejects.jsonl"):
                continue
            style = src.name.removeprefix("rewrites-").removesuffix(".jsonl")
            rows = read(src)
            keys |= {(r["code"], str(r["articolo"]), r["forma"]) for r in rows}
            name = f"{teacher}-{style}.jsonl"
            (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                            for r in rows), encoding="utf-8")
            written[name] = len(rows)
    base: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for p in pool:
        if p.name.endswith(".rejects.jsonl"):
            continue
        for r in read(p):
            k = (r.get("code"), str(r.get("articolo")), r.get("forma"))
            if k in keys and k not in base[k[2]]:
                base[k[2]][k] = r
    for style, rows in sorted(base.items()):
        name = f"{BASE}-{style}.jsonl"
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                        for r in rows.values()), encoding="utf-8")
        written[name] = len(rows)
    return written


def teacher_of(item_id: str) -> str:
    return item_id.split(":", 1)[0].split("-", 1)[0]


def report(items: list[dict], verdict_files: list[Path]) -> str:
    by_id = {it["id"]: it for it in items}
    verdicts: dict[str, dict[str, str]] = defaultdict(dict)
    for p in verdict_files:
        for r in read(p):
            if r.get("id") in by_id and r.get("verdict"):
                verdicts[r["id"]][r["checker"]] = r["verdict"]
    checkers = sorted({c for v in verdicts.values() for c in v})

    def rate(teacher: str, style: str, judges: list[str]) -> tuple[int, int]:
        """(flagged by every judge, judged by every judge) for a teacher's style."""
        flagged = n = 0
        for i, v in verdicts.items():
            if teacher_of(i) != teacher or by_id[i]["forma"] != style:
                continue
            if all(j in v for j in judges):
                n += 1
                flagged += all(v[j] == "non fedele" for j in judges)
        return flagged, n

    styles = sorted({it["forma"] for it in items})
    teachers = sorted({teacher_of(i) for i in by_id} - {BASE})
    lines = [f"checkers: {', '.join(checkers)}"]
    for t in teachers:
        judges = [c for c in checkers if c != SELF.get(t)]
        lines.append(f"{t}, judged by {' + '.join(judges)}:")
        for s in styles:
            f, n = rate(t, s, judges)
            bf, bn = rate(BASE, s, judges)
            if not n:
                continue
            lines.append(f"    {s:12s} {t}: {f}/{n} flagged by all ({f / n:.0%})   "
                         f"{BASE} on the same articles: {bf}/{bn} "
                         f"({bf / bn:.0%})" if bn else f"    {s:12s} {t}: {f}/{n} ({f / n:.0%})")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--pilot-dir", type=Path, required=True)
    c.add_argument("--pool", nargs="+", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--items", type=Path, required=True)
    r.add_argument("verdicts", nargs="+", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        for name, n in sorted(collect(a.pilot_dir, a.pool).items()):
            print(f"[pilot] {name}: {n} rows", flush=True)
        return 0
    print(report(read(a.items), a.verdicts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
