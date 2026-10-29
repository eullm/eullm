#!/usr/bin/env python3
"""The case-law part of the memory corpus: every card's principles rewritten, checked by program.

    python forge/scripts/principle_rewrites.py \\
        --cards $WORK/eullm_runs/cds/schede*.jsonl $WORK/eullm_runs/cass/schede-civ*.jsonl \\
        --serve $WORK/gguf/qwen3-30b-a3b-instruct-2507/qwen3-30b-a3b-instruct-2507-q8_0.gguf \\
        --gpus 0,1,2 --out $WORK/eullm_runs/memory/principi.jsonl

One request per card (`eullm_forge.caselaw.rewrites`): each principle in
other words and as a question with its answer, one row per principle and
form, refused by program if it adds an article, a number or a trace of the
case. Only the cards that export_clean_cards.py would let leave are asked
(``why_not``: no special category, no criminal matter, no party, place or
amount); ``--courts`` names the courts, the Corte di cassazione penale only
when asked for.

The cards are shuffled once (fixed seed), so a run cut short covers the
courts and years evenly. Resumable and chained like law_rewrites.py:
siblings (``principi*.jsonl``), ``--reverse``, ``--start``,
``--stop-after``. Rows carry the ruling id, so they stay on the cluster like
the cards; the ruling number is never in a rewrite. Prints counts only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import threading
import time
import urllib.error
import zlib
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_clean_cards import why_not  # noqa: E402
from law_rewrites import ask, siblings  # noqa: E402

from eullm_forge.caselaw.rewrites import PROMPT_VERSION, messages, parse  # noqa: E402


def done_reqs(*paths: Path) -> set[str]:
    """Requests answered (kept or refused) under the current version."""
    out: set[str] = set()
    for p in paths:
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("pv", 1) >= PROMPT_VERSION and row.get("req"):
                            out.add(row["req"])
    return out


def load_cards(paths: list[Path], courts: set[str]) -> tuple[list[dict], Counter]:
    """The cards that may be asked, and why the others are not."""
    cards, skipped, seen = [], Counter(), set()
    for p in paths:
        # the refusals, and the export with made-up references, are not cards
        if p.name.endswith(".rejects.jsonl") or "clean" in p.name:
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                court = row.get("court", "cds")
                if (court not in courts or not row.get("id") or row["id"] in seen
                        or not row.get("principi")):
                    continue
                seen.add(row["id"])
                why = why_not(row, court)
                if why:
                    skipped[why] += 1
                else:
                    cards.append(row)
    return cards, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards", nargs="+", type=Path, required=True)
    ap.add_argument("--courts", default="cds,snciv", help="cds, snciv, snpen")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--url", action="append", default=[])
    ap.add_argument("--serve")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--llama-server")
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=6144)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--teacher", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stop-after", type=float, default=0)
    args = ap.parse_args(argv)
    if bool(args.url) == bool(args.serve):
        ap.error("give either --url or --serve")

    cards, skipped = load_cards(args.cards, set(args.courts.split(",")))
    random.Random(0).shuffle(cards)
    todo = [(f"{c['id']}|{k}", c, k) for k in range(args.samples) for c in cards]
    if args.reverse:
        todo.reverse()
    cut = int(len(todo) * min(max(args.start, 0.0), 1.0))
    todo = todo[cut:] + todo[:cut]
    rejects_path = args.out.with_name(args.out.name.replace(".jsonl", "") + ".rejects.jsonl")
    skip = done_reqs(*siblings(args.out), rejects_path)
    total = len(todo)
    todo = [t for t in todo if t[0] not in skip]
    if args.limit:
        todo = todo[:args.limit]
    by_court = Counter(c.get("court") for c in cards)
    courts = ", ".join(f"{k} {v:,}" for k, v in sorted(by_court.items()))
    out_why = ", ".join(f"{k} {v:,}" for k, v in skipped.most_common())
    print(f"[principi] {total:,} cards asked ({courts}), "
          f"{sum(skipped.values()):,} left out ({out_why}), "
          f"{total - len(todo):,} done already, {len(todo):,} to do", flush=True)
    if not todo:
        print("[principi] nothing left to do", flush=True)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    stats: Counter = Counter()
    lock = threading.Lock()
    with contextlib.ExitStack() as stack:
        urls = list(args.url)
        if args.serve:
            from eullm_forge.eval.gguf import LlamaServer
            for gpu in args.gpus.split(","):
                srv = LlamaServer(args.serve, binary=args.llama_server, parallel=args.parallel,
                                  ctx_per_slot=args.ctx, devices=gpu,
                                  log_path=args.out.with_name(f"server-gpu{gpu}.log"))
                urls.append(stack.enter_context(srv).url)
            print(f"[principi] {len(urls)} teacher server(s) up after "
                  f"{time.monotonic() - t0:.0f} s", flush=True)
        teacher = args.teacher or Path(args.serve or "").stem or "teacher"
        out = stack.enter_context(args.out.open("a", encoding="utf-8"))
        rej = stack.enter_context(rejects_path.open("a", encoding="utf-8"))
        t_start = time.monotonic()

        def work(n, item):
            req, card, k = item
            try:
                reply = ask(urls[n % len(urls)], messages(card), args.max_tokens,
                            args.temperature, zlib.crc32(req.encode()))
            except (urllib.error.URLError, OSError, ValueError) as e:
                return item, [], [(-1, "request:" + type(e).__name__)], {}
            choice = (reply.get("choices") or [{}])[0]
            if choice.get("finish_reason") == "length":
                return item, [], [(-1, "truncated")], reply.get("usage") or {}
            kept, refused = parse(card, (choice.get("message") or {}).get("content") or "")
            return item, kept, refused, reply.get("usage") or {}

        def record(item, kept, refused, usage):
            req, card, k = item
            with lock:
                stats["prompt"] += usage.get("prompt_tokens", 0)
                stats["completion"] += usage.get("completion_tokens", 0)
                stats["cards"] += 1
                for i, reason in refused:
                    stats["refused"] += 1
                    stats["why:" + reason] += 1
                    if not reason.startswith("request"):     # asked again next run
                        rej.write(json.dumps({"req": req, "principio": i, "reason": reason,
                                              "pv": PROMPT_VERSION}) + "\n")
                if refused:
                    rej.flush()
                for i, form, text in kept:
                    stats["kept"] += 1
                    out.write(json.dumps({
                        "id": f"{req}|{i}|{form}", "req": req, "card": card["id"],
                        "court": card.get("court"), "materia": card.get("materia"),
                        "principio": i, "forma": form, "sample": k, "pv": PROMPT_VERSION,
                        "teacher": teacher, "text": text}, ensure_ascii=False) + "\n")
                if kept:
                    out.flush()
                if stats["cards"] % 1000 == 0:
                    report(stats, t_start)

        in_flight = max(1, args.parallel * len(urls))
        with ThreadPoolExecutor(max_workers=in_flight) as pool:
            pending: set = set()
            it = iter(enumerate(todo))
            stopped = False
            others, refreshed = set(), time.monotonic()
            while True:
                if time.monotonic() - refreshed > 300:      # what a sibling chain did since
                    others = done_reqs(*[p for p in siblings(args.out)
                                         if p not in (args.out, rejects_path)])
                    refreshed = time.monotonic()
                while not stopped and len(pending) < in_flight:
                    if args.stop_after and time.monotonic() - t0 > args.stop_after:
                        stopped = True
                        print("[principi] time is up: finishing what is in flight", flush=True)
                        break
                    nxt = next(it, None)
                    while nxt is not None and nxt[1][0] in others:
                        nxt = next(it, None)
                    if nxt is None:
                        stopped = True
                        break
                    pending.add(pool.submit(work, *nxt))
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    record(*fut.result())
        report(stats, t_start)
    reasons = ", ".join(f"{k[4:]} {v}" for k, v in sorted(stats.items()) if k.startswith("why:"))
    print(f"[principi] {stats['cards']:,} cards, {stats['kept']:,} rows kept, "
          f"{stats['refused']:,} refused" + (f" ({reasons})" if reasons else "")
          + f" -> {args.out}", flush=True)
    return 0


def report(stats: Counter, t_start: float) -> None:
    dt = max(1e-6, time.monotonic() - t_start)
    print(f"[principi] {stats['cards']:,} cards in {dt / 60:.1f} min: "
          f"{stats['cards'] / dt * 60:.0f}/min, rows kept {stats['kept']:,}, refused "
          f"{stats['refused']:,}, prompt {stats['prompt'] / dt:,.0f} tok/s, generated "
          f"{stats['completion'] / dt:,.0f} tok/s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
