#!/usr/bin/env python3
"""The teacher-written part of the memory corpus: every article rewritten, checked by program.

    python forge/scripts/law_rewrites.py --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --exclude $WORK/eval/closed-book.jsonl \\
        --serve $WORK/gguf/qwen3-30b-a3b-instruct-2507/qwen3-30b-a3b-instruct-2507-q8_0.gguf \\
        --gpus 0,1,2 --out $WORK/eullm_runs/memory/rewrites.jsonl

Each article in force is rewritten in the styles of
`eullm_forge.datasets.law_corpus.DEFAULT_STYLES` (``--samples`` times each), all
articles in one style before the next, so a run cut short still covers every
article. A rewrite is kept only if `check_rewrite` passes it: it names the
article and adds no other article, number, deadline or penalty. Refusals go
to ``<out>.rejects.jsonl`` with their reason. No question-and-answer rewrite
is written for an article an evaluation set (``--exclude``) asks about.

Resumable and safe to chain, like cds_schede.py: what is in this output, a
sibling (``rewrites*.jsonl`` beside it) or the rejects of the current checks
is not asked again; ``--reverse`` takes the list from the other end and ``--start 0.5``
begins halfway, so up to four chains share it without asking the same
article twice, re-reading each other's files every five minutes; and
``--stop-after`` stops taking new articles in time for a two-hour link.
Prints counts and throughput, never a text.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.law_corpus import (  # noqa: E402
    CHECKS_VERSION,
    DEFAULT_STYLES,
    PROMPT_VERSION,
    STYLES,
    by_code,
    check_rewrite,
    clean_reply,
    excluded,
    rewritable,
    rewrite_messages,
)
from eullm_forge.eval.norm_exam import articles_from_records  # noqa: E402


def load_articles(paths: list[Path]):
    records = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            records += [json.loads(line) for line in f if line.strip()]
    return articles_from_records(records)


def ask(url: str, msgs: list[dict], max_tokens: int, temperature: float, seed: int,
        timeout: float = 600.0) -> dict:
    body = {"messages": msgs, "temperature": temperature, "max_tokens": max_tokens,
            "seed": seed, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def siblings(out: Path) -> list[Path]:
    """This output and every other of its family beside it, rejects included."""
    stem = out.name.split(".")[0].split("-")[0]
    return sorted(out.parent.glob(f"{stem}*.jsonl")) + [out]


def done_ids(*paths: Path) -> set[str]:
    """Rewrites kept, or refused under the current checks."""
    out: set[str] = set()
    for p in paths:
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "reason" in row and row.get("v", 1) < CHECKS_VERSION:
                        continue
                    out.add(row["id"])
    return out


def work_list(articles, styles, samples: int, asked: set[tuple[str, str]]):
    """(id, article, style, sample) in the order they are asked: style by style."""
    arts = [a for code_arts in by_code(articles).values() for a in code_arts if rewritable(a)]
    todo = []
    for k in range(samples):
        for style in styles:
            for a in arts:
                if style == "domande" and (a.code, a.number.lower()) in asked:
                    continue
                todo.append((f"{a.code}|{a.number}|{style}|{k}", a, style, k))
    return todo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norms", nargs="+", type=Path, required=True)
    ap.add_argument("--exclude", nargs="*", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--styles", default=",".join(DEFAULT_STYLES),
                    help=f"comma-separated, of {', '.join(STYLES)}")
    ap.add_argument("--samples", type=int, default=1, help="rewrites per article and style")
    ap.add_argument("--codes", default="", help="comma-separated codes (default: all)")
    ap.add_argument("--reverse", action="store_true", help="take the list from the other end")
    ap.add_argument("--start", type=float, default=0.0,
                    help="begin this far into the list (0-1), so more chains share it")
    ap.add_argument("--url", action="append", default=[])
    ap.add_argument("--serve", help="teacher GGUF: start one llama-server per --gpus entry")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--llama-server")
    ap.add_argument("--parallel", type=int, default=16, help="requests in flight per server")
    ap.add_argument("--ctx", type=int, default=6144, help="context per request (tokens)")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--teacher", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stop-after", type=float, default=0)
    args = ap.parse_args(argv)
    if bool(args.url) == bool(args.serve):
        ap.error("give either --url or --serve")
    styles = [s for s in args.styles.split(",") if s]
    if not styles or set(styles) - set(STYLES):
        ap.error(f"styles are {', '.join(STYLES)}")

    articles = load_articles(args.norms)
    if args.codes:
        keep = set(args.codes.split(","))
        articles = {k: a for k, a in articles.items() if a.code in keep}
    _, asked = excluded([p for p in args.exclude if p.is_file()])
    todo = work_list(articles, styles, args.samples, asked)
    if args.reverse:
        todo.reverse()
    cut = int(len(todo) * min(max(args.start, 0.0), 1.0))
    todo = todo[cut:] + todo[:cut]
    rejects_path = args.out.with_name(args.out.name.replace(".jsonl", "") + ".rejects.jsonl")
    skip = done_ids(*siblings(args.out), rejects_path)
    total = len(todo)
    todo = [t for t in todo if t[0] not in skip]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[rewrites] {total:,} rewrites asked ({len({t[1].code for t in todo})} codes, "
          f"{', '.join(styles)} x {args.samples}), {total - len(todo):,} done already, "
          f"{len(todo):,} to do", flush=True)
    if not todo:
        print("[rewrites] nothing left to do", flush=True)
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
            print(f"[rewrites] {len(urls)} teacher server(s) up after "
                  f"{time.monotonic() - t0:.0f} s", flush=True)
        teacher = args.teacher or Path(args.serve or "").stem or "teacher"
        out = stack.enter_context(args.out.open("a", encoding="utf-8"))
        rej = stack.enter_context(rejects_path.open("a", encoding="utf-8"))
        t_start = time.monotonic()

        def work(n, item):
            key, a, style, k = item
            try:
                reply = ask(urls[n % len(urls)], rewrite_messages(a, style), args.max_tokens,
                            args.temperature, zlib.crc32(key.encode()))
            except (urllib.error.URLError, OSError, ValueError) as e:
                return item, None, "request:" + type(e).__name__, {}
            choice = (reply.get("choices") or [{}])[0]
            text = clean_reply((choice.get("message") or {}).get("content") or "")
            reason = ("truncated" if choice.get("finish_reason") == "length"
                      else check_rewrite(a, style, text))
            return item, text, reason, reply.get("usage") or {}

        def record(item, text, reason, usage):
            key, a, style, k = item
            with lock:
                stats["prompt"] += usage.get("prompt_tokens", 0)
                stats["completion"] += usage.get("completion_tokens", 0)
                if reason:
                    stats["rejected"] += 1
                    stats["why:" + reason] += 1
                    if not reason.startswith("request"):    # asked again next run
                        rej.write(json.dumps({"id": key, "reason": reason,
                                              "v": CHECKS_VERSION}) + "\n")
                        rej.flush()
                    return
                stats["kept"] += 1
                out.write(json.dumps({"id": key, "code": a.code, "articolo": a.number,
                                      "forma": style, "sample": k, "source": "teacher",
                                      "pv": PROMPT_VERSION,
                                      "teacher": teacher, "text": text},
                                     ensure_ascii=False) + "\n")
                out.flush()
                if stats["kept"] % 500 == 0:
                    report(stats, t_start)

        in_flight = max(1, args.parallel * len(urls))
        with ThreadPoolExecutor(max_workers=in_flight) as pool:
            pending: set = set()
            it = iter(enumerate(todo))
            stopped = False
            others, refreshed = set(), time.monotonic()
            while True:
                if time.monotonic() - refreshed > 300:      # what a sibling chain did since
                    others = done_ids(*[p for p in siblings(args.out)
                                        if p not in (args.out, rejects_path)])
                    refreshed = time.monotonic()
                while not stopped and len(pending) < in_flight:
                    if args.stop_after and time.monotonic() - t0 > args.stop_after:
                        stopped = True
                        print("[rewrites] time is up: finishing what is in flight", flush=True)
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
    print(f"[rewrites] {stats['kept']:,} kept, {stats['rejected']:,} refused"
          + (f" ({reasons})" if reasons else "") + f" -> {args.out}", flush=True)
    return 0


def report(stats: Counter, t_start: float) -> None:
    dt = max(1e-6, time.monotonic() - t_start)
    n = stats["kept"] + stats["rejected"]
    print(f"[rewrites] {stats['kept']:,} kept in {dt / 60:.1f} min: {n / dt * 60:.0f}/min, "
          f"prompt {stats['prompt'] / dt:,.0f} tok/s, generated {stats['completion'] / dt:,.0f} "
          f"tok/s, refused {stats['rejected']:,}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
