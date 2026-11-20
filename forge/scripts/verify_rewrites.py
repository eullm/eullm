#!/usr/bin/env python3
"""How wrong is the memory corpus? Checked by models whose own error is measured.

No lawyer at hand (2026-11-14), so the corpus is checked by language models,
and each checker is first measured on errors we put in ourselves:

    # 1. a sample of the corpus, and for each row copies made wrong on purpose
    python forge/scripts/verify_rewrites.py prepare \\
        --norms $WORK/norms/legislazione_*.chunks.jsonl \\
        --rows $WORK/eullm_runs/memory/program.jsonl $WORK/eullm_runs/memory/rewrites*.jsonl \\
        --per-form 60 --out $WORK/eullm_runs/verify/items.jsonl
    # 2. one checker model reads every item (one job a checker, resumable)
    python forge/scripts/verify_rewrites.py check --items .../items.jsonl \\
        --serve <checker.gguf> --gpus 0,1,2 --label q30b --out .../verdicts-q30b.jsonl
    # 3. what each checker catches, and what the corpus is estimated to hold
    python forge/scripts/verify_rewrites.py report --items .../items.jsonl .../verdicts-*.jsonl

The whole corpus, not a sample (2026-11-15: 8% of the sample flagged by every
checker, and of sixteen of those read twelve were wrong): ``prepare --all``
writes one item per row, no planted copies, its id the row's file and line;
``check --shard i/n`` takes one part of them, so several jobs share a checker;
``filter`` writes the row files again without the rows every named checker
flags:

    python forge/scripts/verify_rewrites.py prepare --all --norms ... \
        --rows $WORK/eullm_runs/memory-v2/rewrites*.jsonl --out .../all/items.jsonl
    python forge/scripts/verify_rewrites.py check --items .../all/items.jsonl --shard 0/4 ...
    python forge/scripts/verify_rewrites.py filter --items .../all/items.jsonl \
        --checkers gemma4-31b gpt-oss-120b --rows $WORK/eullm_runs/memory-v2/rewrites*.jsonl \
        --out-dir $WORK/eullm_runs/memory-v3 .../all/verdicts-*.jsonl

The wrong copies change one thing a program can change and a reader would
call an error: a number ("trenta" to "sessanta", 30 to 60), the modal ("può"
to "deve"), a negation (added or taken away), the party (creditore and
debitore, locatore and conduttore, ...). On them the right verdict is known,
so each checker's recall is measured, by kind of error. On the corpus rows the
verdict is not known: the share a checker flags, read against its recall and
its flags on rows that are only rephrased, estimates the corpus's error
rate; rows that every checker flags are listed (ids only) to be dropped.

A third, model-free signal is in the report: rows whose modal or negation
differs from every sentence of their article ("deve" where the article only
says "può").

Counts and ids only are printed; the items file holds the texts, on $WORK.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import urllib.request
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NUMBER_WORDS = ["due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci",
                "quindici", "venti", "trenta", "quaranta", "sessanta", "novanta", "cento"]
PARTIES = [("creditore", "debitore"), ("locatore", "conduttore"), ("venditore", "compratore"),
           ("attore", "convenuto"), ("datore di lavoro", "lavoratore"),
           ("mandante", "mandatario"), ("donante", "donatario"), ("imputato", "persona offesa"),
           ("pubblico ministero", "giudice"), ("usufruttuario", "proprietario")]
MODALS = [("può", "deve"), ("possono", "devono"), ("è facoltà", "è obbligo")]

CHECK_SYSTEM = (
    "Sei un controllore di testi giuridici. Ti viene dato il testo di un articolo di legge "
    "e un'affermazione scritta su quell'articolo. Devi dire se l'affermazione dice solo "
    "ciò che l'articolo dice, senza cambiarne il senso: stessi soggetti, stessi obblighi e "
    "facoltà, stessi termini, stesse pene, stesse condizioni. Una riformulazione con parole "
    "diverse è fedele; un dettaglio cambiato, aggiunto o negato non lo è. Rispondi con una "
    "prima riga «VERDETTO: FEDELE» oppure «VERDETTO: NON FEDELE», poi una riga con il motivo.")


# ------------------------------------------------------------------ mutations
def _swap_once(text: str, a: str, b: str) -> str | None:
    """``text`` with the first whole-word ``a`` turned into ``b``, or None."""
    m = re.search(rf"(?i)\b{re.escape(a)}\b", text)
    if not m:
        return None
    rep = b if text[m.start()].islower() else b[:1].upper() + b[1:]
    return text[:m.start()] + rep + text[m.end():]


def mutations(text: str, rng: random.Random) -> dict[str, str]:
    """Wrong copies of ``text``, at most one of each kind that applies."""
    out: dict[str, str] = {}
    # not the article's own number ("art. 1"): changing it names another article
    own = re.compile(r"(?i)\b(?:art\.?|articol[oi])\s*$")
    digits = [m for m in re.finditer(r"\b(\d{1,3})\b", text)
              if not own.search(text[max(0, m.start() - 12):m.start()])]
    words = [w for w in NUMBER_WORDS if re.search(rf"\b{w}\b", text)]
    if digits:
        m = rng.choice(digits)
        n = int(m.group(1))
        out["numero"] = text[:m.start()] + str(n * 2 if n < 500 else n + 7) + text[m.end():]
    elif words:
        w = rng.choice(words)
        other = NUMBER_WORDS[(NUMBER_WORDS.index(w) + 3) % len(NUMBER_WORDS)]
        out["numero"] = _swap_once(text, w, other)
    for a, b in MODALS:
        for x, y in ((a, b), (b, a)):
            s = _swap_once(text, x, y)
            if s:
                out["modale"] = s
                break
        if "modale" in out:
            break
    m = re.search(r"(?i)\bnon\s+", text)
    if m:
        out["negazione"] = text[:m.start()] + text[m.end():]
    else:
        m = re.search(r"\b(è|sono|può|deve|possono|devono|si applica|si applicano)\b", text)
        if m:
            out["negazione"] = text[:m.start()] + "non " + text[m.start():]
    for a, b in PARTIES:
        for x, y in ((a, b), (b, a)):
            s = _swap_once(text, x, y)
            if s:
                out["parte"] = s
                break
        if "parte" in out:
            break
    return {k: v for k, v in out.items() if v and v != text}


def lexical_flags(text: str, article: str) -> list[str]:
    """Modal or negation of ``text`` that no sentence of ``article`` supports."""
    flags = []
    low, art = text.lower(), article.lower()
    for a, b in MODALS:
        for x, y in ((a, b), (b, a)):
            if re.search(rf"\b{x}\b", low) and not re.search(rf"\b{x}\b", art) \
                    and re.search(rf"\b{y}\b", art):
                flags.append(f"{x} dove l'articolo dice {y}")
    if re.search(r"\bnon\b", art) is None and re.search(r"\bnon\b", low):
        flags.append("negazione assente nell'articolo")
    return flags


# ------------------------------------------------------------------ prepare
def read_rows(paths: list[Path]) -> list[dict]:
    """The rows that name an article, each with ``_src``: its file and line."""
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("text") and r.get("code") and r.get("articolo"):
                    rows.append({**r, "_src": f"{Path(p).name}:{n}"})
    return rows


def _articles(norms: list[Path]):
    from eullm_forge.eval.norm_exam import articles_from_records

    records = []
    for p in norms:
        with open(p, encoding="utf-8") as f:
            records += [json.loads(x) for x in f if x.strip()]
    return articles_from_records(records)


def prepare_all(norms: list[Path], row_files: list[Path]) -> list[dict]:
    """One item per row of ``row_files`` whose article is in the norms, nothing planted."""
    arts = _articles(norms)
    return [{"id": r["_src"], "kind": "originale", "forma": r.get("forma") or "?",
             "code": r["code"], "articolo": r["articolo"], "source": r.get("source", ""),
             "article": arts[(r["code"], r["articolo"])].text, "text": r["text"]}
            for r in read_rows(row_files) if (r["code"], r["articolo"]) in arts]


def in_shard(item_id: str, shard: str) -> bool:
    """Whether ``item_id`` belongs to ``shard`` ("i/n"; "" is every item)."""
    if not shard:
        return True
    i, n = (int(x) for x in shard.split("/"))
    return zlib.crc32(item_id.encode()) % n == i


def flagged_by_all(verdict_files: list[Path], checkers: list[str]) -> set[str]:
    """Ids every one of ``checkers`` judged "non fedele"."""
    seen: dict[str, dict[str, str]] = defaultdict(dict)
    for p in verdict_files:
        for line in p.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("checker") in checkers and r.get("verdict"):
                seen[r["id"]][r["checker"]] = r["verdict"]
    return {i for i, v in seen.items()
            if all(v.get(c) == "non fedele" for c in checkers)}


def filter_rows(row_files: list[Path], flagged: set[str], out_dir: Path) -> list[str]:
    """Each row file written again into ``out_dir`` without the flagged rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for p in row_files:
        dst = out_dir / p.name
        if dst.resolve() == Path(p).resolve():
            raise SystemExit(f"{dst} is the input: give another --out-dir")
        kept, dropped = 0, Counter()
        with open(p, encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
            for n, line in enumerate(f, 1):
                if f"{p.name}:{n}" in flagged:
                    try:
                        dropped[json.loads(line).get("forma", "?")] += 1
                    except json.JSONDecodeError:
                        dropped["?"] += 1
                    continue
                out.write(line)
                kept += 1
        lines.append(f"{p.name}: {kept:,} kept, {sum(dropped.values()):,} dropped ("
                     + ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())) + ")")
    return lines


def prepare(norms: list[Path], row_files: list[Path], per_form: int, seed: int) -> list[dict]:
    arts = _articles(norms)
    by_form = defaultdict(list)
    for r in read_rows(row_files):
        if (r["code"], r["articolo"]) in arts:
            by_form[r.get("forma") or "?"].append(r)
    rng = random.Random(seed)
    items = []
    for form in sorted(by_form):
        pool = by_form[form]
        for r in rng.sample(pool, min(per_form, len(pool))):
            article = arts[(r["code"], r["articolo"])].text
            rid = r.get("id") or f"{form}|{r['code']}|{r['articolo']}"
            base = {"article": article, "code": r["code"], "articolo": r["articolo"],
                    "forma": form, "source": r.get("source", "")}
            items.append({**base, "id": f"{rid}|orig", "kind": "originale", "text": r["text"],
                          "lexical": lexical_flags(r["text"], article)})
            for kind, wrong in mutations(r["text"], rng).items():
                items.append({**base, "id": f"{rid}|{kind}", "kind": kind, "text": wrong})
    return items


# ------------------------------------------------------------------ check
def verdict_of(reply: str) -> str | None:
    """'fedele', 'non fedele', or None for a reply without a verdict."""
    m = re.search(r"(?i)verdetto\s*:\s*(non\s+fedele|fedele)", reply or "")
    if not m:
        return None
    return "non fedele" if m.group(1).lower().startswith("non") else "fedele"


def ask(url: str, msgs: list[dict], timeout: float = 600.0) -> str:
    # room for a reasoning model's thinking before its verdict (gpt-oss reasons
    # by default; llama-server returns the reasoning apart from the content)
    body = {"messages": msgs, "temperature": 0.0, "max_tokens": 1024, "repeat_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def check_messages(item: dict) -> list[dict]:
    return [{"role": "system", "content": CHECK_SYSTEM},
            {"role": "user", "content": f"--- ARTICOLO ---\n{item['article']}\n--- FINE ---\n\n"
                                        f"--- AFFERMAZIONE ---\n{item['text']}\n--- FINE ---"}]


def check(items: list[dict], urls: list[str], out: Path, label: str, ask_fn=ask,
          stop_after: float = 0, shard: str = "") -> int:
    done = set()
    if out.is_file():
        for line in out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    items = [it for it in items if in_shard(it["id"], shard)]
    todo = [it for it in items if it["id"] not in done]
    print(f"[verify] {label}: {len(items):,} items, {len(done):,} done, {len(todo):,} to do",
          flush=True)
    lock, t0, n = threading.Lock(), time.monotonic(), [0]
    with out.open("a", encoding="utf-8") as f:
        def work(k, it):
            if stop_after and time.monotonic() - t0 > stop_after:
                return
            try:
                reply = ask_fn(urls[k % len(urls)], check_messages(it))
            except Exception as exc:  # a lost request is asked again next run
                print(f"[verify] request failed: {type(exc).__name__}", flush=True)
                return
            row = {"id": it["id"], "checker": label, "verdict": verdict_of(reply),
                   "why": reply.strip().splitlines()[-1][:300] if reply.strip() else ""}
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n[0] += 1
                if n[0] % 200 == 0:
                    print(f"[verify] {n[0]:,} checked", flush=True)

        with ThreadPoolExecutor(max_workers=max(1, 8 * len(urls))) as pool:
            list(pool.map(lambda kv: work(*kv), enumerate(todo)))
    print(f"[verify] {label}: {n[0]:,} written in {time.monotonic() - t0:.0f} s", flush=True)
    return 0


# ------------------------------------------------------------------ report
def report(items: list[dict], verdict_files: list[Path]) -> str:
    by_id = {it["id"]: it for it in items}
    verdicts: dict[str, dict[str, str]] = defaultdict(dict)
    for p in verdict_files:
        for line in p.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id") in by_id and r.get("verdict"):
                verdicts[r["checker"]][r["id"]] = r["verdict"]
    lines = []
    for checker, v in sorted(verdicts.items()):
        caught, seen = Counter(), Counter()
        for i, verdict in v.items():
            seen[by_id[i]["kind"]] += 1
            caught[by_id[i]["kind"]] += verdict == "non fedele"
        rec = {k: caught[k] / seen[k] for k in seen if k != "originale"}
        mut_seen = sum(seen[k] for k in seen if k != "originale")
        recall = sum(caught[k] for k in seen if k != "originale") / mut_seen if mut_seen else 0
        flagged = caught["originale"] / seen["originale"] if seen["originale"] else 0
        lines.append(f"{checker}: recall {recall:.0%} on {mut_seen} planted errors ("
                     + ", ".join(f"{k} {x:.0%}" for k, x in sorted(rec.items()))
                     + f"); flags {flagged:.0%} of {seen['originale']} corpus rows")
        by_form = defaultdict(Counter)
        for i, verdict in v.items():
            if by_id[i]["kind"] == "originale":
                by_form[by_id[i]["forma"]][verdict] += 1
        lines.append("    flagged by form: " + ", ".join(
            f"{f} {c['non fedele']}/{sum(c.values())}" for f, c in sorted(by_form.items())))
    origs = [i for i, it in by_id.items() if it["kind"] == "originale"]
    if verdicts:
        allc = [i for i in origs if all(v.get(i) == "non fedele" for v in verdicts.values())
                and all(i in v for v in verdicts.values())]
        lines.append(f"flagged by every checker: {len(allc)} of {len(origs)} corpus rows")
        lines += [f"    {i}" for i in sorted(allc)]
    lex = [i for i in origs if by_id[i].get("lexical")]
    lines.append(f"lexical flags (no model): {len(lex)} of {len(origs)} corpus rows")
    lines += [f"    {i}: {'; '.join(by_id[i]['lexical'])}" for i in sorted(lex)]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--norms", nargs="+", type=Path, required=True)
    p.add_argument("--rows", nargs="+", type=Path, required=True)
    p.add_argument("--per-form", type=int, default=60)
    p.add_argument("--all", action="store_true",
                   help="every row, nothing planted (the whole corpus, to be filtered)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--items", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--url", action="append", default=[])
    c.add_argument("--serve", help="checker GGUF: one llama-server per --gpus entry")
    c.add_argument("--gpus", default="0", help="as law_rewrites.py: '0,1,2' or '0+1+2'")
    c.add_argument("--llama-server")
    c.add_argument("--engine", metavar="EULLM_BIN",
                   help="serve --serve with this EULLM Engine binary instead of llama-server "
                        "(its projector, if any, left on disk)")
    c.add_argument("--parallel", type=int, default=8, help="requests in flight per server")
    c.add_argument("--ctx", type=int, default=8192, help="context per request")
    c.add_argument("--stop-after", type=float, default=0)
    c.add_argument("--shard", default="", help="'i/n': only the items of part i of n")
    r = sub.add_parser("report")
    r.add_argument("--items", type=Path, required=True)
    r.add_argument("verdicts", nargs="+", type=Path)
    fl = sub.add_parser("filter")
    fl.add_argument("--checkers", nargs="+", required=True,
                    help="a row is dropped when every one of these flags it")
    fl.add_argument("--rows", nargs="+", type=Path, required=True)
    fl.add_argument("--out-dir", type=Path, required=True)
    fl.add_argument("verdicts", nargs="+", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "filter":
        flagged = flagged_by_all(a.verdicts, a.checkers)
        print(f"[verify] {len(flagged):,} rows flagged by every one of {', '.join(a.checkers)}")
        for line in filter_rows(a.rows, flagged, a.out_dir):
            print(f"[verify] {line}")
        return 0
    if a.cmd == "prepare":
        items = (prepare_all(a.norms, a.rows) if a.all
                 else prepare(a.norms, a.rows, a.per_form, a.seed))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        kinds = Counter(it["kind"] for it in items)
        print(f"[verify] {len(items):,} items -> {a.out}: " +
              ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())), flush=True)
        return 0
    items = [json.loads(x) for x in a.items.open(encoding="utf-8") if x.strip()]
    if a.cmd == "report":
        print(report(items, a.verdicts))
        return 0
    if bool(a.url) == bool(a.serve):
        ap.error("give either --url or --serve")
    import contextlib
    with contextlib.ExitStack() as stack:
        urls = list(a.url)
        if a.serve:
            from eullm_forge.eval.gguf import EullmServer, LlamaServer
            for gpu in a.gpus.split(","):
                log = a.out.with_name(f"server-{a.label}-gpu{gpu}.log")
                if a.engine:
                    srv = EullmServer(a.serve, binary=a.engine, parallel=a.parallel,
                                      ctx_per_slot=a.ctx, devices=gpu.replace("+", ","),
                                      log_path=log)
                else:
                    srv = LlamaServer(a.serve, binary=a.llama_server, parallel=a.parallel,
                                      ctx_per_slot=a.ctx, devices=gpu.replace("+", ","),
                                      log_path=log)
                urls.append(stack.enter_context(srv).url)
        return check(items, urls, a.out, a.label, stop_after=a.stop_after, shard=a.shard)


if __name__ == "__main__":
    raise SystemExit(main())
