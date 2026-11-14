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
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("text") and r.get("code") and r.get("articolo"):
                    rows.append(r)
    return rows


def prepare(norms: list[Path], row_files: list[Path], per_form: int, seed: int) -> list[dict]:
    from eullm_forge.eval.norm_exam import articles_from_records

    records = []
    for p in norms:
        with open(p, encoding="utf-8") as f:
            records += [json.loads(x) for x in f if x.strip()]
    arts = articles_from_records(records)
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
    body = {"messages": msgs, "temperature": 0.0, "max_tokens": 1024,
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
          stop_after: float = 0) -> int:
    done = set()
    if out.is_file():
        for line in out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
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
    c.add_argument("--stop-after", type=float, default=0)
    r = sub.add_parser("report")
    r.add_argument("--items", type=Path, required=True)
    r.add_argument("verdicts", nargs="+", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "prepare":
        items = prepare(a.norms, a.rows, a.per_form, a.seed)
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
            from eullm_forge.eval.gguf import LlamaServer
            for gpu in a.gpus.split(","):
                srv = LlamaServer(a.serve, binary=a.llama_server, parallel=8, ctx_per_slot=8192,
                                  devices=gpu.replace("+", ","),
                                  log_path=a.out.with_name(f"server-{a.label}-gpu{gpu}.log"))
                urls.append(stack.enter_context(srv).url)
        return check(items, urls, a.out, a.label, stop_after=a.stop_after)


if __name__ == "__main__":
    raise SystemExit(main())
