#!/usr/bin/env python3
"""AutoBench: does `"model": "auto"` send a request to the small model when
that costs nothing, and to the large one when it does not?

MVP 3 of the Reflex roadmap (docs/reflex-roadmap.md). Two models answer
every item of a labelled set; routers then decide which of the two each
request should have gone to, and the report says, per set and router:

  * how many large-model calls it avoids, and the large model's GPU time;
  * the accuracy it keeps, against always asking the large model, with a
    95% confidence interval, and the answers it loses and gains;
  * how well its score separates the requests the small model handles from
    the others (AUROC), and what deciding costs.

Reflex — the server's own router, `POST /api/route` — is compared with what
a deployment could do without it: a threshold on the request's length, the
nearest labelled requests by their embeddings, random routing at the same
share, and the oracle. If the length threshold or the embeddings avoid as
many calls at the same accuracy, Reflex is not the answer for routing (the
kill criterion); the report says so per set.

Stages, all on by default (`--stages`):

  1. `generate`: both models answer every item, each named explicitly,
     deterministically (temperature 0, top_k 1, seed 1; thinking off unless
     --think). `--answers FILE` keeps the answers: a second run reuses them.
  2. `route`: decisions only, no generation.
  3. `e2e`: `"model": "auto"` itself, on the first --e2e-limit items of each
     test half, at each --concurrency (1, 4 and 16): whether the model a
     request is routed to answers it with the text it gave when named —
     which is what makes stage 2's scores hold — and the time to the first
     token a client sees, routing and loads included.

Start a server with both models as auto candidates, a decision model and,
for the `knn` baseline, an embedding model, with an audit trail of its own:

    EULLM_AUDIT_DIR=/tmp/autobench-audit eullm serve --max-loaded-models 2 \\
      --auto-model 'qwen3-4b=Short everyday requests and simple facts' \\
      --auto-model 'qwen3-8b=Multi-step reasoning, maths, code, long answers' \\
      --decision-model jev-style-0.8b-decision-v3-gguf-q4_k_m \\
      --embedding-model qwen3-embedding-0.6b-gguf-q8_0
    python3 bench/reflexbench/autobench.py --small qwen3-4b --large qwen3-8b \\
      --embed-model qwen3-embedding-0.6b-gguf-q8_0 \\
      --sets gsm8k,arc-easy,arc-challenge,mmlu --limit 200 --out auto.json --details auto.jsonl

Only the Python standard library is needed.
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_data  # noqa: E402
import ab_grade  # noqa: E402
import ab_methods  # noqa: E402
import ab_metrics  # noqa: E402
from rb_metrics import percentile  # noqa: E402

STAGES = ("generate", "route", "e2e")


def answer_key(item, model):
    return f"{item.set}\t{item.id}\t{model}"


def answers_how(args):
    """What an answer depends on besides its item and model. A kept answer
    asked another way — thinking, a longer limit, or from the prompt cache,
    as runs before `cache_prompt` were — is asked again, not reused."""
    return dict(ab_methods.REPRODUCIBLE, think=args.think, max_tokens=args.max_tokens)


def load_answers(path, how):
    """The answers a previous run kept the way `how` says, by set, item and
    model."""
    answers = {}
    if path and pathlib.Path(path).exists():
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("how") == how:
                    answers[row["key"]] = ab_methods.Answer.from_json(row["answer"])
    return answers


def progress(n, total, started, shown):
    """Print where a long stage is, every 50 items or 30 seconds."""
    now = time.perf_counter()
    if n % 50 == 0 or n == total or now - shown >= 30:
        print(f"    {n}/{total} ({now - started:.0f} s)", file=sys.stderr, flush=True)
        return now
    return shown


def generate_all(args, datasets, answers):
    """Stage 1: both models answer every item not answered yet."""
    out = open(args.answers, "a", encoding="utf-8") if args.answers else None
    try:
        for dataset in datasets:
            for model in (args.small, args.large):
                todo = [i for i in dataset.items if answer_key(i, model) not in answers]
                print(f"  {dataset.name}: {model} answers {len(todo)}", file=sys.stderr)
                started = shown = time.perf_counter()
                for n, item in enumerate(todo, 1):
                    answer = ab_methods.generate(
                        args.url,
                        item,
                        model,
                        args.api_key,
                        args.timeout,
                        args.think,
                        args.max_tokens,
                    )
                    if item.grader != "judge":
                        answer.correct = ab_grade.correct(item, answer.text)
                    answers[answer_key(item, model)] = answer
                    if out:
                        row = {
                            "key": answer_key(item, model),
                            "how": answers_how(args),
                            "answer": answer.to_json(),
                        }
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                    shown = progress(n, len(todo), started, shown)
    finally:
        if out:
            out.close()


def judge_all(args, dataset, answers):
    """Grade the `judge` items: the small model's answer is right when the
    judge does not prefer the large one's, the large model's always."""
    for item in dataset.items:
        if item.grader != "judge":
            continue
        small = answers[answer_key(item, args.small)]
        large = answers[answer_key(item, args.large)]
        if small.correct is None:
            verdict = ab_methods.judge(
                args.url, args.judge_model, item, small.text, large.text, args.api_key, args.timeout
            )
            small.correct, large.correct = verdict != "large", True
            item.verdict = verdict


def judge_agreement(args, datasets):
    """Cohen's kappa between the judge and hand labels (--judge-labels):
    {"id": ..., "verdict": "small" | "large" | "tie"} per line."""
    if not args.judge_labels:
        return None
    labels = {}
    for line in pathlib.Path(args.judge_labels).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[str(row["id"])] = row["verdict"]
    pairs = [
        (getattr(item, "verdict", None), labels[item.id])
        for dataset in datasets
        for item in dataset.items
        if item.id in labels and getattr(item, "verdict", None)
    ]
    return {
        "items": len(pairs),
        "kappa": ab_metrics.cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]),
    }


def decide_all(router, items, label):
    """Every item's decision by `router`, after one untimed decision on the
    first, so a cold start is not counted — and which decision model
    answers, said before anything is counted."""
    if items:
        model = router.decide(items[0]).server.get("decision_model")
        if model:
            print(f"    decision model: {model}", file=sys.stderr, flush=True)
    decisions = []
    started = shown = time.perf_counter()
    for n, item in enumerate(items, 1):
        decisions.append(router.decide(item))
        shown = progress(n, len(items), started, shown)
    print(f"    {label}: {len(decisions)} decisions", file=sys.stderr, flush=True)
    return decisions


def rows_for_set(args, dataset, answers, details):
    """Stage 2 on one set: every router's decisions, scored on the test half."""
    dev, test = ab_data.split(dataset, args.seed)
    every = dev + test

    def grades(items, model):
        return [bool(answers[answer_key(i, model)].correct) for i in items]

    small_dev, large_dev = grades(dev, args.small), grades(dev, args.large)
    small_test, large_test = grades(test, args.small), grades(test, args.large)
    large_ms = [answers[answer_key(i, args.large)].total_ms for i in test]
    ok_dev = [s or not lg for s, lg in zip(small_dev, large_dev, strict=True)]
    ok_test = [s or not lg for s, lg in zip(small_test, large_test, strict=True)]
    rows = []

    def row(router, routed, scores=None, latency=None, extra=None):
        metrics = ab_metrics.summarize(
            test, routed, small_test, large_test, large_ms, scores, latency
        )
        metrics.update(extra or {})
        rows.append({"set": dataset.name, "router": router, "metrics": metrics})

    def fitted(router, dev_scores, test_scores, latency=None, extra=None):
        threshold = ab_metrics.fit_threshold(dev_scores, small_dev, large_dev)
        if threshold is None:
            return
        routed = [s >= threshold for s in test_scores]
        extra = dict(extra or {}, threshold=threshold)
        row(f"{router} (fitted)", routed, test_scores, latency, extra)

    row("always-large", [False] * len(test))
    row("always-small", [True] * len(test))
    row("oracle", ok_test)

    decisions = {}
    print(f"  {dataset.name}: reflex", file=sys.stderr, flush=True)
    reflex = ab_methods.Reflex(
        args.url, args.small, args.large, args.api_key, args.timeout, args.think, args.max_tokens
    )
    decisions["reflex"] = decide_all(reflex, every, "reflex")
    for spec in args.question_specs:
        replay = ab_methods.Replay(spec, args.url, args.small, args.api_key, args.timeout)
        decisions[replay.name] = [replay.decide(d) for d in decisions["reflex"]]
    share = None
    for name, made in decisions.items():
        dev_made, test_made = made[: len(dev)], made[len(dev) :]
        scores = [score_of(d) for d in test_made]
        latency = [d.ms for d in test_made if d.ms is not None]
        server = [d.server.get("decision_ms") for d in test_made]
        server = [ms for ms in server if isinstance(ms, (int, float))]
        reasons = {}
        for d in test_made:
            reason = d.server.get("reason") or "replay"
            reasons[reason] = reasons.get(reason, 0) + 1
        served = sorted({d.server.get("decision_model") for d in made} - {None})
        extra = {"reasons": reasons, "decision_model": ", ".join(served) or None}
        if server:
            extra["decision_ms"] = {"p50": percentile(server, 0.5), "p95": percentile(server, 0.95)}
        routed = [bool(d.small) for d in test_made]
        row(f"{name} (own)", routed, scores, latency, extra)
        fitted(name, [score_of(d) for d in dev_made], scores, latency, extra)
        if name == "reflex":
            share = sum(routed) / len(routed) if routed else 0.0

    row(
        "random (Reflex's share)",
        ab_methods.random_routing(len(test), share or 0.0, args.seed),
    )
    fitted("length", ab_methods.length_scores(dev), ab_methods.length_scores(test))

    if args.embed_model:
        print(f"  {dataset.name}: knn", file=sys.stderr, flush=True)
        knn = ab_methods.Knn(args.url, args.embed_model, args.api_key, args.timeout, args.k)
        knn.fit(dev, ok_dev)
        made = decide_all(knn, test, "knn")
        decisions["knn"] = [ab_methods.Decision()] * len(dev) + made
        scores = [d.score for d in made]
        latency = [d.ms for d in made]
        row("knn (own)", [d.small for d in made], scores, latency)
        # The neighbours of a dev item include itself: its score would be
        # fitted on its own label. Leave one out on the dev half instead.
        fitted("knn", leave_one_out(knn.dev, args.k), scores, latency)

    if details:
        sides = ["dev"] * len(dev) + ["test"] * len(test)
        oks = ok_dev + ok_test
        for n, item in enumerate(every):
            details.write(
                json.dumps(
                    {
                        "set": dataset.name,
                        "id": item.id,
                        "side": sides[n],
                        "small_ok": oks[n],
                        "small": answers[answer_key(item, args.small)].to_json(),
                        "large": answers[answer_key(item, args.large)].to_json(),
                        "decisions": {k: v[n].to_json() for k, v in decisions.items()},
                    }
                )
                + "\n"
            )
        details.flush()
    return rows


def ask_auto(args, item):
    """`item` asked of `"model": "auto"`, as a client asks it: the answer,
    or the error it got."""
    try:
        return ab_methods.generate(
            args.url, item, "auto", args.api_key, args.timeout, args.think, args.max_tokens
        )
    except (ab_methods.ServerError, OSError, ValueError) as e:
        return e


def e2e_rows(args, dataset, answers):
    """Stage 3 on one set: the first `--e2e-limit` items of its test half
    asked of `"model": "auto"`, at each `--concurrency`. Stage 2 scored a
    route by the chosen model's stage-1 grade; that holds only if the model
    a request is routed to answers it as it did when named, so the share of
    answers that are the same text is the first thing reported. The rest is
    what a client sees: time to the first token, routing and loads
    included, against the large model's alone."""
    _, test = ab_data.split(dataset, args.seed)
    subset = test[: args.e2e_limit] if args.e2e_limit else test
    large_alone = [answers[answer_key(i, args.large)] for i in subset]
    rows = []
    for concurrency in args.concurrency:
        print(
            f"  {dataset.name}: auto, {len(subset)} items at concurrency {concurrency}",
            file=sys.stderr,
            flush=True,
        )
        with ThreadPoolExecutor(concurrency) as pool:
            got = list(pool.map(lambda i: ask_auto(args, i), subset))
        made = [
            (i, a) for i, a in zip(subset, got, strict=True) if isinstance(a, ab_methods.Answer)
        ]
        errors = [str(a)[:200] for a in got if not isinstance(a, ab_methods.Answer)]
        same = small = right = 0
        reasons = {}
        for item, answer in made:
            alone = answers.get(answer_key(item, answer.model))
            same += alone is not None and alone.text == answer.text
            small += answer.model == args.small
            if item.grader != "judge":
                right += bool(ab_grade.correct(item, answer.text))
            reason = (answer.route or {}).get("reason") or "unrouted"
            reasons[reason] = reasons.get(reason, 0) + 1
        n = len(made)
        graded = [i for i, _ in made if i.grader != "judge"]
        routing = [
            a.route["decision_ms"]
            for _, a in made
            if isinstance((a.route or {}).get("decision_ms"), (int, float))
        ]
        metrics = {
            "items": len(subset),
            "answered": n,
            "same_answer": same / n if n else None,
            "routed_small": small,
            "accuracy": right / len(graded) if graded else None,
            "accuracy_large": (
                sum(bool(answers[answer_key(i, args.large)].correct) for i in graded) / len(graded)
                if graded
                else None
            ),
            "ttft_ms": latency([a.ttft_ms for _, a in made]),
            "ttft_large_alone_ms": latency([a.ttft_ms for a in large_alone]),
            "total_ms": latency([a.total_ms for _, a in made]),
            "routing_ms": latency(routing),
            "reasons": reasons,
            "errors": errors[:5],
            "error_count": len(errors),
        }
        rows.append({"set": dataset.name, "concurrency": concurrency, "metrics": metrics})
    return rows


def latency(ms):
    """p50 and p95 of `ms`, None when empty."""
    if not ms:
        return None
    return {"p50": percentile(ms, 0.5), "p95": percentile(ms, 0.95)}


E2E_COLUMNS = (
    "set",
    "concurrency",
    "items",
    "same answer",
    "routed small",
    "accuracy",
    "large accuracy",
    "TTFT p50 ms",
    "TTFT p95 ms",
    "large alone TTFT p50 ms",
    "routing p50 ms",
    "routing p95 ms",
    "not decided",
    "errors",
)


def e2e_table(rows):
    """Stage 3 as Markdown, one row per set and concurrency."""
    lines = ["| " + " | ".join(E2E_COLUMNS) + " |", "|---" * len(E2E_COLUMNS) + "|"]

    def pct(x):
        return "—" if x is None else f"{100 * x:.1f}%"

    def ms(stats, key):
        return "—" if not stats else f"{stats[key]:.1f}"

    for r in rows:
        m = r["metrics"]
        undecided = sum(n for reason, n in m["reasons"].items() if reason != "decided")
        cells = [
            r["set"],
            str(r["concurrency"]),
            str(m["items"]),
            pct(m["same_answer"]),
            str(m["routed_small"]),
            pct(m["accuracy"]),
            pct(m["accuracy_large"]),
            ms(m["ttft_ms"], "p50"),
            ms(m["ttft_ms"], "p95"),
            ms(m["ttft_large_alone_ms"], "p50"),
            ms(m["routing_ms"], "p50"),
            ms(m["routing_ms"], "p95"),
            str(undecided),
            str(m["error_count"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def score_of(decision):
    """P(small); when the router gave none (a fallback), what it decided."""
    if decision.score is not None:
        return decision.score
    return 1.0 if decision.small else 0.0


def leave_one_out(dev, k):
    """Each dev item's kNN score among the other dev items."""
    scores = []
    for n, (vector, _) in enumerate(dev):
        others = dev[:n] + dev[n + 1 :]
        near = sorted(others, key=lambda d: -sum(a * b for a, b in zip(vector, d[0], strict=True)))[
            :k
        ]
        scores.append(sum(ok for _, ok in near) / len(near) if near else 0.0)
    return scores


COLUMNS = (
    "set",
    "router",
    "test items",
    "routed small",
    "large calls avoided",
    "large GPU-s avoided",
    "accuracy",
    "Δ vs large",
    "95% CI",
    "lost",
    "gained",
    "AUROC",
    "p50 ms",
    "p95 ms",
)


def table(rows):
    """The report as Markdown, one row per set and router."""
    head = "| " + " | ".join(COLUMNS) + " |"
    lines = [head, "|---" * len(COLUMNS) + "|"]

    def pct(x):
        return "—" if x is None else f"{100 * x:.1f}%"

    def num(x, digits=3):
        return "—" if x is None else f"{x:.{digits}f}"

    for r in rows:
        m = r["metrics"]
        ci = m.get("delta_ci95")
        lat = m.get("latency_ms") or {}
        cells = [
            r["set"],
            r["router"],
            str(m["test_items"]),
            str(m["routed_small"]),
            pct(m.get("calls_avoided")),
            num(m.get("large_gpu_s_avoided"), 1),
            pct(m.get("accuracy")),
            "—" if m.get("delta") is None else f"{100 * m['delta']:+.1f}",
            "—" if ci is None else f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]",
            str(m["lost"]),
            str(m["gained"]),
            num(m.get("auroc")),
            num(lat.get("p50"), 1),
            num(lat.get("p95"), 1),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:11434", help="EuLLM server URL")
    parser.add_argument("--small", required=True, help="the small model, an auto candidate")
    parser.add_argument("--large", required=True, help="the large model, an auto candidate")
    parser.add_argument("--embed-model", default=None, help="embedding model for `knn`")
    parser.add_argument("--k", type=int, default=10, help="neighbours for `knn`")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EULLM_API_KEY"),
        help="API key, when the server requires one (default: $EULLM_API_KEY)",
    )
    parser.add_argument(
        "--sets",
        default="gsm8k,arc-easy,arc-challenge,mmlu",
        help=f"comma-separated: {', '.join(ab_data.SETS)}",
    )
    parser.add_argument("--data", action="append", default=[], help="a JSONL set of your own")
    parser.add_argument("--limit", type=int, default=100, help="items per set, 0 for all")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stages", default=",".join(STAGES), help="comma-separated stages")
    parser.add_argument("--answers", default=None, help="keep and reuse stage 1's answers")
    parser.add_argument("--think", action="store_true", help="let the models think first")
    parser.add_argument("--max-tokens", type=int, default=768, help="most tokens an answer gets")
    parser.add_argument(
        "--questions",
        default=None,
        help="JSON list of {name, instructions?, reverse?, descriptions?}: other wordings "
        "and option orders, asked of /v1/systemone about the server's own state",
    )
    parser.add_argument("--judge-model", default=None, help="judge for `judge` items")
    parser.add_argument("--judge-labels", default=None, help="hand verdicts, for Cohen's kappa")
    parser.add_argument(
        "--e2e-limit", type=int, default=50, help="test items per set for `e2e`, 0 for all"
    )
    parser.add_argument(
        "--concurrency", default="1,4,16", help="comma-separated concurrencies for `e2e`"
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request timeout, s")
    parser.add_argument("--out", default=None, help="the report, as JSON")
    parser.add_argument("--details", default=None, help="every item, one JSON line each")
    args = parser.parse_args(argv)
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in args.stages if s not in STAGES]
    if unknown:
        parser.error(f"unknown stages {unknown}: {', '.join(STAGES)}")
    try:
        args.concurrency = [int(c) for c in args.concurrency.split(",") if c.strip()]
    except ValueError:
        parser.error(f"--concurrency takes numbers: {args.concurrency}")
    if not args.concurrency or min(args.concurrency) < 1:
        parser.error("--concurrency needs one level or more, each at least 1")
    args.question_specs = (
        json.loads(pathlib.Path(args.questions).read_text(encoding="utf-8"))
        if args.questions
        else []
    )
    return args


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(args.out or f"autobench-{stamp}.json")
    datasets = [
        ab_data.load(s.strip(), args.limit, args.seed) for s in args.sets.split(",") if s.strip()
    ]
    datasets += [ab_data.from_jsonl(path, args.limit, args.seed) for path in args.data]
    if any(i.grader == "judge" for d in datasets for i in d.items) and not args.judge_model:
        raise SystemExit("the set has judge items: name a judge with --judge-model")

    before = ab_methods.server_state(args.url, args.api_key, args.timeout)
    answers = load_answers(args.answers, answers_how(args))
    if "generate" in args.stages:
        print("stage 1: generate", file=sys.stderr, flush=True)
        generate_all(args, datasets, answers)
    missing = [
        (d.name, i.id, m)
        for d in datasets
        for i in d.items
        for m in (args.small, args.large)
        if answer_key(i, m) not in answers
    ]
    if missing:
        raise SystemExit(f"{len(missing)} answers missing, e.g. {missing[0]}: run `generate`")
    for dataset in datasets:
        judge_all(args, dataset, answers)

    details = open(args.details, "w", encoding="utf-8") if args.details else None
    rows = []
    e2e = []
    try:
        if "route" in args.stages:
            print("stage 2: route", file=sys.stderr, flush=True)
            for dataset in datasets:
                rows += rows_for_set(args, dataset, answers, details)
        if "e2e" in args.stages:
            print("stage 3: end to end", file=sys.stderr, flush=True)
            if args.embed_model:
                done = ab_methods.release_embedder(
                    args.url, args.embed_model, args.api_key, args.timeout
                )
                print(f"  {args.embed_model}: {done}", file=sys.stderr, flush=True)
            for dataset in datasets:
                e2e += e2e_rows(args, dataset, answers)
    finally:
        if details:
            details.close()
        after = ab_methods.server_state(args.url, args.api_key, args.timeout)
        loads = None
        if ab_methods.evictions(before) is not None and ab_methods.evictions(after) is not None:
            loads = ab_methods.evictions(after) - ab_methods.evictions(before)
        decision_models = sorted({r["metrics"].get("decision_model") for r in rows} - {None})
        report = {
            "when": stamp,
            "url": args.url,
            "server": before,
            "small": args.small,
            "large": args.large,
            "embed_model": args.embed_model,
            "sets": [d.name for d in datasets],
            "limit": args.limit,
            "seed": args.seed,
            "think": args.think,
            "max_tokens": args.max_tokens,
            "generation_evictions_during_run": loads,
            "decision_models": decision_models,
            "judge": {"model": args.judge_model, "agreement": judge_agreement(args, datasets)},
            "results": rows,
            "kill_criterion": ab_metrics.kill_criterion(rows) if rows else None,
            "e2e_limit": args.e2e_limit,
            "end_to_end": e2e,
        }
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if rows:
            version = (before or {}).get("version") or {}
            print(f"server {version.get('version')}, {args.small} / {args.large}")
            if decision_models:
                print(f"decision model: {', '.join(decision_models)}")
            print()
            print(table(rows))
            for set_name, beaten in (report["kill_criterion"] or {}).items():
                if beaten:
                    print(f"\n{set_name}: kill criterion met by {', '.join(beaten)}")
        if e2e:
            print('\nend to end, model "auto":\n')
            print(e2e_table(e2e))
            low = [
                f"{r['set']} at {r['concurrency']}"
                for r in e2e
                if r["metrics"]["same_answer"] is not None and r["metrics"]["same_answer"] < 0.99
            ]
            if low:
                print(
                    f"\nunder 99% the same answer as the model alone ({', '.join(low)}): "
                    "stage 2's scores do not hold there"
                )
        print(f"\nreport: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
