#!/usr/bin/env python3
"""ReflexBench, RAG gate: do the passages retrieved for a question suffice
to answer it?

MVP 1 of the Reflex roadmap (docs/reflex-roadmap.md). A RAG system retrieves
passages and hands them to a large model, which answers whether or not the
facts it needs are there. A gate between the two decides: `answer`,
`retrieve_more` (some facts are there, one is missing), or `abstain`
(nothing there helps). For each case of a labelled set, every method scores
how likely the passages suffice; the report says, per set and method:

  * AUROC: how well that score separates sufficient passages from the rest;
  * the share of insufficient cases the gate stops (`caught`) and of
    sufficient ones it stops anyway (`blocked`), at the method's own
    decision and at a threshold fitted on the dev half;
  * for the three-way decision, accuracy and macro-F1;
  * ECE for Reflex's probabilities, and what a decision costs.

The methods (see rg_methods.py): `embed-max`, the best question-passage
similarity from EuLLM's embeddings (needs --embed-model); `reflex-gate`, one
`/v1/systemone` choice among the three; `reflex-yesno`, one yes/no.

    EULLM_AUDIT_DIR=/tmp/ragbench-audit \\
      eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m \\
                  --embedding-model qwen3-embedding-0.6b-gguf-q8_0
    python bench/reflexbench/ragbench.py --limit 1000 --embed-model qwen3-embedding-0.6b-gguf-q8_0

Only the Python standard library is needed.
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rg_data  # noqa: E402
import rg_methods  # noqa: E402
import rg_metrics  # noqa: E402

METHODS = ("embed-max", "reflex-gate", "reflex-yesno")


def build_methods(args):
    methods = []
    for name in args.methods.split(","):
        name = name.strip()
        if name == "embed-max":
            if not args.embed_model:
                print("  embed-max skipped: no --embed-model", file=sys.stderr)
                continue
            prefix = args.embed_query_prefix.replace("\\n", "\n")
            methods.append(
                rg_methods.EmbedMax(args.url, args.embed_model, args.api_key, args.timeout, prefix)
            )
        elif name == "reflex-gate":
            methods.append(rg_methods.ReflexGate(args.url, args.model, args.api_key, args.timeout))
        elif name == "reflex-yesno":
            methods.append(rg_methods.ReflexYesNo(args.url, args.model, args.api_key, args.timeout))
        elif name:
            raise SystemExit(f"unknown method {name!r}: {', '.join(METHODS)}")
    return methods


def run(method, cases, details, set_name, sides):
    """Decide every case, after one untimed decision on the last."""
    if cases:
        model = method.decide(cases[-1]).server.get("model")
        if model:
            print(f"    decision model: {model}", file=sys.stderr, flush=True)
    decisions = []
    started = shown = time.perf_counter()
    for n, case in enumerate(cases, 1):
        decision = method.decide(case)
        decisions.append(decision)
        if details:
            details.write(
                json.dumps(
                    {
                        "set": set_name,
                        "method": method.name,
                        "id": case.id,
                        "side": sides[case.id],
                        "label": case.label,
                        "score": decision.score,
                        "choice": decision.choice,
                        "probabilities": decision.probabilities,
                        "ms": round(decision.ms, 2),
                        "server": decision.server,
                    }
                )
                + "\n"
            )
            details.flush()
        now = time.perf_counter()
        if n % 50 == 0 or n == len(cases) or now - shown >= 30:
            shown = now
            print(f"    {n}/{len(cases)} ({now - started:.0f} s)", file=sys.stderr, flush=True)
    return decisions


def table(results):
    """The report as Markdown: the two-way decision first."""
    head = (
        "| set | method | test cases | AUROC | AUROC within a question | "
        "own: caught | own: blocked | fitted: caught | fitted: blocked | "
        "3-way accuracy | 3-way macro-F1 | ECE | "
        "p50 ms | p95 ms | tokens/decision |"
    )
    lines = [head, "|---" * (head.count("|") - 1) + "|"]

    def pct(x):
        return "—" if x is None else f"{100 * x:.1f}%"

    def num(x, digits=3):
        return "—" if x is None else f"{x:.{digits}f}"

    for r in results:
        m = r["metrics"]
        own, fitted = m.get("own") or {}, m.get("fitted") or {}
        three = m.get("own_three_way") or m.get("fitted_three_way") or {}
        lat = m["latency_ms"]
        tokens = m.get("evaluated_tokens_mean")
        lines.append(
            f"| {r['set']} | {r['method']} | {m['test_cases']} | {num(m.get('auroc'))} | "
            f"{num(m.get('auroc_within_question'))} | "
            f"{pct(own.get('caught'))} | {pct(own.get('blocked'))} | "
            f"{pct(fitted.get('caught'))} | {pct(fitted.get('blocked'))} | "
            f"{pct(three.get('accuracy'))} | {pct(three.get('macro_f1'))} | "
            f"{num(m.get('ece'))} | {num(lat['p50'], 1)} | {num(lat['p95'], 1)} | "
            f"{'—' if tokens is None else f'{tokens:.0f}'} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:11434", help="EuLLM server URL")
    parser.add_argument("--model", default=None, help="decision model (default: the one loaded)")
    parser.add_argument("--embed-model", default=None, help="embedding model for `embed-max`")
    parser.add_argument(
        "--embed-query-prefix",
        default="",
        help="text put before each question for `embed-max`; \\n is a newline",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EULLM_API_KEY"),
        help="API key, when the server requires one (default: $EULLM_API_KEY)",
    )
    parser.add_argument(
        "--sets", default="musique", help=f"comma-separated: {', '.join(rg_data.SETS)}"
    )
    parser.add_argument("--data", action="append", default=[], help="a JSONL set of your own")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--limit", type=int, default=100, help="questions per set, 3 cases each")
    parser.add_argument("--passages", type=int, default=5, help="passages per case")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout, seconds")
    parser.add_argument("--out", default=None, help="the report, as JSON")
    parser.add_argument("--details", default=None, help="every decision, one JSON line each")
    args = parser.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(args.out or f"ragbench-{stamp}.json")
    details = open(args.details, "w", encoding="utf-8") if args.details else None
    datasets = [
        rg_data.load(s.strip(), args.limit, args.seed, args.passages)
        for s in args.sets.split(",")
        if s.strip()
    ]
    datasets += [rg_data.from_jsonl(path) for path in args.data]

    results = []
    try:
        for dataset in datasets:
            dev, test = rg_data.split(dataset, args.seed)
            sides = {c.id: "dev" for c in dev} | {c.id: "test" for c in test}
            cases = dev + test
            print(
                f"{dataset.name}: {len(cases)} cases ({len(dev)} dev, {len(test)} test)",
                file=sys.stderr,
                flush=True,
            )
            for method in build_methods(args):
                print(f"  {method.name}", file=sys.stderr, flush=True)
                decisions = run(method, cases, details, dataset.name, sides)
                paired = list(zip(cases, decisions, strict=True))
                served = sorted({d.server["model"] for d in decisions if d.server.get("model")})
                results.append(
                    {
                        "set": dataset.name,
                        "method": method.name,
                        "decision_model": ", ".join(served) or None,
                        "metrics": rg_metrics.summarize(paired[: len(dev)], paired[len(dev) :]),
                    }
                )
    finally:
        if details:
            details.close()
        report = {
            "when": stamp,
            "url": args.url,
            "decision_model_asked": args.model,
            "decision_models": sorted({r["decision_model"] for r in results} - {None}),
            "embed_model": args.embed_model,
            "embed_query_prefix": args.embed_query_prefix,
            "limit": args.limit,
            "passages": args.passages,
            "seed": args.seed,
            "results": results,
        }
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if results:
            if report["decision_models"]:
                print(f"decision model: {', '.join(report['decision_models'])}\n")
            print(table(results))
        print(f"\nreport: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
