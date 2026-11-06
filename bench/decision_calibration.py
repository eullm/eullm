#!/usr/bin/env python3
"""Calibration comparison for the decision layer (`POST /v1/systemone`).

Runs a labelled dataset through a server's decision model once, then scores
four ways of turning the model's log-probabilities into probabilities:

  none            the raw probabilities, renormalized over the answers
  content_free    the model's answer about the state "N/A" divided out
                  (Zhao et al., 2021)
  temperature     temperature scaling, T fitted on the other folds
  cf+temperature  both

with the metrics that say whether a probability can be trusted, each with a
95% bootstrap interval:

  accuracy  share of items whose most probable answer is the label
  NLL       mean -log p(label): proper scoring rule, punishes confident errors
  Brier     mean squared error of the whole distribution: proper, bounded
  ECE       expected calibration error of the top answer, 15 equal-width bins

plus the mean coverage (the share of probability the model put on valid answer
codes at all) and, per method, the accuracy on the items it would answer at a
few confidence thresholds — the trade-off behind "above p, automate; below,
ask a human".

One request per item asks for `content_free` calibration, which returns the
raw log-probabilities and the prior, so every method is computed offline from
the same answers. T is fitted by 5-fold cross-fitting: each item is scored
with a T fitted on the other four folds, so a method is never scored on the
data it was fitted on.

Dataset: JSONL, one item per line.

    {"state": "Payouts failing for 3 days...",
     "question": {"type": "choice", "instructions": "Which team?",
                  "criteria": {"billing": "...", "tech": "...", "other": "..."}},
     "label": "billing"}

`label` is the option name for `choice`, the level number (0-based) for
`score`, and true/false (or "yes"/"no") for `noul`.

    python bench/decision_calibration.py dataset.jsonl --url http://localhost:11434
    python bench/decision_calibration.py dataset.jsonl --json results/calibration.json

Only the Python standard library is needed.
"""

import argparse
import json
import math
import random
import sys
import urllib.error
import urllib.request

METHODS = ["none", "content_free", "temperature", "cf+temperature"]
ECE_BINS = 15
FOLDS = 5
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]


def post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"HTTP {e.code}: {e.read().decode(errors='replace')}"
        ) from None


def label_index(question, labels, label):
    """Index of the gold label among the answer labels the server returned."""
    kind = question["type"]
    if kind == "noul":
        if isinstance(label, bool):
            return 0 if label else 1
        text = str(label).strip().lower()
        if text in ("yes", "true", "1"):
            return 0
        if text in ("no", "false", "0"):
            return 1
        raise ValueError(f"noul label must be true/false or yes/no, got {label!r}")
    key = str(label)
    if key not in labels:
        raise ValueError(f"label {label!r} is not one of {labels}")
    return labels.index(key)


def softmax(values):
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def probabilities(item, method, temperature=1.0):
    lp = item["logprobs"]
    if method in ("content_free", "cf+temperature"):
        lp = [a - b for a, b in zip(lp, item["prior_logprobs"])]
    t = temperature if method in ("temperature", "cf+temperature") else 1.0
    return softmax([v / t for v in lp])


def nll(items, method, temperature):
    total = 0.0
    for item in items:
        p = probabilities(item, method, temperature)[item["label"]]
        total -= math.log(max(p, 1e-12))
    return total / len(items)


def fit_temperature(items, method):
    """T minimizing NLL, by golden-section search on log T in [0.05, 20]."""
    lo, hi = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = nll(items, method, math.exp(a)), nll(items, method, math.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = nll(items, method, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = nll(items, method, math.exp(b))
    return math.exp((lo + hi) / 2)


def scored(items, method, seed):
    """Per-item probability vectors for `method`, temperatures cross-fitted."""
    if method in ("none", "content_free"):
        return [probabilities(item, method) for item in items], []
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    folds = [order[k::FOLDS] for k in range(FOLDS)]
    out = [None] * len(items)
    temperatures = []
    for fold in folds:
        held_out = set(fold)
        train = [items[i] for i in order if i not in held_out]
        t = fit_temperature(train, method) if train else 1.0
        temperatures.append(t)
        for i in fold:
            out[i] = probabilities(items[i], method, t)
    return out, temperatures


def metrics(items, probs):
    n = len(items)
    correct = 0
    total_nll = 0.0
    total_brier = 0.0
    bins = [
        [0, 0.0, 0.0] for _ in range(ECE_BINS)
    ]  # count, sum confidence, sum correct
    for item, p in zip(items, probs):
        y = item["label"]
        top = max(range(len(p)), key=p.__getitem__)
        hit = 1.0 if top == y else 0.0
        correct += hit
        total_nll -= math.log(max(p[y], 1e-12))
        total_brier += sum(
            (pc - (1.0 if c == y else 0.0)) ** 2 for c, pc in enumerate(p)
        )
        b = min(int(p[top] * ECE_BINS), ECE_BINS - 1)
        bins[b][0] += 1
        bins[b][1] += p[top]
        bins[b][2] += hit
    ece = sum(abs(s_hit - s_conf) for _, s_conf, s_hit in bins) / n
    return {
        "accuracy": correct / n,
        "nll": total_nll / n,
        "brier": total_brier / n,
        "ece": ece,
    }


def bootstrap(items, probs, rounds, seed):
    rng = random.Random(seed)
    n = len(items)
    samples = {k: [] for k in ("accuracy", "nll", "brier", "ece")}
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        m = metrics([items[i] for i in idx], [probs[i] for i in idx])
        for k, v in m.items():
            samples[k].append(v)
    ci = {}
    for k, values in samples.items():
        values.sort()
        ci[k] = (
            values[int(0.025 * rounds)],
            values[min(int(0.975 * rounds), rounds - 1)],
        )
    return ci


def selective(items, probs):
    """Share of items answered and accuracy on them, per confidence threshold."""
    rows = []
    for threshold in THRESHOLDS:
        answered = [(item, p) for item, p in zip(items, probs) if max(p) >= threshold]
        if answered:
            hits = sum(
                1
                for item, p in answered
                if max(range(len(p)), key=p.__getitem__) == item["label"]
            )
            rows.append((threshold, len(answered) / len(items), hits / len(answered)))
        else:
            rows.append((threshold, 0.0, None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", help="labelled JSONL (see the module docstring)")
    parser.add_argument(
        "--url", default="http://localhost:11434", help="server base URL"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="decision model to use (default: the one the server has loaded)",
    )
    parser.add_argument(
        "--mode",
        default="shared_prefix",
        choices=["shared_prefix", "separate"],
        help="evaluation mode; calibrate in the mode that will serve",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="bootstrap rounds for the CIs"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json", default=None, help="write the results to this file")
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/v1/systemone"
    with open(args.dataset, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) < FOLDS * 2:
        print(
            f"Need at least {FOLDS * 2} labelled items, got {len(rows)}",
            file=sys.stderr,
        )
        return 1

    items = []
    for n, row in enumerate(rows, 1):
        payload = {
            "state": row["state"],
            "questions": {"q": row["question"]},
            "eullm": {"calibration": "content_free", "mode": args.mode},
        }
        if args.model:
            payload["model"] = args.model
        answer = post(url, payload, args.timeout)["answers"]["q"]
        ext = answer["eullm"]
        labels = list(ext["logprobs"].keys())
        items.append(
            {
                "type": row["question"]["type"],
                "label": label_index(row["question"], labels, row["label"]),
                "logprobs": [ext["logprobs"][k] for k in labels],
                "prior_logprobs": [ext["prior_logprobs"][k] for k in labels],
                "coverage": ext["coverage"],
            }
        )
        if n % 50 == 0:
            print(f"  {n}/{len(rows)} items", file=sys.stderr)

    coverages = sorted(item["coverage"] for item in items)
    print(
        f"{len(items)} items; coverage mean {sum(coverages) / len(coverages):.3f}, "
        f"min {coverages[0]:.3f}, 5th percentile {coverages[len(coverages) // 20]:.3f}\n"
    )

    header = f"{'method':<15} {'accuracy':>17} {'NLL':>17} {'Brier':>17} {'ECE':>17}  T (per fold)"
    print(header)
    print("-" * len(header))
    results = {}
    for method in METHODS:
        probs, temperatures = scored(items, method, args.seed)
        m = metrics(items, probs)
        ci = bootstrap(items, probs, args.bootstrap, args.seed)
        cells = " ".join(
            f"{m[k]:>6.3f} [{ci[k][0]:.3f},{ci[k][1]:.3f}]"
            for k in ("accuracy", "nll", "brier", "ece")
        )
        temps = ", ".join(f"{t:.2f}" for t in temperatures) or "-"
        print(f"{method:<15} {cells}  {temps}")
        results[method] = {
            "metrics": m,
            "ci95": ci,
            "temperatures": temperatures,
            "selective": selective(items, probs),
        }

    print("\nAnswered share / accuracy on the answered items, by confidence threshold:")
    print(f"{'method':<15} " + " ".join(f"{'p>=' + str(t):>14}" for t in THRESHOLDS))
    for method in METHODS:
        cells = " ".join(
            f"{share:>5.0%} / {acc:>5.1%}" if acc is not None else f"{'-':>14}"
            for _, share, acc in results[method]["selective"]
        )
        print(f"{method:<15} {cells}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "url": args.url,
                    "model": args.model,
                    "mode": args.mode,
                    "items": len(items),
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
