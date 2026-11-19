"""What the AutoBench report says about a router.

A router sends each request to the small model or to the large one. Since
both models answered every item deterministically (stage 1), what a router
would have scored is known without asking again: the chosen model's grade
on each item. Per set and router, on the test half:

  * calls avoided: the share routed to the small model, and the large
    model's GPU time those requests would have taken;
  * accuracy, and its difference from always asking the large model with a
    95% confidence interval from a paired bootstrap over the items;
  * answers lost (the large model was right, the chosen one wrong) and
    gained (the other way round);
  * AUROC of the router's score, P(small), against `small_ok`: the small
    model was right, or both were wrong — routing small cost nothing;
  * what deciding cost: latency p50 and p95.

A router with a score is reported twice: at its own decision, and at a
threshold fitted on the dev half to avoid the most calls for at most one
point of accuracy lost (`fit_threshold`).
"""

import math
import random
import statistics

from rb_metrics import percentile
from rg_metrics import auroc

# Accuracy a fitted threshold may give up against always-large, in points.
MAX_LOSS_POINTS = 1.0


def paired_bootstrap(a, b, rounds=2000, seed=1):
    """95% confidence interval of mean(a) - mean(b), resampling the items in
    pairs. `a` and `b` are per-item scores of the same items."""
    if not a:
        return None
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(rounds):
        picks = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in picks) / n)
    diffs.sort()
    return diffs[int(0.025 * rounds)], diffs[min(rounds - 1, int(0.975 * rounds))]


def fit_threshold(scores, small_correct, large_correct, max_loss=MAX_LOSS_POINTS):
    """The threshold on P(small) — route small at or above it — that routes
    the most items small while accuracy stays within `max_loss` points of
    always-large, on the items given (the dev half). None when no threshold
    keeps within it; then the router is used at its own decision only."""
    if not scores:
        return None
    n = len(scores)
    large = sum(large_correct)
    best, best_small = None, -1
    for t in sorted(set(scores)) + [math.inf]:
        routed = [s >= t for s in scores]
        right = sum(
            sc if r else lc for r, sc, lc in zip(routed, small_correct, large_correct, strict=True)
        )
        if 100 * (large - right) / n <= max_loss and sum(routed) > best_small:
            best, best_small = t, sum(routed)
    return best


def cohen_kappa(a, b):
    """Agreement of two raters over the same items, beyond chance."""
    if not a or len(a) != len(b):
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def summarize(items, routed, small_correct, large_correct, large_ms, scores=None, latency=None):
    """One router on one set's test half. `routed`: True where the request
    goes to the small model; `scores`: P(small), when the router has one;
    `large_ms`: what each request took on the large model; `latency`: what
    deciding took, in ms, when it is not free."""
    n = len(items)
    chosen = [s if r else lc for r, s, lc in zip(routed, small_correct, large_correct, strict=True)]
    report = {
        "test_items": n,
        "routed_small": sum(routed),
        "calls_avoided": sum(routed) / n if n else None,
        "large_gpu_s_avoided": sum(ms for r, ms in zip(routed, large_ms, strict=True) if r) / 1000,
        "accuracy": sum(chosen) / n if n else None,
        "accuracy_large": sum(large_correct) / n if n else None,
        "lost": sum(lc and not c for c, lc in zip(chosen, large_correct, strict=True)),
        "gained": sum(c and not lc for c, lc in zip(chosen, large_correct, strict=True)),
    }
    if n:
        report["delta"] = report["accuracy"] - report["accuracy_large"]
        report["delta_ci95"] = paired_bootstrap(
            [float(c) for c in chosen], [float(c) for c in large_correct]
        )
    if scores is not None:
        ok = [sc or not lc for sc, lc in zip(small_correct, large_correct, strict=True)]
        report["auroc"] = auroc(scores, ok)
    if latency:
        report["latency_ms"] = {"p50": percentile(latency, 0.5), "p95": percentile(latency, 0.95)}
    return report


def within_ci(report):
    """Whether a router's accuracy is indistinguishable from always-large's:
    the confidence interval of the difference contains zero, or lies above."""
    ci = report.get("delta_ci95")
    return ci is not None and ci[1] >= 0


def kill_criterion(rows):
    """The roadmap's kill criterion, per set: a router that is not Reflex —
    the length threshold or the embeddings' neighbours — avoiding at least as
    many large-model calls as Reflex's best row, at an accuracy just as
    indistinguishable from always-large. Returns {set: [routers that meet
    it]}; an empty list means Reflex earns its place there."""
    verdict = {}
    for set_name in sorted({r["set"] for r in rows}):
        mine = [r for r in rows if r["set"] == set_name]
        reflex = [
            r["metrics"]["calls_avoided"]
            for r in mine
            if r["router"].startswith("reflex") and within_ci(r["metrics"])
        ]
        best = max(reflex, default=0.0)
        verdict[set_name] = [
            r["router"]
            for r in mine
            if r["router"].startswith(("length", "knn"))
            and within_ci(r["metrics"])
            and r["metrics"]["calls_avoided"] >= best
        ]
    return verdict


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None
