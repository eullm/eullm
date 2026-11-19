"""What ReflexBench reports about a method on a set.

The three questions MVP 0 has to answer (docs/reflex-roadmap.md):

  1. how many tools can be dropped without losing the right one: recall@k,
     and the k that keeps 95% and 99% of the requests whole;
  2. how much of the tool-calling model's prompt that saves: the specs of
     the k tools kept against all of them;
  3. what the decision costs: latency p50 and p95, and for Reflex the
     tokens the decision model evaluated.
"""

import math
import statistics

KS = (1, 3, 5, 10, 20)
COVERAGES = (0.95, 0.99)


def ranks(order, needed, offered):
    """1-based positions of the needed tools. A request the decision model
    refused ranks nothing, and its tools count as the last of the `offered`:
    every one of them would have to be sent."""
    position = {name: i + 1 for i, name in enumerate(order)}
    return [position.get(name, offered) for name in needed]


def worst_rank(order, needed, offered):
    """The position of the needed tool ranked lowest: the k it takes to keep
    every tool the request needs."""
    return max(ranks(order, needed, offered))


def k_for(ranks, coverage):
    """The smallest k that keeps every needed tool for `coverage` of the
    requests."""
    ordered = sorted(ranks)
    return ordered[max(0, math.ceil(coverage * len(ordered)) - 1)]


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))]


def chars(tools):
    return sum(len(t.spec) for t in tools)


def summarize(items, rankings):
    """`items` and their `rankings`, one method on one set."""
    with_tools = [(i, r) for i, r in zip(items, rankings, strict=True) if i.needed]
    without = [(i, r) for i, r in zip(items, rankings, strict=True) if not i.needed]
    report = {"items": len(items), "items_needing_tools": len(with_tools)}
    if with_tools:
        worst = [worst_rank(r.order, i.needed, len(i.candidates)) for i, r in with_tools]
        report["recall_at"] = {str(k): sum(rank <= k for rank in worst) / len(worst) for k in KS}
        report["mrr"] = statistics.mean(
            1 / min(ranks(r.order, i.needed, len(i.candidates))) for i, r in with_tools
        )
        report["candidates_mean"] = statistics.mean(len(i.candidates) for i, _ in with_tools)
        for coverage in COVERAGES:
            k = k_for(worst, coverage)
            # A refused request sends every spec.
            kept = [
                chars([t for t in i.candidates if t.name in r.order[:k]])
                / max(1, chars(i.candidates))
                if r.order
                else 1.0
                for i, r in with_tools
            ]
            report[f"k{round(coverage * 100)}"] = k
            report[f"spec_kept_at_k{round(coverage * 100)}"] = statistics.mean(kept)
        all_chars = [chars(i.candidates) for i, _ in with_tools]
        report["spec_chars_all_mean"] = statistics.mean(all_chars)
        # About 4 characters a token for English and JSON; the exact count
        # needs the tool-calling model's own tokenizer.
        report["spec_tokens_all_mean_estimate"] = statistics.mean(all_chars) / 4
    asked = [(i, r) for i, r in without if r.none_score is not None]
    if asked:
        # Requests no offered tool fits: how often "no tool" won.
        report["abstain_accuracy"] = sum(r.abstains() for _, r in asked) / len(asked)
    asked = [(i, r) for i, r in with_tools if r.none_score is not None]
    if asked:
        # Requests that do need a tool: how often "no tool" won anyway.
        report["false_abstain_rate"] = sum(r.abstains() for _, r in asked) / len(asked)
    report["refused"] = sum(r.refused for r in rankings)
    # What deciding cost, over the requests that got a ranking.
    decided = [r for r in rankings if r.order]
    ms = [r.ms for r in decided]
    report["latency_ms"] = {"p50": percentile(ms, 0.5), "p95": percentile(ms, 0.95)}
    served = [r for r in decided if r.server.get("evaluated_tokens") is not None]
    if served:
        report["evaluated_tokens_mean"] = statistics.mean(
            r.server["evaluated_tokens"] for r in served
        )
        reused = sum(bool(r.server.get("prefix_reused")) for r in served)
        report["prefix_reused_rate"] = reused / len(served)
        report["questions_mean"] = statistics.mean(r.server.get("questions", 1) for r in served)
    return report
