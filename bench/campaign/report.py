"""A readable summary of a campaign's results: one table per group.

`campaign.py report --queue Q [--campaign c06-mtp ...]` prints, for every
group, a row per configuration that was measured: the parameters that vary
within the group (model, batch, --mtp, runtime...) and the metrics that fit
the kind of point, averaged over the results that share a configuration
(retries, rounds). Points that failed or did not fit are counted under the
table, so a group with holes says so.
"""

from __future__ import annotations

from collections import defaultdict

# Parameters a group may vary; the ones that do become the table's first
# columns, in this order.
PARAMS = ("model", "runtime", "gcds", "replicas", "batch", "concurrency", "slot_ctx", "kv",
          "prompt_tokens", "decision_mode", "state_tokens", "questions", "extra_args",
          "env", "trial", "round", "engine_label")

# What each kind is read by: (column in summary.csv, heading).
# Every kind that runs on the GPUs also shows the most VRAM a device held and
# how busy the devices were: a server that silently ran on the CPU, or shared
# its GCD with a stranger, shows there before anywhere else.
DEVICE = (("vram_start_mib_max", "vram0"), ("vram_peak_mib_max", "vram MiB"),
          ("use_mean", "use%"), ("neighbours_at_start", "neigh"))
METRICS = {
    "throughput": (("agg_tok_s_mean", "tok/s"), ("agg_tok_s_cv_pct", "cv%"),
                   ("decode_tok_s", "decode"), ("prefill_tok_s", "prefill"),
                   ("ttft_ms_p50", "ttft ms"), ("load_wall_s", "load s"),
                   ("draft_accept_pct", "kept%")) + DEVICE,
    "workload": (("accuracy", "accuracy"), ("agg_tok_s_mean", "tok/s"),
                 ("consistency", "consist"), ("requests", "requests"),
                 ("duration_s", "dur s"), ("died", "died"),
                 ("draft_accept_pct", "kept%")) + DEVICE,
    "decision": (("dec_per_s", "dec/s"), ("dec_client_ms_p50", "p50 ms"),
                 ("dec_client_ms_p99", "p99 ms"), ("dec_wait_ms_p50", "wait ms"),
                 ("dec_decode_ms_p50", "decode ms"), ("dec_consistency", "consist"),
                 ("dec_same_choices", "same choice"), ("dec_flipped", "flipped"),
                 ("dec_largest_shift", "max shift"),
                 ("dec_together", "together")) + DEVICE,
    "finetune": (("ft_loss_before", "loss0"), ("ft_loss_after", "loss1"),
                 ("ft_tok_s", "tok/s")),
}

SHORT = {"model": 34, "extra_args": 26, "env": 34}


def _number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(values):
    """The mean of the numbers among `values`; for text (accuracy is
    `set=0.8 ...`), the first value, since it is not averaged."""
    nums = [n for n in (_number(v) for v in values) if n is not None]
    if nums:
        m = sum(nums) / len(nums)
        return f"{m:.0f}" if abs(m) >= 100 else f"{m:.3g}"
    texts = [v for v in values if v not in (None, "")]
    return texts[0] if texts else "-"


def _cell(name, value):
    value = "-" if value in (None, "") else str(value)
    if name == "extra_args":
        value = value.replace("--fit-strict", "").strip() or "(none)"
    width = SHORT.get(name)
    return value if not width or len(value) <= width else value[: width - 1] + "…"


def table(rows, kind):
    """The lines of one group's table, `rows` its measured results."""
    varying = [p for p in PARAMS if len({r.get(p) for r in rows}) > 1]
    if not varying:
        varying = ["model"]
    if "env" in varying:
        # the variables every row sets alike say nothing about the row
        common = set.intersection(*(set((r.get("env") or "").split()) for r in rows))
        rows = [r | {"env": " ".join(v for v in (r.get("env") or "").split()
                                     if v not in common)} for r in rows]
    # A metric no row of the group has (MTP drafts without --mtp) is no column.
    metrics = [m for m in METRICS.get(kind, ())
               if any(r.get(m[0]) not in (None, "") for r in rows)]
    by_config = defaultdict(list)
    for r in rows:
        by_config[tuple(r.get(p) for p in varying)].append(r)
    header = [p for p in varying] + [h for _, h in metrics] + ["n"]
    body = []
    for key in sorted(by_config, key=lambda k: tuple(_sort_key(v) for v in k)):
        group = by_config[key]
        body.append([_cell(p, v) for p, v in zip(varying, key)]
                    + [_mean([r.get(c) for r in group]) for c, _ in metrics]
                    + [str(len(group))])
    widths = [max(len(header[i]), *(len(b[i]) for b in body)) for i in range(len(header))]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    return [fmt.format(*header), fmt.format(*("-" * w for w in widths))] + [
        fmt.format(*b) for b in body]


def _sort_key(v):
    n = _number(v)
    return (0, n, "") if n is not None else (1, 0, str(v))


def report(rows, failed=None, campaigns=None) -> str:
    """The whole report: every campaign in `rows` (or those named), every
    group, measured configurations first and holes counted after.
    `failed` maps (campaign, group) to how many points failed."""
    failed = failed or {}
    out = []
    by_group = defaultdict(list)
    for r in rows:
        if campaigns and r.get("campaign") not in campaigns:
            continue
        by_group[(r.get("campaign"), r.get("group"))].append(r)
    for (campaign, group) in sorted(set(by_group) | {k for k in failed
                                                     if not campaigns or k[0] in campaigns}):
        rs = by_group.get((campaign, group), [])
        measured = [r for r in rs if r.get("outcome") == "measured"]
        other = defaultdict(int)
        for r in rs:
            if r.get("outcome") != "measured":
                other[r.get("outcome") or "?"] += 1
        if failed.get((campaign, group)):
            other["failed"] += failed[(campaign, group)]
        kind = (measured or rs or [{}])[0].get("kind", "?")
        out.append(f"=== {campaign} / {group} ({kind}, {len(measured)} measured) ===")
        if measured:
            out.extend("  " + line for line in table(measured, kind))
        if other:
            out.append("  not measured: " + ", ".join(f"{k} {v}" for k, v in sorted(other.items())))
        out.append("")
    return "\n".join(out)


# ── a round against the ones before it ────────────────────────────────────

# What a configuration is read by when it is compared across engine builds,
# the first that both measurements have deciding. A throughput point is read
# by its decode speed first: its tokens/s over a request with a long prompt
# is mostly how many tokens the model wrote before it stopped, which the
# engine does not choose (Qwen3.6-35B-A3B with an 8,000-token prompt wrote
# 11.6 to 32.1 tokens/s from round to round at the same 83-92 of decode and
# a first token at 3.0-3.3 s).
HEADLINE = {"throughput": (("decode_tok_s", "decode"), ("agg_tok_s_mean", "tok/s")),
            "workload": (("agg_tok_s_mean", "tok/s"),),
            "decision": (("dec_per_s", "dec/s"),)}
# What makes two results the same configuration: what the point was planned
# with, not the round, the job or the engine that measured it.
CONFIG = ("campaign", "group", "kind", "model", "runtime", "gcds", "replicas", "mode",
          "batch", "ctx", "slot_ctx", "kv", "concurrency", "prompt_tokens", "decision_mode",
          "state_tokens", "questions", "extra_args", "env", "trial")
# What a row is named by in the comparison, when it is not the group's only one.
LABEL = ("model", "runtime", "gcds", "replicas", "batch", "concurrency", "slot_ctx",
         "prompt_tokens", "kv", "decision_mode", "state_tokens", "questions", "extra_args",
         "env")


def _label(row, params=LABEL) -> str:
    parts = []
    for p in params:
        v = row.get(p)
        if v in (None, "", "-") or (p == "runtime" and v == "eullm"):
            continue
        v = _cell(p, v)
        parts.append(v if p == "model" else f"{p}={v}")
    return " ".join(parts) or "-"


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2) if n else None


def _avg(rows, column):
    nums = [n for n in (_number(r.get(column)) for r in rows) if n is not None]
    return sum(nums) / len(nums) if nums else None


def compare(rows, is_new, threshold=0.95) -> str:
    """Every configuration measured by the rows `is_new` picks against its
    last earlier measurement: the mean of each of its headline metrics
    (decode and tokens/s for throughput, tokens/s for workloads,
    decisions/s) in the new rows over the mean in the latest earlier round
    that measured it. One line a group: how many configurations, the median
    ratio of the first metric and its range, the other metrics' medians, and
    the slowest configuration by name; a group whose median is below
    `threshold` is marked. Configurations measured for the first time are
    counted, not compared."""
    new, old = defaultdict(list), defaultdict(list)
    for r in rows:
        if r.get("outcome") != "measured" or r.get("kind") not in HEADLINE:
            continue
        if all(_number(r.get(c)) is None for c, _ in HEADLINE[r["kind"]]):
            continue
        key = tuple(r.get(k) for k in CONFIG)
        (new if is_new(r) else old)[key].append(r)
    groups = defaultdict(list)
    first = defaultdict(int)
    for key, rs in new.items():
        latest = max((r.get("ended") or "" for r in rs), default="")
        before = [r for r in old.get(key, []) if not latest or (r.get("ended") or "") < latest]
        if not before:
            first[(key[0], key[1])] += 1
            continue
        last_round = max(before, key=lambda r: r.get("ended") or "").get("round")
        ref = [r for r in before if r.get("round") == last_round]
        ratios = {}
        for column, name in HEADLINE[rs[0]["kind"]]:
            now, then = _avg(rs, column), _avg(ref, column)
            if now is not None and then:
                ratios[name] = (now / then, now, then)
        if ratios:
            groups[(key[0], key[1])].append((ratios, last_round, rs[0]))
    out, primaries = [], []
    for g in sorted(set(groups) | set(first)):
        cs = groups.get(g, [])
        line = f"{g[0]} / {g[1]}: "
        if cs:
            kind = cs[0][2].get("kind")
            names = [n for _, n in HEADLINE[kind] if any(n in c[0] for c in cs)]
            main = names[0]
            main_cs = [c for c in cs if main in c[0]]
            rs = [c[0][main][0] for c in main_cs]
            primaries.extend(rs)
            med = _median(rs)
            slow = min(main_cs, key=lambda c: c[0][main][0])
            ratio, now, then = slow[0][main]
            # named by what varies among the group's configurations
            varying = [p for p in LABEL if len({c[2].get(p) for c in cs}) > 1] or ["model"]
            line += f"{len(cs)} compared, {main} new/old {med:.2f} ({min(rs):.2f}-{max(rs):.2f})"
            for other in names[1:]:
                line += f", {other} {_median([c[0][other][0] for c in cs if other in c[0]]):.2f}"
            line += (f"; slowest {_label(slow[2], varying)} {ratio:.2f}"
                     f" ({then:.4g} -> {now:.4g}, {slow[1]})")
            if med < threshold:
                line = "[!!] " + line
        else:
            line += "nothing measured before"
        if first.get(g):
            line += f"; {first[g]} first measured"
        out.append(line)
    if primaries:
        out.append(f"all: {len(primaries)} configurations compared, new/old median "
                   f"{_median(primaries):.2f} ({min(primaries):.2f}-{max(primaries):.2f}) "
                   f"by each group's first metric, {sum(r < threshold for r in primaries)} "
                   f"below {threshold}")
    else:
        out.append("nothing to compare: no configuration measured before")
    return "\n".join(out)
