#!/usr/bin/env python3
"""Generate comparison charts from TurboQuant benchmark results."""

import json
import os
import statistics

# ── Try matplotlib, fall back to text-only ──────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")
    print("         Generating text summary only.\n")

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = RESULTS_DIR

# ── Load data ────────────────────────────────────────────────────────────────

FILES = {
    "F16 (ctx=30K)": os.path.join(RESULTS_DIR, "f16.json"),
    "TQ4_0 (ctx=131K)": os.path.join(RESULTS_DIR, "tq4_0.json"),
    "TQ3_0 (ctx=131K)": os.path.join(RESULTS_DIR, "tq3_0.json"),
}

data = {}
for label, path in FILES.items():
    if os.path.exists(path):
        with open(path) as f:
            data[label] = json.load(f)

if not data:
    print("ERROR: No result files found")
    exit(1)


def aggregate_by_concurrency(results):
    """Group rounds by concurrency, compute median of medians."""
    by_conc = {}
    for r in results:
        c = r["concurrency"]
        if c not in by_conc:
            by_conc[c] = {"ttft_p50": [], "tps_p50": [], "agg_tps": [], "wall": []}
        by_conc[c]["ttft_p50"].append(r["ttft"]["p50"])
        by_conc[c]["tps_p50"].append(r["tokens_per_sec"]["p50"])
        by_conc[c]["agg_tps"].append(r["aggregate_throughput"])
        by_conc[c]["wall"].append(r["wall_ms"])
    out = {}
    for c, vals in sorted(by_conc.items()):
        out[c] = {
            "ttft_p50": statistics.median(vals["ttft_p50"]),
            "tps_p50": statistics.median(vals["tps_p50"]),
            "agg_tps": statistics.median(vals["agg_tps"]),
            "wall": statistics.median(vals["wall"]),
        }
    return out


agg = {label: aggregate_by_concurrency(d["results"]) for label, d in data.items()}

# ── Text summary ─────────────────────────────────────────────────────────────

print("=" * 70)
print("  TurboQuant Benchmark Results — Qwen3-14B @ RTX 5070 Ti 16GB")
print("=" * 70)
print()
print(f"{'Cache Type':<22} {'Conc':>4} {'TTFT P50':>10} {'tok/s':>8} {'Agg tok/s':>10} {'Wall':>10}")
print("-" * 70)
for label, concs in agg.items():
    for c, v in concs.items():
        print(f"{label:<22} {c:>4} {v['ttft_p50']:>9.1f}ms {v['tps_p50']:>7.1f} {v['agg_tps']:>9.1f} {v['wall']:>9.0f}ms")
print()

# KV cache VRAM comparison
print("  KV Cache VRAM (estimated):")
print("  F16  @ 30K ctx:  K=480 MiB + V=480 MiB  = ~960 MiB")
print("  TQ4_0 @ 131K ctx: K=2560 MiB + V=2560 MiB = ~5120 MiB")
print("  TQ3_0 @ 131K ctx: K=1920 MiB + V=1920 MiB = ~3840 MiB")
print()
print("  F16 max context on 16GB: ~30K tokens")
print("  TQ4_0 context on 16GB:  131K tokens (4.3x more)")
print("  TQ3_0 context on 16GB:  131K tokens (4.3x more)")
print()

if not HAS_MPL:
    print("Install matplotlib for charts: pip install matplotlib")
    exit(0)

# ── Colors ───────────────────────────────────────────────────────────────────

COLORS = {
    "F16 (ctx=30K)": "#6c757d",       # gray
    "TQ4_0 (ctx=131K)": "#0d6efd",    # blue
    "TQ3_0 (ctx=131K)": "#198754",    # green
}

# ── Chart 1: Throughput by concurrency ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
conc_levels = sorted(set(c for concs in agg.values() for c in concs))
bar_width = 0.25
x_pos = range(len(conc_levels))

for i, (label, concs) in enumerate(agg.items()):
    values = [concs.get(c, {}).get("agg_tps", 0) for c in conc_levels]
    offset = (i - 1) * bar_width
    bars = ax.bar([x + offset for x in x_pos], values, bar_width,
                  label=label, color=COLORS.get(label, "#333"), edgecolor="white")
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlabel("Concurrent Requests", fontsize=12)
ax.set_ylabel("Aggregate Throughput (tok/s)", fontsize=12)
ax.set_title("TurboQuant KV Cache — Throughput Comparison\nQwen3-14B @ RTX 5070 Ti 16GB", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(conc_levels)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_throughput.png"), dpi=150)
print(f"  Saved: chart_throughput.png")

# ── Chart 2: TTFT by concurrency ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

for i, (label, concs) in enumerate(agg.items()):
    values = [concs.get(c, {}).get("ttft_p50", 0) for c in conc_levels]
    offset = (i - 1) * bar_width
    bars = ax.bar([x + offset for x in x_pos], values, bar_width,
                  label=label, color=COLORS.get(label, "#333"), edgecolor="white")
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlabel("Concurrent Requests", fontsize=12)
ax.set_ylabel("Time to First Token (ms)", fontsize=12)
ax.set_title("TurboQuant KV Cache — TTFT Comparison\nQwen3-14B @ RTX 5070 Ti 16GB", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(conc_levels)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_ttft.png"), dpi=150)
print(f"  Saved: chart_ttft.png")

# ── Chart 3: Context capacity (bar chart) ───────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
cache_types = ["F16", "TQ4_0", "TQ3_0"]
max_ctx = [30720, 131072, 131072]
colors = ["#6c757d", "#0d6efd", "#198754"]
kv_vram = [960, 5120, 3840]  # MiB

bars = ax.bar(cache_types, [c / 1024 for c in max_ctx], color=colors, edgecolor="white", width=0.5)
for bar, ctx, vram in zip(bars, max_ctx, kv_vram):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{ctx//1024}K ctx\nKV: {vram/1024:.1f} GB",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("Max Context (K tokens)", fontsize=12)
ax.set_title("Max Context Window on RTX 5070 Ti 16GB\nQwen3-14B Q4_K_M", fontsize=14, fontweight="bold")
ax.set_ylim(0, 160)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}K"))
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

# Add 16GB VRAM line
ax.axhline(y=16*1024/1024, color="red", linestyle="--", alpha=0.0)  # invisible, just for reference

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_context_capacity.png"), dpi=150)
print(f"  Saved: chart_context_capacity.png")

# ── Chart 4: Per-request tok/s at concurrency=4 ─────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

for label, concs in agg.items():
    if 4 in concs:
        v = concs[4]
        ax.bar(label, v["tps_p50"], color=COLORS.get(label, "#333"), edgecolor="white", width=0.4)
        ax.text(ax.patches[-1].get_x() + ax.patches[-1].get_width()/2,
                v["tps_p50"] + 0.5,
                f"{v['tps_p50']:.1f} t/s",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Per-Request Throughput (tok/s)", fontsize=12)
ax.set_title("Per-Request Speed @ 4 Concurrent\nQwen3-14B @ RTX 5070 Ti 16GB", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_per_request_speed.png"), dpi=150)
print(f"  Saved: chart_per_request_speed.png")

print()
print(f"All charts saved to: {OUTPUT_DIR}")
