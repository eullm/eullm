#!/usr/bin/env python3
"""Generate quality comparison charts from per-arm JSON results.

Reads bench/results/quality_*.json (produced by `turboquant_quality.py collect`)
and emits chart_quality_comparison.png + chart_quality_radar.png.

Arms recognized (any subset can be present):
  F16, Q8_0, Q4_0, TQ4_0, TQ3_0

The label inside each JSON drives ordering and styling.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Display order + style for each known arm.
ARM_STYLE = {
    "F16":   {"color": "#6c757d", "label": "F16 (baseline)"},
    "Q8_0":  {"color": "#212529", "label": "Q8_0 (8-bit native llama.cpp)"},
    "Q4_0":  {"color": "#dc3545", "label": "Q4_0 (4-bit native llama.cpp)"},
    "TQ4_0": {"color": "#0d6efd", "label": "TQ4_0 (4-bit TurboQuant)"},
    "TQ3_0": {"color": "#198754", "label": "TQ3_0 (3-bit TurboQuant)"},
}
ARM_ORDER = ["F16", "Q8_0", "Q4_0", "TQ4_0", "TQ3_0"]

CATEGORIES = ["matrix", "math", "factual", "logic", "code"]
CATEGORY_DISPLAY = ["Matrix", "Math", "Factual", "Logic", "Code"]


def load_arms(pattern: str) -> dict:
    """Load every quality_*.json matching the pattern. Returns {label: data}."""
    arms = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            data = json.load(fh)
        label = data.get("label", os.path.basename(path))
        arms[label.upper()] = data
    return arms


def ordered_arms(arms: dict) -> list:
    """Return arms in canonical display order; unknown labels appended at end."""
    known = [a for a in ARM_ORDER if a in arms]
    extra = [a for a in arms if a not in ARM_ORDER]
    return known + extra


def bar_chart(arms: dict, out_path: str) -> None:
    labels = ordered_arms(arms)
    if not labels:
        print("No arms to plot — skipping bar chart.")
        return

    # Build matrix [arm][category] of passed counts and totals.
    passed = {a: [] for a in labels}
    totals = {a: [] for a in labels}
    for a in labels:
        cats = arms[a].get("categories", {})
        for c in CATEGORIES:
            v = cats.get(c, {"passed": 0, "total": 0})
            passed[a].append(v["passed"])
            totals[a].append(v["total"])

    # Add TOTAL column
    cat_display = CATEGORY_DISPLAY + ["TOTAL"]
    for a in labels:
        passed[a].append(sum(passed[a]))
        totals[a].append(sum(totals[a]))

    n_arms = len(labels)
    width = 0.8 / max(n_arms, 1)
    x = np.arange(len(cat_display))

    fig, ax = plt.subplots(figsize=(15, 7.5))
    for i, a in enumerate(labels):
        style = ARM_STYLE.get(a, {"color": "#888", "label": a})
        offset = (i - (n_arms - 1) / 2) * width
        pcts = [p / t * 100 if t else 0 for p, t in zip(passed[a], totals[a])]
        bars = ax.bar(x + offset, pcts, width, label=style["label"],
                      color=style["color"], edgecolor="white")
        for bar, p, t in zip(bars, passed[a], totals[a]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{p}/{t}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    # Metadata from any arm
    meta = next(iter(arms.values()))
    model = meta.get("model", "unknown model")
    temp = meta.get("temperature", "?")
    total_tests = meta.get("score", {}).get("total", "?")

    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(
        f"KV Cache Quality Impact — {total_tests} Verified Tests\n"
        f"{model}, temperature={temp}",
        fontsize=15, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cat_display, fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc="lower right", ncol=1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def radar_chart(arms: dict, out_path: str) -> None:
    labels = ordered_arms(arms)
    if not labels:
        print("No arms to plot — skipping radar chart.")
        return

    angles = np.linspace(0, 2 * np.pi, len(CATEGORIES), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for a in labels:
        style = ARM_STYLE.get(a, {"color": "#888", "label": a})
        cats = arms[a].get("categories", {})
        vals = []
        for c in CATEGORIES:
            v = cats.get(c, {"passed": 0, "total": 1})
            vals.append(v["passed"] / max(v["total"], 1) * 100)
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2.5, label=style["label"],
                color=style["color"], markersize=7)
        ax.fill(angles, vals, alpha=0.08, color=style["color"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORY_DISPLAY, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=9)

    meta = next(iter(arms.values()))
    model = meta.get("model", "unknown model")
    total_tests = meta.get("score", {}).get("total", "?")
    ax.set_title(
        f"Quality by Category — {total_tests} Tests, {model}",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.legend(loc="lower right", fontsize=10, bbox_to_anchor=(1.25, -0.05))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS_DIR, "quality_*.json")
    arms = load_arms(pattern)
    if not arms:
        print(f"ERROR: no JSON files matched {pattern}", file=sys.stderr)
        print("Run `bench/run_quality_arms.sh` first to produce per-arm results.",
              file=sys.stderr)
        return 1

    print(f"Loaded arms: {', '.join(ordered_arms(arms))}")
    bar_chart(arms, os.path.join(RESULTS_DIR, "chart_quality_comparison.png"))
    radar_chart(arms, os.path.join(RESULTS_DIR, "chart_quality_radar.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
