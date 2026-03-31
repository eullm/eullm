#!/usr/bin/env python3
"""Generate quality comparison chart for TurboQuant README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data from 100-test benchmark ─────────────────────────────────────────────

categories = ["Matrix", "Math", "Factual", "Logic", "Code", "TOTAL"]

f16   = [18, 18, 15, 17, 18, 86]
tq4_0 = [17, 18, 15, 17, 18, 85]
tq3_0 = [17, 18, 15, 17, 18, 85]

totals = [20, 20, 20, 20, 20, 100]

f16_pct   = [v/t*100 for v, t in zip(f16, totals)]
tq4_pct   = [v/t*100 for v, t in zip(tq4_0, totals)]
tq3_pct   = [v/t*100 for v, t in zip(tq3_0, totals)]

# ── Chart: Quality comparison bar chart ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(categories))
width = 0.25

bars1 = ax.bar(x - width, f16_pct, width, label="F16 (baseline)", color="#6c757d", edgecolor="white")
bars2 = ax.bar(x, tq4_pct, width, label="TQ4_0 (4-bit KV)", color="#0d6efd", edgecolor="white")
bars3 = ax.bar(x + width, tq3_pct, width, label="TQ3_0 (3-bit KV)", color="#198754", edgecolor="white")

# Add value labels
for bars, values, total_vals in [(bars1, f16, totals), (bars2, tq4_0, totals), (bars3, tq3_0, totals)]:
    for bar, val, tot in zip(bars, values, total_vals):
        pct = val/tot*100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val}/{tot}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Styling
ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("TurboQuant Quality Impact — 100 Verified Tests\n"
             "Qwen3-14B Q4_K_M, temperature=0, RTX 5070 Ti 16GB",
             fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 105)
ax.legend(fontsize=12, loc="lower right")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

# Add annotation
ax.annotate("1% degradation\n4.3× more context",
            xy=(5, 87), fontsize=14, fontweight="bold",
            color="#198754", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#d4edda", edgecolor="#198754", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_quality_comparison.png"), dpi=150)
print(f"Saved: {OUTPUT_DIR}/chart_quality_comparison.png")

# ── Chart 2: Radar/spider chart ──────────────────────────────────────────────

cats_radar = ["Matrix", "Math", "Factual", "Logic", "Code"]
f16_r = [90, 90, 75, 85, 90]
tq4_r = [85, 90, 75, 85, 90]
tq3_r = [85, 90, 75, 85, 90]

angles = np.linspace(0, 2 * np.pi, len(cats_radar), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

f16_r += f16_r[:1]
tq4_r += tq4_r[:1]
tq3_r += tq3_r[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

ax.plot(angles, f16_r, 'o-', linewidth=2.5, label='F16', color='#6c757d', markersize=8)
ax.fill(angles, f16_r, alpha=0.1, color='#6c757d')

ax.plot(angles, tq4_r, 's-', linewidth=2.5, label='TQ4_0', color='#0d6efd', markersize=8)
ax.fill(angles, tq4_r, alpha=0.1, color='#0d6efd')

ax.plot(angles, tq3_r, '^-', linewidth=2.5, label='TQ3_0', color='#198754', markersize=8)
ax.fill(angles, tq3_r, alpha=0.1, color='#198754')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats_radar, fontsize=13)
ax.set_ylim(0, 100)
ax.set_yticks([25, 50, 75, 100])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=9)
ax.set_title("Quality by Category\n100 Tests, Qwen3-14B", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="lower right", fontsize=11, bbox_to_anchor=(1.15, -0.05))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_quality_radar.png"), dpi=150)
print(f"Saved: {OUTPUT_DIR}/chart_quality_radar.png")
