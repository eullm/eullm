#!/usr/bin/env python3
"""Generate GPU scaling projection chart for TurboQuant."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data ─────────────────────────────────────────────────────────────────────

gpus = ["RTX 5070 Ti\n16 GB", "RTX 5090\n32 GB", "A100\n80 GB", "H100\n80 GB"]

# Max concurrent slots at 8K context per slot for Qwen3-14B (~9GB model weights)
# Available VRAM for KV = total - model_weights
# KV per slot (F16): 2 * 40layers * 8192ctx * 8heads * 128dim * 2bytes = 1.31 GB
# KV per slot (TQ4_0): 2 * 40 * 8192 * 8 * 128 * 0.5 = 0.33 GB

model_vram = 9.0  # GB for 14B Q4_K_M
kv_per_slot_f16 = 1.31   # GB
kv_per_slot_tq4 = 0.33   # GB

vram_total = [16, 32, 80, 80]
slots_f16 = [max(1, int((v - model_vram) / kv_per_slot_f16)) for v in vram_total]
slots_tq4 = [max(1, int((v - model_vram) / kv_per_slot_tq4)) for v in vram_total]

# ── Chart: Concurrent Slots ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(gpus))
width = 0.35

bars_f16 = ax.bar(x - width/2, slots_f16, width, label="F16 KV Cache",
                   color="#6c757d", edgecolor="white")
bars_tq4 = ax.bar(x + width/2, slots_tq4, width, label="TQ4_0 KV Cache",
                   color="#0d6efd", edgecolor="white")

for bar, val in zip(bars_f16, slots_f16):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha="center", va="bottom", fontsize=12, fontweight="bold", color="#6c757d")

for bar, val, f16_val in zip(bars_tq4, slots_tq4, slots_f16):
    ratio = val / f16_val if f16_val > 0 else 0
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{val}\n({ratio:.0f}x)", ha="center", va="bottom", fontsize=12,
            fontweight="bold", color="#0d6efd")

ax.set_ylabel("Concurrent Slots (8K ctx each)", fontsize=13)
ax.set_title("TurboQuant Enterprise Scaling — Concurrent Users per GPU\n"
             "Qwen3-14B Q4_K_M @ 8K Context per Slot",
             fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(gpus, fontsize=12)
ax.legend(fontsize=12, loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_gpu_scaling.png"), dpi=150)
print(f"Saved: chart_gpu_scaling.png")

# ── Chart: Cost savings ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

scenarios = ["1000 users\nF16", "1000 users\nTQ4_0", "3000 users\nF16", "3000 users\nTQ4_0"]
# H100 slots: F16=54, TQ4_0=215
# Nodes needed = ceil(users / slots_per_node)
import math
h100_f16_slots = slots_f16[3]  # ~54
h100_tq4_slots = slots_tq4[3]  # ~215
cost_per_node = 30  # EUR K/month

nodes_1k_f16 = math.ceil(1000 / h100_f16_slots)
nodes_1k_tq4 = math.ceil(1000 / h100_tq4_slots)
nodes_3k_f16 = math.ceil(3000 / h100_f16_slots)
nodes_3k_tq4 = math.ceil(3000 / h100_tq4_slots)

costs = [nodes_1k_f16 * cost_per_node,
         nodes_1k_tq4 * cost_per_node,
         nodes_3k_f16 * cost_per_node,
         nodes_3k_tq4 * cost_per_node]

colors = ["#dc3545", "#198754", "#dc3545", "#198754"]

bars = ax.bar(scenarios, costs, color=colors, edgecolor="white", width=0.6)
for bar, cost, nodes in zip(bars, costs,
                             [nodes_1k_f16, nodes_1k_tq4, nodes_3k_f16, nodes_3k_tq4]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"EUR {cost}K/mo\n({nodes} nodes)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

# Add savings annotations
saving_1k = (nodes_1k_f16 - nodes_1k_tq4) * cost_per_node
saving_3k = (nodes_3k_f16 - nodes_3k_tq4) * cost_per_node
ax.annotate(f"Save EUR {saving_1k}K/mo",
            xy=(0.5, max(costs[0], costs[1])/2), fontsize=13, fontweight="bold",
            color="#198754", ha="center")
ax.annotate(f"Save EUR {saving_3k}K/mo",
            xy=(2.5, max(costs[2], costs[3])/2), fontsize=13, fontweight="bold",
            color="#198754", ha="center")

ax.set_ylabel("Infrastructure Cost (EUR K/month)", fontsize=12)
ax.set_title("TurboQuant Infrastructure Cost Savings\n"
             "H100 80GB nodes @ EUR 30K/month — Qwen3-14B, 8K ctx/user",
             fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart_cost_savings.png"), dpi=150)
print(f"Saved: chart_cost_savings.png")
