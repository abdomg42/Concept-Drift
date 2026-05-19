"""
DRL Concept Drift – Full Analysis
===================================
Generates all report figures and prints evaluation tables.
Run as: python notebooks/analysis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# -------------------------------------------------------
# Load results
# -------------------------------------------------------
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "all_results.json")
PLOTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

with open(RESULTS_PATH) as f:
    all_results = json.load(f)

print(f"Loaded {len(all_results)} runs.")
for k in all_results:
    s = all_results[k]["summary"]
    print(f"  {k:45s}  pre={s['mean_pre_drift_return']:6.1f}  "
          f"during={s['mean_during_drift_return']:7.1f}  "
          f"det_rate={s['detection_rate']:.2f}  FAR={s['false_alarm_rate']:.2f}")


def smooth(x, k=20):
    arr = np.array(x, dtype=float)
    if len(arr) < k:
        return arr
    return np.convolve(arr, np.ones(k)/k, mode="valid")


COLORS = {
    "fine_tune": "#1565C0",
    "maml":      "#2E7D32",
    "ensemble":  "#E65100",
}

# -------------------------------------------------------
# Figure 1 – Learning curves comparison for CartPole-Gravity
# -------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle("Figure 1 – CartPole-Gravity: Strategy Comparison", fontsize=13, fontweight="bold")

ax_ret, ax_eps = axes

for strategy, color in COLORS.items():
    key = f"CartPole-Gravity__{strategy}"
    if key not in all_results:
        continue
    r       = all_results[key]
    returns = np.array(r["episode_returns"])
    eps     = np.array(r["epsilon_trace"])
    sm      = smooth(returns)
    x_sm    = np.arange(1, len(sm)+1)
    ax_ret.plot(np.arange(1, len(returns)+1), returns, alpha=0.15, color=color, lw=0.6)
    ax_ret.plot(x_sm, sm, color=color, lw=2.0, label=strategy)
    ax_eps.plot(np.arange(1, len(eps)+1), eps, color=color, lw=1.5, alpha=0.8)

# Drift marker
ax_ret.axvline(150, color="#C62828", ls="--", lw=2, label="True drift (ep 150)")
ax_ret.set_ylabel("Episode Return", fontsize=11)
ax_ret.legend(fontsize=9)
ax_ret.grid(alpha=0.25)
ax_ret.set_title("Smoothed Return by Strategy")

# Detection timestamps for fine_tune
for ep in all_results.get("CartPole-Gravity__fine_tune", {}).get("detection_timestamps", []):
    ax_ret.axvline(ep, color="#7B1FA2", ls=":", lw=0.9, alpha=0.6)

ax_eps.axvline(150, color="#C62828", ls="--", lw=2)
ax_eps.set_ylabel("Epsilon", fontsize=11)
ax_eps.set_xlabel("Episode", fontsize=11)
ax_eps.set_ylim(0, 1.05)
ax_eps.grid(alpha=0.25)
ax_eps.set_title("Exploration Rate (spikes = drift response)")

legend_extra = [
    mpatches.Patch(color="#C62828", label="True drift"),
    mpatches.Patch(color="#7B1FA2", label="Detected drift (fine_tune)"),
]
ax_ret.legend(handles=ax_ret.get_legend_handles_labels()[0] + legend_extra,
              labels=ax_ret.get_legend_handles_labels()[1] + [p.get_label() for p in legend_extra],
              fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig1_learning_curves.png"), dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig1_learning_curves.png")


# -------------------------------------------------------
# Figure 2 – Multi-environment performance heatmap
# -------------------------------------------------------
metrics_to_show = [
    ("mean_pre_drift_return",    "Pre-drift\nReturn"),
    ("mean_during_drift_return", "During-drift\nReturn"),
    ("mean_post_adapt_return",   "Post-adapt\nReturn"),
    ("detection_rate",           "Detection\nRate"),
    ("false_alarm_rate",         "False\nAlarm Rate"),
    ("mean_detection_delay",     "Detect\nDelay"),
    ("catastrophic_forgetting",  "Catastrophic\nForgetting"),
    ("transfer_efficiency",      "Transfer\nEfficiency"),
]

run_names = list(all_results.keys())
data = np.zeros((len(run_names), len(metrics_to_show)))
for i, rn in enumerate(run_names):
    s = all_results[rn]["summary"]
    for j, (mk, _) in enumerate(metrics_to_show):
        v = s.get(mk, 0.0)
        if v is None or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
            v = 0.0
        data[i, j] = float(v)

# Normalize
col_min = data.min(axis=0)
col_max = data.max(axis=0)
col_range = np.where(col_max - col_min > 1e-8, col_max - col_min, 1.0)
norm = (data - col_min) / col_range

fig, ax = plt.subplots(figsize=(14, max(4, len(run_names)*0.55 + 1.5)))
im = ax.imshow(norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(metrics_to_show)))
ax.set_xticklabels([m[1] for m in metrics_to_show], fontsize=8)
ax.set_yticks(range(len(run_names)))
ax.set_yticklabels(run_names, fontsize=8)
for i in range(len(run_names)):
    for j in range(len(metrics_to_show)):
        ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="black" if 0.25 < norm[i,j] < 0.75 else "white")
plt.colorbar(im, ax=ax, label="Normalised (green=better)")
ax.set_title("Figure 2 – Experiment Metrics Heatmap", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig2_metrics_heatmap.png"), dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig2_metrics_heatmap.png")


# -------------------------------------------------------
# Figure 3 – Reward distribution shift visualization
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Figure 3 – Reward Distribution Shift by Environment", fontsize=13, fontweight="bold")

env_configs = [
    ("CartPole-Gravity__fine_tune",  "CartPole – Gravity Drift"),
    ("CartPole-Length__fine_tune",   "CartPole – Length Drift"),
    ("CartPole-Reward__fine_tune",   "CartPole – Reward Drift"),
]
DRIFT_EP = 150

for ax, (key, title) in zip(axes, env_configs):
    if key not in all_results:
        ax.set_visible(False)
        continue
    returns = np.array(all_results[key]["episode_returns"])
    pre     = returns[:DRIFT_EP]
    post    = returns[DRIFT_EP:]
    ax.hist(pre,  bins=25, alpha=0.65, color="#1565C0", label=f"Pre-drift  (μ={pre.mean():.1f})")
    ax.hist(post, bins=25, alpha=0.65, color="#C62828", label=f"Post-drift (μ={post.mean():.1f})")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Episode Return")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig3_reward_distributions.png"), dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig3_reward_distributions.png")


# -------------------------------------------------------
# Figure 4 – Detection timeline (all runs)
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, max(4, len(all_results)*0.45 + 1.5)))
fig.suptitle("Figure 4 – Drift Detection Timelines", fontsize=13, fontweight="bold")

y_labels = []
for idx, (run_name, r) in enumerate(all_results.items()):
    y = idx
    y_labels.append(run_name)
    gt_flags = np.array(r["ground_truth_flags"])
    if gt_flags.any():
        gt_ep = int(np.argmax(gt_flags)) + 1
        ax.axvline(gt_ep, color="#C62828", ls="--", lw=1.5, alpha=0.4)
    for ep in r.get("detection_timestamps", []):
        col = COLORS.get(run_name.split("__")[-1], "#7B1FA2")
        ax.scatter(ep, y, marker="|", s=120, color=col, linewidths=2)

ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=8)
ax.axvline(150, color="#C62828", ls="--", lw=2, label="True drift ep 150")
ax.set_xlabel("Episode", fontsize=11)
ax.set_title("Each tick = a detected drift event", fontsize=9)
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig4_detection_timeline.png"), dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig4_detection_timeline.png")


# -------------------------------------------------------
# Figure 5 – System Architecture Diagram (SVG-style in matplotlib)
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")
fig.suptitle("Figure 5 – DRL Concept Drift System Architecture", fontsize=13, fontweight="bold")

def draw_box(ax, x, y, w, h, title, body, fc, ec="#333"):
    rect = plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.3, title, ha="center", va="top",
            fontsize=9, fontweight="bold", zorder=4, color="#111")
    for i, line in enumerate(body):
        ax.text(x + 0.15, y + h - 0.65 - i*0.32, f"• {line}",
                ha="left", va="top", fontsize=7, zorder=4, color="#333")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5), zorder=5)

# Environment
draw_box(ax, 0.3, 4.0, 2.8, 2.5, "ENVIRONMENT",
         ["CartPole variants", "MountainCar", "Drift injection", "Gravity / length / reward"],
         fc="#E3F2FD")

# Agent
draw_box(ax, 3.7, 4.0, 2.8, 2.5, "DQL AGENT (DQN)",
         ["Encoder (latent z)", "Ensemble Q-heads (×3)", "Dueling arch.", "Soft target update"],
         fc="#F3E5F5")

# Replay buffer
draw_box(ax, 3.7, 1.0, 2.8, 2.5, "ADAPTIVE REPLAY",
         ["Prioritized (PER)", "Recency weighting", "Partial reset on drift", "TD-error priorities"],
         fc="#FFF8E1")

# Detection
draw_box(ax, 7.1, 4.0, 2.8, 2.5, "DRIFT DETECTION",
         ["Performance monitor", "KS Test", "Page-Hinkley", "ADWIN · Quorum vote"],
         fc="#E8F5E9")

# Adaptation
draw_box(ax, 10.5, 4.0, 2.8, 2.5, "ADAPTATION",
         ["Fine-tune (warm start)", "MAML adapter", "Multi-policy ensemble", "Progressive net"],
         fc="#FBE9E7")

# Evaluation
draw_box(ax, 7.1, 1.0, 6.2, 2.5, "EVALUATION FRAMEWORK",
         ["Detection delay · False alarm rate · Recovery time",
          "Pre/during/post returns · Catastrophic forgetting",
          "Transfer efficiency · Learning curves",
          "Metrics heatmap · Detection timeline"],
         fc="#F5F5F5")

# Arrows
arrow(ax, 3.1,  5.25, 3.7,  5.25)   # Env → Agent
arrow(ax, 5.0,  4.0,  5.0,  3.5)    # Agent → Buffer
arrow(ax, 5.0,  3.5,  5.0,  3.5)
arrow(ax, 5.3,  3.5,  5.3,  4.0)    # Buffer → Agent (samples)
arrow(ax, 6.5,  5.25, 7.1,  5.25)   # Agent → Detection
arrow(ax, 9.9,  5.25, 10.5, 5.25)   # Detection → Adaptation
arrow(ax, 11.9, 4.0,  8.5,  3.5)    # Adaptation → Eval
arrow(ax, 8.5,  4.0,  8.5,  3.5)    # Detection → Eval

ax.text(7.0, 6.8, "Step: obs → action → reward → detect → adapt → learn",
        ha="center", fontsize=8, style="italic", color="#555")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig5_architecture.png"), dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig5_architecture.png")


# -------------------------------------------------------
# Print final summary table
# -------------------------------------------------------
print("\n" + "="*90)
print(f"{'RUN':<45} {'Pre':>6} {'During':>8} {'Post':>6} {'Det%':>6} {'FAR':>5} {'Delay':>6}")
print("="*90)
for rn, r in all_results.items():
    s = r["summary"]
    delay = s.get("mean_detection_delay", float("inf"))
    delay_str = f"{delay:.1f}" if not np.isinf(delay) else "  inf"
    print(f"{rn:<45} {s['mean_pre_drift_return']:6.1f} "
          f"{s['mean_during_drift_return']:8.1f} "
          f"{s['mean_post_adapt_return']:6.1f} "
          f"{s['detection_rate']:6.2f} "
          f"{s['false_alarm_rate']:5.2f} "
          f"{delay_str:>6}")
print("="*90)
print("\nAll analysis complete. Plots saved to results/plots/")
