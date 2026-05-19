"""
Visualization Module
====================
Generates all analysis plots:
- Learning curves with drift markers
- Detection accuracy heatmap
- Comparison across strategies
- Q-value variance (ensemble disagreement)
- Catastrophic forgetting / recovery metrics bar charts
- t-SNE of latent embeddings (if embeddings recorded)
"""

from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Optional


PALETTE = {
    "fine_tune": "#2196F3",
    "maml":      "#4CAF50",
    "ensemble":  "#FF9800",
    "gt_drift":  "#F44336",
    "detected":  "#9C27B0",
}


def smooth(x, k=15):
    if len(x) < k:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(k) / k, mode="valid")


# -----------------------------------------------------------------------
# 1. Learning curve for a single run
# -----------------------------------------------------------------------
def plot_learning_curve(result: dict, title: str = "", save_path: Optional[str] = None):
    returns = np.array(result["episode_returns"])
    gt      = np.array(result["ground_truth_flags"])
    det     = np.array(result["drift_flags"])
    eps     = np.array(result["epsilon_trace"])
    episodes = np.arange(1, len(returns) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title or result.get("summary", {}).get("run_name", ""), fontsize=13, fontweight="bold")

    # --- Return ---
    ax = axes[0]
    smoothed = smooth(returns)
    x_smooth = np.arange(1, len(smoothed) + 1)
    ax.plot(episodes, returns, alpha=0.25, color="#90CAF9", lw=0.7)
    ax.plot(x_smooth, smoothed, color="#1565C0", lw=1.8, label="Smoothed return")

    # Ground-truth drift onset
    if gt.any():
        first_gt = int(np.argmax(gt)) + 1
        ax.axvline(first_gt, color=PALETTE["gt_drift"], ls="--", lw=1.5, label="True drift onset")

    # Detection timestamps
    for ep in result.get("detection_timestamps", []):
        ax.axvline(ep, color=PALETTE["detected"], ls=":", lw=1.2, alpha=0.8)
    ax.set_ylabel("Episode Return")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # --- Epsilon ---
    ax = axes[1]
    ax.plot(episodes, eps, color="#FF6F00", lw=1.5)
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # --- Q variance ---
    ax = axes[2]
    qv = np.array(result.get("q_variance_trace", [0] * len(episodes)))
    ax.plot(episodes, qv, color="#7B1FA2", lw=1.2, alpha=0.85)
    ax.set_ylabel("Q Ensemble Variance")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# -----------------------------------------------------------------------
# 2. Strategy comparison for a single environment
# -----------------------------------------------------------------------
def plot_strategy_comparison(all_results: dict, env_prefix: str,
                              save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Strategy Comparison — {env_prefix}", fontsize=13, fontweight="bold")

    for strategy, color in PALETTE.items():
        if strategy in ("gt_drift", "detected"):
            continue
        key = f"{env_prefix}__{strategy}"
        if key not in all_results:
            continue
        r = all_results[key]
        returns   = np.array(r["episode_returns"])
        smoothed  = smooth(returns, k=20)
        x_smooth  = np.arange(1, len(smoothed) + 1)

        axes[0].plot(x_smooth, smoothed, color=color, lw=1.8, label=strategy)
        axes[1].bar(strategy, r["summary"].get("mean_post_adapt_return", 0),
                    color=color, alpha=0.85)

    # Drift line
    for key, r in all_results.items():
        if key.startswith(env_prefix):
            gt = np.array(r["ground_truth_flags"])
            if gt.any():
                axes[0].axvline(int(np.argmax(gt)) + 1, color=PALETTE["gt_drift"],
                                ls="--", lw=1.5, label="True drift")
                break

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Smoothed Return")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("Mean Post-Adapt Return")
    axes[1].set_title("Post-Adaptation Performance")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# -----------------------------------------------------------------------
# 3. Metrics summary table / heatmap
# -----------------------------------------------------------------------
def plot_metrics_heatmap(all_results: dict, save_path: Optional[str] = None):
    metric_keys = [
        "mean_pre_drift_return",
        "mean_during_drift_return",
        "mean_post_adapt_return",
        "detection_rate",
        "false_alarm_rate",
        "mean_detection_delay",
        "mean_recovery_time",
        "catastrophic_forgetting",
        "transfer_efficiency",
    ]

    run_names = list(all_results.keys())
    data = np.zeros((len(run_names), len(metric_keys)))

    for i, rn in enumerate(run_names):
        s = all_results[rn]["summary"]
        for j, mk in enumerate(metric_keys):
            v = s.get(mk, 0.0)
            if v is None or (isinstance(v, float) and np.isinf(v)):
                v = 0.0
            data[i, j] = float(v)

    # Normalize columns for visual comparison
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    col_range = np.where(col_max - col_min > 1e-8, col_max - col_min, 1.0)
    data_norm = (data - col_min) / col_range

    fig, ax = plt.subplots(figsize=(16, max(5, len(run_names) * 0.55)))
    im = ax.imshow(data_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metric_keys], fontsize=7)
    ax.set_yticks(range(len(run_names)))
    ax.set_yticklabels(run_names, fontsize=8)

    # Annotate with raw values
    for i in range(len(run_names)):
        for j in range(len(metric_keys)):
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if 0.3 < data_norm[i, j] < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Normalised score (green=better)")
    ax.set_title("Experiment Metrics Heatmap", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# -----------------------------------------------------------------------
# 4. Detection quality across experiments
# -----------------------------------------------------------------------
def plot_detection_quality(all_results: dict, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Drift Detection Quality", fontsize=13, fontweight="bold")

    names, det_rates, far_rates, delays = [], [], [], []
    for rn, r in all_results.items():
        s = r["summary"]
        names.append(rn.replace("__", "\n"))
        det_rates.append(s.get("detection_rate", 0))
        far_rates.append(s.get("false_alarm_rate", 0))
        d = s.get("mean_detection_delay", 0)
        delays.append(d if not np.isinf(d) else 0)

    colors = ["#2196F3" if "fine" in n else "#4CAF50" if "maml" in n else "#FF9800"
              for n in names]

    axes[0].bar(range(len(names)), det_rates, color=colors, alpha=0.85)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, fontsize=6, rotation=45, ha="right")
    axes[0].set_ylabel("Detection Rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(1.0, ls="--", color="gray", alpha=0.5)
    axes[0].set_title("Detection Rate (higher=better)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(len(names)), far_rates, color=colors, alpha=0.85)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=6, rotation=45, ha="right")
    axes[1].set_ylabel("False Alarm Rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("False Alarm Rate (lower=better)")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(range(len(names)), delays, color=colors, alpha=0.85)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, fontsize=6, rotation=45, ha="right")
    axes[2].set_ylabel("Episodes")
    axes[2].set_title("Mean Detection Delay (lower=better)")
    axes[2].grid(axis="y", alpha=0.3)

    legend_patches = [
        mpatches.Patch(color="#2196F3", label="fine_tune"),
        mpatches.Patch(color="#4CAF50", label="maml"),
        mpatches.Patch(color="#FF9800", label="ensemble"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# -----------------------------------------------------------------------
# 5. Forgetting & recovery summary
# -----------------------------------------------------------------------
def plot_forgetting_recovery(all_results: dict, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Catastrophic Forgetting & Recovery", fontsize=13, fontweight="bold")

    names, forgetting, recovery = [], [], []
    for rn, r in all_results.items():
        s = r["summary"]
        names.append(rn.replace("__", "\n"))
        forgetting.append(s.get("catastrophic_forgetting", 0))
        rt = s.get("mean_recovery_time", 0)
        recovery.append(rt if not np.isinf(rt) else 300)

    colors = ["#2196F3" if "fine" in n else "#4CAF50" if "maml" in n else "#FF9800"
              for n in names]

    axes[0].bar(range(len(names)), forgetting, color=colors, alpha=0.85)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, fontsize=6, rotation=45, ha="right")
    axes[0].set_ylabel("Forgetting Score")
    axes[0].set_title("Catastrophic Forgetting (lower=better)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(len(names)), recovery, color=colors, alpha=0.85)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=6, rotation=45, ha="right")
    axes[1].set_ylabel("Episodes")
    axes[1].set_title("Recovery Time to 90% Optimal (lower=better)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# -----------------------------------------------------------------------
# Master: generate all plots from saved results
# -----------------------------------------------------------------------
def generate_all_plots(results_dir: str = "results", plots_dir: str = "results/plots"):
    os.makedirs(plots_dir, exist_ok=True)

    all_path = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(all_path):
        print(f"No all_results.json found in {results_dir}")
        return

    with open(all_path) as f:
        all_results = json.load(f)

    # Individual learning curves
    for run_name, result in all_results.items():
        plot_learning_curve(
            result,
            title=run_name,
            save_path=os.path.join(plots_dir, f"curve_{run_name}.png"),
        )

    # Strategy comparisons per environment
    env_prefixes = set()
    for key in all_results:
        parts = key.split("__")
        if len(parts) >= 2:
            env_prefixes.add(parts[0])

    for prefix in env_prefixes:
        plot_strategy_comparison(
            all_results,
            env_prefix=prefix,
            save_path=os.path.join(plots_dir, f"strategy__{prefix}.png"),
        )

    # Aggregate plots
    plot_metrics_heatmap(all_results, save_path=os.path.join(plots_dir, "metrics_heatmap.png"))
    plot_detection_quality(all_results, save_path=os.path.join(plots_dir, "detection_quality.png"))
    plot_forgetting_recovery(all_results, save_path=os.path.join(plots_dir, "forgetting_recovery.png"))

    print(f"All plots saved to {plots_dir}/")


if __name__ == "__main__":
    generate_all_plots()
