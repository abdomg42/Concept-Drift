"""Visualize a single result JSON and save plots to results/plots.

Usage:
    python scripts/visualize_result.py results/CartPole-Reward__maml.json
"""
import sys
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualize(json_path: str, out_path: str):
    with open(json_path) as f:
        data = json.load(f)

    returns = data.get("episode_returns", [])
    eps = data.get("epsilon_trace", [])
    detection_ts = data.get("detection_timestamps", []) or []
    drift_flags = data.get("drift_flags", [])
    gt_flags = data.get("ground_truth_flags", [])

    episodes = list(range(1, len(returns) + 1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    # Returns with detection/adaptation markers
    axes[0].plot(episodes, returns, label="Return")
    for d in detection_ts:
        try:
            axes[0].axvline(int(d), color="r", linestyle="--", alpha=0.7, label="Adaptation")
        except Exception:
            pass
    axes[0].set_ylabel("Return")
    axes[0].legend(loc="upper right")

    # Epsilon trace
    if eps:
        axes[1].plot(episodes, eps, label="Epsilon", color="orange")
        axes[1].set_ylabel("Epsilon")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No epsilon_trace", ha="center", va="center")

    # Drift flags: detected vs ground truth
    # Represent as binary bars
    axes[2].step(episodes, [1 if x else 0 for x in gt_flags], where="mid", label="Ground-truth", color="gray")
    axes[2].step(episodes, [1 if x else 0 for x in drift_flags], where="mid", label="Detected", color="red")
    axes[2].set_ylabel("Drift")
    axes[2].set_ylim(-0.1, 1.5)
    axes[2].legend()

    axes[2].set_xlabel("Episode")
    plt.tight_layout()
    fig.savefig(out_path)

    print("Saved:", out_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        in_file = sys.argv[1]
    else:
        in_file = os.path.join("results", "CartPole-Reward__maml.json")
    base = os.path.splitext(os.path.basename(in_file))[0]
    out_file = os.path.join("results", "plots", f"{base}.png")
    visualize(in_file, out_file)
