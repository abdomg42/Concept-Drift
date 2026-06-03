import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import copy
import json
import numpy as np
from scripts.train import train

BASE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")

EXPERIMENTS = [
    {"env_id": "cartpole_gravity",  "drift_start": 150, "run_name": "CartPole-Gravity"},
    {"env_id": "cartpole_length",   "drift_start": 150, "run_name": "CartPole-Length"},
    {"env_id": "cartpole_reward",   "drift_start": 150, "run_name": "CartPole-Reward"},
    {"env_id": "mountaincar_force", "drift_start": 100, "run_name": "MountainCar-Force",
     "n_episodes": 300, "eval": {"optimal_return": -100.0}},
]

# Adaptation strategies to compare (ablation)
STRATEGIES = ["fine_tune", "maml", "ensemble"]


def run_all(save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    with open(BASE_CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    all_results = {}

    for exp in EXPERIMENTS:
        for strategy in STRATEGIES:
            cfg = copy.deepcopy(base_cfg)
            cfg.update({k: v for k, v in exp.items() if k != "run_name"})
            if "eval" in exp:
                cfg["eval"].update(exp["eval"])
            cfg["adaptation"]["strategy"] = strategy
            run_key = f"{exp['run_name']}__{strategy}"
            print(f"\n{'='*60}")
            print(f"  Running: {run_key}")
            print(f"{'='*60}")

            try:
                result = train(cfg, run_name=run_key)
                # Store serialisable summary only
                all_results[run_key] = {
                    "summary":            result["summary"],
                    "episode_returns":    result["episode_returns"],
                    "drift_flags":        result["drift_flags"],
                    "ground_truth_flags": result["ground_truth_flags"],
                    "detection_timestamps": result["detection_timestamps"],
                    "epsilon_trace":      result["epsilon_trace"],
                    "q_variance_trace":   result["q_variance_trace"],
                }
                print(f"  Summary: {result['summary']}")

                # Save individual run
                path = os.path.join(save_dir, f"{run_key}.json")
                with open(path, "w") as f:
                    json.dump(all_results[run_key], f, indent=2)

            except Exception as e:
                print(f"  ERROR in {run_key}: {e}")
                import traceback; traceback.print_exc()

    # Save combined
    combined_path = os.path.join(save_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")
    return all_results


if __name__ == "__main__":
    run_all()
